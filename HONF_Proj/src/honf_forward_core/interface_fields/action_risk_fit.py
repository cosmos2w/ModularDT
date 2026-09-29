"""Small train-family grouped fitting for the action-aware risk head.

Only input-only action/receiver features enter either model. Family and case
keys are used for grouped splitting and within-case ranking, never as features.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from .action_aware_frontier import (
    ActionAwareRiskHead,
    action_risk_loss,
    select_action_by_risk,
    signed_log_role_risk,
)


@dataclass(frozen=True)
class ActionEvidenceRow:
    family_key: str
    case_key: str
    action_key: str
    forward_sha256: str
    packet_rows: torch.Tensor
    budget_vector: torch.Tensor
    receiver_role_features: torch.Tensor
    candidate_role_error: torch.Tensor
    incumbent_role_error: torch.Tensor
    numerical_floor: torch.Tensor
    exact_work: float
    nonredundant_k: int
    trained_sparse: bool
    full_access: bool

    @property
    def target_log_risk(self) -> torch.Tensor:
        return signed_log_role_risk(
            self.candidate_role_error, self.incumbent_role_error, self.numerical_floor
        )


def _validate_rows(rows: Sequence[ActionEvidenceRow], current_forward_sha256: str) -> None:
    if not rows or not current_forward_sha256:
        raise ValueError("Action evidence and current forward state hash are required.")
    first = rows[0]
    if first.packet_rows.ndim != 2 or first.budget_vector.ndim != 1 or first.receiver_role_features.ndim != 2 or first.candidate_role_error.ndim != 1:
        raise ValueError("Action evidence tensors need packet, budget, receiver-role, and error axes.")
    dimensions = (
        first.packet_rows.shape[1], first.budget_vector.shape[0],
        first.receiver_role_features.shape[1], first.candidate_role_error.shape[0],
    )
    seen: set[tuple[str, str]] = set()
    families_by_case: dict[str, str] = {}
    full_by_case: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        if row.forward_sha256 != current_forward_sha256:
            raise ValueError("Action evidence is stale for the current physical/scorer weights.")
        if not all((row.family_key, row.case_key, row.action_key)):
            raise ValueError("Action evidence needs nonempty split and action keys.")
        if (row.case_key, row.action_key) in seen:
            raise ValueError("Duplicate case/action evidence row.")
        seen.add((row.case_key, row.action_key))
        if row.case_key in families_by_case and families_by_case[row.case_key] != row.family_key:
            raise ValueError("One physical case cannot appear in multiple family splits.")
        families_by_case[row.case_key] = row.family_key
        full_by_case[row.case_key] += int(row.full_access)
        if (row.packet_rows.ndim != 2 or row.packet_rows.shape[0] < 1
                or row.budget_vector.ndim != 1 or row.receiver_role_features.ndim != 2
                or row.candidate_role_error.ndim != 1
                or row.packet_rows.shape[1] != dimensions[0]
                or row.budget_vector.shape[0] != dimensions[1]
                or row.receiver_role_features.shape != (dimensions[3], dimensions[2])
                or row.incumbent_role_error.shape != (dimensions[3],)
                or row.numerical_floor.shape != (dimensions[3],)):
            raise ValueError("Action evidence feature and physical-role schemas must agree.")
        if (row.packet_rows.device != first.packet_rows.device
                or row.budget_vector.device != first.packet_rows.device
                or row.receiver_role_features.device != first.packet_rows.device
                or row.packet_rows.dtype != first.packet_rows.dtype
                or row.budget_vector.dtype != first.packet_rows.dtype
                or row.receiver_role_features.dtype != first.packet_rows.dtype):
            raise ValueError("Action evidence features must share one device and dtype.")
        if not all(torch.isfinite(value).all() for value in (
            row.packet_rows, row.budget_vector, row.receiver_role_features
        )):
            raise ValueError("Action evidence input features must be finite.")
        if (not math.isfinite(row.exact_work) or row.exact_work < 0
                or not 1 <= row.nonredundant_k <= row.packet_rows.shape[0]):
            raise ValueError("Action work and nonredundant packet count must be finite and valid.")
        if row.full_access and row.trained_sparse:
            raise ValueError("Full access must be separate from trained sparse success.")
    if any(count != 1 for count in full_by_case.values()):
        raise ValueError("Each physical case requires exactly one explicit full-access row.")


@dataclass(frozen=True)
class ActionRiskFit:
    model: ActionAwareRiskHead
    train_families: tuple[str, ...]
    train_case_keys: tuple[str, ...]
    updates: int
    first_loss: float
    last_loss: float


def fit_action_risk_head(
    rows: Sequence[ActionEvidenceRow], *, current_forward_sha256: str,
    train_families: Sequence[str], updates: int = 200, seed: int = 0,
    learning_rate: float = 3e-4,
) -> ActionRiskFit:
    """Fit only exposed actions in the named training families."""

    _validate_rows(rows, current_forward_sha256)
    if updates < 1 or learning_rate <= 0:
        raise ValueError("Action risk fit needs positive updates and learning rate.")
    allowed = set(train_families)
    if not allowed or not allowed.issubset({row.family_key for row in rows}):
        raise ValueError("Training families must exist in the measured table.")
    eligible = [
        row for row in rows
        if row.family_key in allowed and (row.trained_sparse or row.full_access)
    ]
    if not eligible:
        raise ValueError("No trained action evidence is available in the fit families.")
    for row in eligible:
        if not torch.isfinite(row.target_log_risk).all():
            raise ValueError("Action risk labels inside the fit split must be finite.")
    by_case: defaultdict[str, list[ActionEvidenceRow]] = defaultdict(list)
    for row in eligible:
        by_case[row.case_key].append(row)
    first = eligible[0]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ActionAwareRiskHead(
            packet_feature_dim=first.packet_rows.shape[1],
            budget_dim=first.budget_vector.shape[0],
            receiver_role_dim=first.receiver_role_features.shape[1],
        ).to(device=first.packet_rows.device, dtype=first.packet_rows.dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses = []
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        case_losses = []
        for case_rows in by_case.values():
            predictions = torch.stack([
                model(
                    row.packet_rows.detach(), row.budget_vector.detach(),
                    row.receiver_role_features.detach(),
                    nonredundant_k=row.nonredundant_k,
                )
                for row in case_rows
            ])
            targets = torch.stack([
                row.target_log_risk.detach().to(device=predictions.device, dtype=predictions.dtype)
                for row in case_rows
            ])
            case_losses.append(action_risk_loss(predictions, targets))
        loss = torch.stack(case_losses).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Action-risk optimization became nonfinite.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return ActionRiskFit(
        model=model.eval(),
        train_families=tuple(sorted(allowed)),
        train_case_keys=tuple(sorted(by_case)),
        updates=updates, first_loss=losses[0], last_loss=losses[-1],
    )


@dataclass(frozen=True)
class RidgeActionRiskFit:
    mean: torch.Tensor
    scale: torch.Tensor
    coefficients: torch.Tensor
    train_families: tuple[str, ...]

    def predict(self, row: ActionEvidenceRow) -> torch.Tensor:
        samples = _ridge_features(row)
        normalized = (samples - self.mean) / self.scale
        augmented = torch.cat((normalized, torch.ones_like(normalized[:, :1])), dim=1)
        return augmented @ self.coefficients


def _ridge_features(row: ActionEvidenceRow) -> torch.Tensor:
    packet = row.packet_rows
    pooled = torch.cat((
        packet.mean(dim=0), packet.sum(dim=0), packet.amax(dim=0),
        packet.new_tensor([math.log1p(row.nonredundant_k)]), row.budget_vector,
    ))
    return torch.cat((
        pooled[None].expand(row.receiver_role_features.shape[0], -1),
        row.receiver_role_features,
    ), dim=1).to(torch.float64)


def fit_ridge_action_baseline(
    rows: Sequence[ActionEvidenceRow], *, current_forward_sha256: str,
    train_families: Sequence[str], ridge: float = 1.0,
) -> RidgeActionRiskFit:
    """Train-only linear action baseline on the same grouped family split."""

    _validate_rows(rows, current_forward_sha256)
    if ridge <= 0:
        raise ValueError("Ridge strength must be positive.")
    allowed = set(train_families)
    if not allowed or not allowed.issubset({row.family_key for row in rows}):
        raise ValueError("Training families must exist in the measured table.")
    train = [
        row for row in rows
        if row.family_key in allowed and (row.trained_sparse or row.full_access)
    ]
    if not train:
        raise ValueError("No trained action evidence is available in the fit families.")
    x = torch.cat([_ridge_features(row) for row in train])
    y = torch.cat([
        row.target_log_risk.detach().to(device=x.device, dtype=torch.float64)
        for row in train
    ])
    mean = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (x - mean) / scale
    augmented = torch.cat((normalized, torch.ones_like(normalized[:, :1])), dim=1)
    identity = torch.eye(augmented.shape[1], dtype=augmented.dtype, device=augmented.device)
    identity[-1, -1] = 0.0
    coefficients = torch.linalg.solve(
        augmented.T @ augmented + ridge * identity,
        augmented.T @ y,
    )
    return RidgeActionRiskFit(mean, scale, coefficients, tuple(sorted(allowed)))


@dataclass(frozen=True)
class GroupedActionCrossfit:
    """Out-of-family risk predictions on exposed actions in the fit population."""

    row_keys: tuple[tuple[str, str], ...]
    family_keys: tuple[str, ...]
    measured: torch.Tensor  # [rows, roles]
    neural: torch.Tensor  # [rows, roles]
    ridge: torch.Tensor  # [rows, roles]
    neural_upper_margin: torch.Tensor  # empirical role residual, never formal coverage
    ridge_upper_margin: torch.Tensor


def crossfit_action_risk(
    rows: Sequence[ActionEvidenceRow], *, current_forward_sha256: str,
    train_families: Sequence[str], folds: int = 5, updates: int = 200,
    seed: int = 0, upper_quantile: float = 0.9,
) -> GroupedActionCrossfit:
    """Fit neural and ridge controls on identical family-held folds.

    Only the named fit population enters this function. Development or test
    families are evaluated later with a final fit and a margin learned here.
    """

    _validate_rows(rows, current_forward_sha256)
    families = tuple(sorted(set(train_families)))
    if (len(families) < 2 or not set(families).issubset({row.family_key for row in rows})
            or not 2 <= folds <= len(families) or not 0 < upper_quantile < 1):
        raise ValueError("Cross-fitting requires at least two named families and valid folds/quantile.")
    eligible = [
        row for row in rows
        if row.family_key in families and (row.trained_sparse or row.full_access)
    ]
    if {row.family_key for row in eligible} != set(families):
        raise ValueError("Every cross-fit family needs a trained action or full-access row.")
    neural_by_key: dict[tuple[str, str], torch.Tensor] = {}
    ridge_by_key: dict[tuple[str, str], torch.Tensor] = {}
    for fold in range(folds):
        held = {family for index, family in enumerate(families) if index % folds == fold}
        training = tuple(family for family in families if family not in held)
        neural_fit = fit_action_risk_head(
            rows, current_forward_sha256=current_forward_sha256,
            train_families=training, updates=updates, seed=seed + fold,
        )
        ridge_fit = fit_ridge_action_baseline(
            rows, current_forward_sha256=current_forward_sha256,
            train_families=training,
        )
        with torch.no_grad():
            for row in eligible:
                if row.family_key not in held:
                    continue
                key = (row.case_key, row.action_key)
                neural_by_key[key] = neural_fit.model(
                    row.packet_rows.detach(), row.budget_vector.detach(),
                    row.receiver_role_features.detach(),
                    nonredundant_k=row.nonredundant_k,
                ).detach().cpu()
                ridge_by_key[key] = ridge_fit.predict(row).detach().cpu()
    keys = tuple((row.case_key, row.action_key) for row in eligible)
    measured = torch.stack([row.target_log_risk.detach().cpu() for row in eligible])
    neural = torch.stack([neural_by_key[key] for key in keys])
    ridge = torch.stack([ridge_by_key[key] for key in keys])
    sparse_mask = torch.tensor([row.trained_sparse for row in eligible], dtype=torch.bool)
    if not bool(sparse_mask.any()):
        raise ValueError("Sparse risk calibration needs exposed sparse actions.")
    def upper_margin(predicted: torch.Tensor) -> torch.Tensor:
        # The residual is measured risk minus predicted risk. A positive value
        # means the sparse candidate was worse than the model predicted.
        residual = (measured[sparse_mask] - predicted[sparse_mask]).clamp_min(0)
        return torch.quantile(residual, upper_quantile, dim=0)
    return GroupedActionCrossfit(
        row_keys=keys,
        family_keys=tuple(row.family_key for row in eligible),
        measured=measured, neural=neural, ridge=ridge,
        neural_upper_margin=upper_margin(neural),
        ridge_upper_margin=upper_margin(ridge),
    )


@dataclass(frozen=True)
class ActionPolicyCaseResult:
    """One measured case; explicit full access stays separate from sparse success."""

    case_key: str
    family_key: str
    selected_action_key: str
    measured_oracle_action_key: str | None
    unsupported_at_budget: bool
    selected_supported_sparse: bool
    measured_adequate_sparse: tuple[str, ...]
    predicted_safe_sparse: tuple[str, ...]
    false_safe_sparse: tuple[str, ...]
    false_reject_sparse: tuple[str, ...]
    selected_work: float
    measured_oracle_sparse_work: float | None
    selected_nonredundant_k: int | None


def evaluate_action_policy(
    rows: Sequence[ActionEvidenceRow],
    predicted_log_risk: Mapping[tuple[str, str], torch.Tensor],
    *,
    current_forward_sha256: str,
    fixed_role_log_limits: torch.Tensor,
    absolute_allowance_by_case: Mapping[str, torch.Tensor],
    empirical_margin: torch.Tensor,
    relative_allowance: float = 0.10,
) -> tuple[ActionPolicyCaseResult, ...]:
    """Select with a train-fixed threshold and audit against measured oracle.

    ``fixed_role_log_limits`` must be computed once from train-only evidence
    and reused for every evaluated case. Per-case incumbent errors and physical
    allowances affect only the post-hoc measured oracle/false-choice audit;
    they never affect deployment action selection.
    """

    _validate_rows(rows, current_forward_sha256)
    limits = fixed_role_log_limits.detach().cpu().double()
    if (limits.ndim != 1 or limits.shape != rows[0].candidate_role_error.shape
            or not bool(torch.isfinite(limits).all())):
        raise ValueError("Fixed train-only role log limits must be finite and align with physical roles.")
    if not math.isfinite(relative_allowance) or relative_allowance < 0:
        raise ValueError("The measured physical relative allowance must be finite and nonnegative.")
    grouped: defaultdict[str, list[ActionEvidenceRow]] = defaultdict(list)
    for row in rows:
        grouped[row.case_key].append(row)
    if set(absolute_allowance_by_case) != set(grouped):
        raise ValueError("Every measured case needs an explicit physical-error allowance.")
    results = []
    for case_key, case_rows in grouped.items():
        baseline = case_rows[0]
        incumbent = baseline.incumbent_role_error.detach().cpu().double()
        allowance = absolute_allowance_by_case[case_key].detach().cpu().double()
        if allowance.shape != limits.shape or not bool(torch.isfinite(allowance).all()) or bool((allowance < 0).any()):
            raise ValueError("Each post-hoc physical allowance must be finite, nonnegative, and role-aligned.")
        for row in case_rows[1:]:
            if (not torch.equal(row.incumbent_role_error.detach().cpu().double(), incumbent)
                    or not torch.equal(row.numerical_floor.detach().cpu().double(), baseline.numerical_floor.detach().cpu().double())):
                raise ValueError("One case's actions must use the same measured incumbent and numerical floor.")
        if any(
            not bool(torch.isfinite(row.candidate_role_error).all())
            or bool((row.candidate_role_error < 0).any())
            for row in case_rows
        ):
            raise ValueError("Measured evaluation role errors must be finite and nonnegative.")
        full_index = next(index for index, row in enumerate(case_rows) if row.full_access)
        predictions = []
        for row in case_rows:
            key = (case_key, row.action_key)
            if key in predicted_log_risk:
                prediction = predicted_log_risk[key].detach().cpu().double()
            elif row.trained_sparse or row.full_access:
                raise ValueError(f"Missing trained-action prediction for {key!r}.")
            else:
                prediction = torch.full_like(limits, torch.inf)
            if prediction.shape != limits.shape or (row.trained_sparse or row.full_access) and not bool(torch.isfinite(prediction).all()):
                raise ValueError("Action predictions must be finite and align with physical roles.")
            predictions.append(prediction)
        # Untrained actions receive a finite placeholder because the selector
        # excludes them with the explicit trained_sparse mask.
        prediction_matrix = torch.stack([
            torch.zeros_like(limits) if not row.trained_sparse and not row.full_access else prediction
            for row, prediction in zip(case_rows, predictions, strict=True)
        ])
        selection = select_action_by_risk(
            prediction_matrix,
            torch.tensor([row.exact_work for row in case_rows], dtype=torch.float64),
            torch.tensor([row.nonredundant_k for row in case_rows]),
            torch.tensor([row.trained_sparse for row in case_rows]),
            limits,
            full_access_index=full_index,
            empirical_margin=empirical_margin.detach().cpu().double(),
        )
        measured_safe = {
            row.action_key for row in case_rows
            if row.trained_sparse and bool((
                row.candidate_role_error.detach().cpu().double()
                <= (1.0 + relative_allowance) * incumbent + allowance
            ).all())
        }
        predicted_safe = {
            case_rows[index].action_key for index in selection.safe_sparse_indices
        }
        oracle = min(
            (row for row in case_rows if row.action_key in measured_safe),
            key=lambda row: (row.exact_work, row.nonredundant_k, row.action_key),
            default=None,
        )
        chosen = case_rows[selection.index]
        results.append(ActionPolicyCaseResult(
            case_key=case_key, family_key=baseline.family_key,
            selected_action_key=chosen.action_key,
            measured_oracle_action_key=None if oracle is None else oracle.action_key,
            unsupported_at_budget=selection.unsupported_at_budget,
            selected_supported_sparse=(chosen.action_key in measured_safe),
            measured_adequate_sparse=tuple(sorted(measured_safe)),
            predicted_safe_sparse=tuple(sorted(predicted_safe)),
            false_safe_sparse=tuple(sorted(predicted_safe - measured_safe)),
            false_reject_sparse=tuple(sorted(measured_safe - predicted_safe)),
            selected_work=chosen.exact_work,
            measured_oracle_sparse_work=None if oracle is None else oracle.exact_work,
            selected_nonredundant_k=None if selection.unsupported_at_budget else chosen.nonredundant_k,
        ))
    return tuple(results)


__all__ = [
    "ActionEvidenceRow",
    "ActionPolicyCaseResult",
    "ActionRiskFit",
    "GroupedActionCrossfit",
    "RidgeActionRiskFit",
    "crossfit_action_risk",
    "evaluate_action_policy",
    "fit_action_risk_head",
    "fit_ridge_action_baseline",
]
