"""Versioned structured request codec for hierarchical HONF inverse design.

Physical design ``D`` and context ``c`` are not encoded here. Request ``R`` is
an unordered set of functional tokens plus separate geometry constraints. Plan
``G`` and realized plan ``G_hat`` consume the tensorized request later. The
codec is vocabulary-parameterized so physical cases own their functional names.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .normalization import ScalarStats


RELATION_NAMES = ("upper_bound", "lower_bound", "target_range", "minimize")
RELATION_TO_ID = {name: index for index, name in enumerate(RELATION_NAMES)}
DEFAULT_PRIORITY_WEIGHTS = {1: 0.5, 2: 1.0, 3: 2.0}
MAX_REQUEST_TOKENS = 4


def _strict_keys(payload: Mapping[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} keys: {unknown}")


def _finite_float(value: Any, *, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _optional_float(value: Any, *, label: str) -> float | None:
    return None if value is None else _finite_float(value, label=label)


def _optional_pair(value: Any, *, label: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} must be null or [low, high].")
    pair = (_finite_float(value[0], label=f"{label}[0]"), _finite_float(value[1], label=f"{label}[1]"))
    if pair[0] > pair[1]:
        raise ValueError(f"{label} must be ordered low <= high.")
    return pair


def _optional_region(value: Any, *, label: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{label} must be null or [x_min,y_min,x_max,y_max].")
    region = tuple(_finite_float(item, label=f"{label}[{index}]") for index, item in enumerate(value))
    if any(item < 0.0 or item > 1.0 for item in region):
        raise ValueError(f"{label} must be normalized to [0,1].")
    if region[0] >= region[2] or region[1] >= region[3]:
        raise ValueError(f"{label} must have positive width and height.")
    return region  # type: ignore[return-value]


@dataclass(frozen=True)
class GeometryConstraints:
    """Geometry constraints carried separately from functional request tokens."""

    module_count_min: int
    module_count_max: int
    minimum_center_distance: float
    wall_clearance: float
    inlet_clearance: float
    outlet_clearance: float
    total_heat_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if int(self.module_count_min) < 0 or int(self.module_count_max) < int(self.module_count_min):
            raise ValueError("Module count bounds must satisfy 0 <= min <= max.")
        for name in (
            "minimum_center_distance",
            "wall_clearance",
            "inlet_clearance",
            "outlet_clearance",
        ):
            value = _finite_float(getattr(self, name), label=name)
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        if self.total_heat_range is not None:
            low, high = self.total_heat_range
            if not np.isfinite([low, high]).all() or float(low) > float(high):
                raise ValueError("total_heat_range must be finite and ordered.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_count_min": int(self.module_count_min),
            "module_count_max": int(self.module_count_max),
            "minimum_center_distance": float(self.minimum_center_distance),
            "wall_clearance": float(self.wall_clearance),
            "inlet_clearance": float(self.inlet_clearance),
            "outlet_clearance": float(self.outlet_clearance),
            "total_heat_range": None if self.total_heat_range is None else list(self.total_heat_range),
        }


@dataclass(frozen=True)
class RequestToken:
    """One functional request token in physical and optional normalized units."""

    request_type: str
    relation: str
    target: float | None
    tolerance: float
    target_range: tuple[float, float] | None
    priority: int
    weight: float
    region: tuple[float, float, float, float] | None
    active: bool = True
    normalized_target: float | None = None
    normalized_tolerance: float | None = None
    normalized_target_range: tuple[float, float] | None = None

    def to_dict(self, *, include_normalized: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "request_type": self.request_type,
            "relation": self.relation,
            "target": self.target,
            "tolerance": float(self.tolerance),
            "target_range": None if self.target_range is None else list(self.target_range),
            "priority": int(self.priority),
            "weight": float(self.weight),
            "region": None if self.region is None else list(self.region),
            "active": bool(self.active),
        }
        if include_normalized:
            payload.update(
                normalized_target=self.normalized_target,
                normalized_tolerance=self.normalized_tolerance,
                normalized_target_range=(
                    None if self.normalized_target_range is None else list(self.normalized_target_range)
                ),
            )
        return payload


@dataclass(frozen=True)
class StructuredRequest:
    """One versioned unordered request set and its separate geometry block."""

    schema_name: str
    schema_version: int
    tokens: tuple[RequestToken, ...]
    geometry_constraints: GeometryConstraints

    def __post_init__(self) -> None:
        if not str(self.schema_name).strip() or int(self.schema_version) <= 0:
            raise ValueError("Request schema name/version are required.")
        if len(self.tokens) > MAX_REQUEST_TOKENS:
            raise ValueError(f"A request may contain at most {MAX_REQUEST_TOKENS} tokens in total.")
        active = [token for token in self.tokens if token.active]
        if not active or len(active) > MAX_REQUEST_TOKENS:
            raise ValueError(f"A request requires 1..{MAX_REQUEST_TOKENS} active tokens.")

    def to_dict(self, *, include_normalized: bool = True) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": int(self.schema_version),
            "tokens": [token.to_dict(include_normalized=include_normalized) for token in self.tokens if token.active],
            "geometry_constraints": self.geometry_constraints.to_dict(),
        }


@dataclass(frozen=True)
class RequestTensors:
    """Fixed-width NumPy tensors for one request; batching stacks a leading axis."""

    type_id: np.ndarray
    relation_id: np.ndarray
    target_raw: np.ndarray
    target_normalized: np.ndarray
    target_mask: np.ndarray
    tolerance_raw: np.ndarray
    tolerance_normalized: np.ndarray
    range_raw: np.ndarray
    range_normalized: np.ndarray
    range_mask: np.ndarray
    priority: np.ndarray
    weight: np.ndarray
    region: np.ndarray
    region_mask: np.ndarray
    active_mask: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(getattr(self, name)) for name in self.__dataclass_fields__}


class RequestCodec:
    """Strict codec for one case-owned functional vocabulary."""

    ROOT_KEYS = {"schema_name", "schema_version", "tokens", "geometry_constraints"}
    TOKEN_KEYS = {
        "request_type",
        "relation",
        "target",
        "normalized_target",
        "tolerance",
        "normalized_tolerance",
        "target_range",
        "normalized_target_range",
        "priority",
        "weight",
        "region",
        "active",
    }
    GEOMETRY_KEYS = {
        "module_count_min",
        "module_count_max",
        "minimum_center_distance",
        "wall_clearance",
        "inlet_clearance",
        "outlet_clearance",
        "total_heat_range",
    }

    def __init__(
        self,
        *,
        schema_name: str,
        request_types: Sequence[str],
        regional_types: Sequence[str] = (),
        normalization: Mapping[str, ScalarStats] | None = None,
        priority_weights: Mapping[int, float] | None = None,
    ) -> None:
        self.schema_name = str(schema_name)
        self.request_types = tuple(str(name) for name in request_types)
        self.type_to_id = {name: index for index, name in enumerate(self.request_types)}
        self.regional_types = frozenset(str(name) for name in regional_types)
        self.normalization = dict(normalization or {})
        self.priority_weights = dict(DEFAULT_PRIORITY_WEIGHTS if priority_weights is None else priority_weights)
        if not self.schema_name or not self.request_types or len(self.type_to_id) != len(self.request_types):
            raise ValueError("Request codec requires a schema name and unique request types.")
        if not self.regional_types.issubset(self.type_to_id):
            raise ValueError("Every regional request type must belong to the vocabulary.")
        if self.normalization and set(self.normalization) != set(self.request_types):
            missing = sorted(set(self.request_types) - set(self.normalization))
            extra = sorted(set(self.normalization) - set(self.request_types))
            raise ValueError(f"Request normalization vocabulary mismatch; missing={missing}, extra={extra}.")

    def load(self, source: Mapping[str, Any] | str | Path) -> StructuredRequest:
        if isinstance(source, Mapping):
            payload = dict(source)
        else:
            path = Path(source).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Request JSON not found: {path}")
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if not isinstance(payload, dict):
                raise TypeError(f"Request JSON root must be an object: {path}")
        return self.parse(payload)

    def parse(self, payload: Mapping[str, Any]) -> StructuredRequest:
        _strict_keys(payload, self.ROOT_KEYS, label="request")
        if payload.get("schema_name") != self.schema_name:
            raise ValueError(
                f"Request schema_name={payload.get('schema_name')!r} does not match {self.schema_name!r}."
            )
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("The current request codec supports schema_version=1 only.")
        raw_tokens = payload.get("tokens")
        if not isinstance(raw_tokens, list):
            raise TypeError("request.tokens must be a JSON array.")
        tokens = tuple(self._parse_token(value, index=index) for index, value in enumerate(raw_tokens))
        active = [token for token in tokens if token.active]
        names = [token.request_type for token in active]
        if len(set(names)) != len(names):
            raise ValueError("Active request tokens must not repeat a request_type.")
        if sum(name in self.regional_types for name in names) > 1:
            raise ValueError("Schema v1 permits at most one active regional request token.")
        geometry = self._parse_geometry(payload.get("geometry_constraints"))
        request = StructuredRequest(self.schema_name, 1, tokens, geometry)
        return self.materialize_normalized(request) if self.normalization else request

    def _parse_token(self, value: Any, *, index: int) -> RequestToken:
        if not isinstance(value, Mapping):
            raise TypeError(f"request.tokens[{index}] must be an object.")
        _strict_keys(value, self.TOKEN_KEYS, label=f"request.tokens[{index}]")
        request_type = str(value.get("request_type", ""))
        relation = str(value.get("relation", ""))
        if request_type not in self.type_to_id:
            raise ValueError(f"request.tokens[{index}].request_type is unsupported: {request_type!r}")
        if relation not in RELATION_TO_ID:
            raise ValueError(f"request.tokens[{index}].relation is unsupported: {relation!r}")
        active = bool(value.get("active", True))
        target = _optional_float(value.get("target"), label=f"request.tokens[{index}].target")
        target_range = _optional_pair(value.get("target_range"), label=f"request.tokens[{index}].target_range")
        tolerance = _finite_float(value.get("tolerance", 0.0), label=f"request.tokens[{index}].tolerance")
        if tolerance < 0.0:
            raise ValueError(f"request.tokens[{index}].tolerance must be nonnegative.")
        if relation in {"upper_bound", "lower_bound"}:
            if target is None or target_range is not None:
                raise ValueError(f"request.tokens[{index}] bound relation requires target and no target_range.")
        elif relation == "target_range":
            if target_range is None:
                raise ValueError(f"request.tokens[{index}] target_range relation requires target_range.")
            midpoint = 0.5 * (target_range[0] + target_range[1])
            if target is None:
                target = midpoint
            elif not np.isclose(target, midpoint, atol=1.0e-6, rtol=1.0e-6):
                raise ValueError(f"request.tokens[{index}].target must equal the range midpoint.")
        else:
            if target is not None or target_range is not None or tolerance != 0.0:
                raise ValueError(f"request.tokens[{index}] minimize requires null target/range and zero tolerance.")
        priority = int(value.get("priority", 2))
        if priority not in (1, 2, 3):
            raise ValueError(f"request.tokens[{index}].priority must be 1, 2, or 3.")
        weight = _finite_float(
            value.get("weight", self.priority_weights[priority]),
            label=f"request.tokens[{index}].weight",
        )
        if weight <= 0.0:
            raise ValueError(f"request.tokens[{index}].weight must be positive.")
        region = _optional_region(value.get("region"), label=f"request.tokens[{index}].region")
        if request_type in self.regional_types and region is None:
            raise ValueError(f"request.tokens[{index}] regional type requires region.")
        if request_type not in self.regional_types and region is not None:
            raise ValueError(f"request.tokens[{index}] nonregional type must not define region.")
        token = RequestToken(
            request_type=request_type,
            relation=relation,
            target=target,
            tolerance=tolerance,
            target_range=target_range,
            priority=priority,
            weight=weight,
            region=region,
            active=active,
            normalized_target=_optional_float(
                value.get("normalized_target"), label=f"request.tokens[{index}].normalized_target"
            ),
            normalized_tolerance=_optional_float(
                value.get("normalized_tolerance"), label=f"request.tokens[{index}].normalized_tolerance"
            ),
            normalized_target_range=_optional_pair(
                value.get("normalized_target_range"),
                label=f"request.tokens[{index}].normalized_target_range",
            ),
        )
        return token

    def _parse_geometry(self, value: Any) -> GeometryConstraints:
        if not isinstance(value, Mapping):
            raise TypeError("request.geometry_constraints must be an object.")
        _strict_keys(value, self.GEOMETRY_KEYS, label="request.geometry_constraints")
        missing = sorted(self.GEOMETRY_KEYS - {"total_heat_range"} - set(value))
        if missing:
            raise ValueError(f"request.geometry_constraints is missing keys: {missing}")
        return GeometryConstraints(
            module_count_min=int(value["module_count_min"]),
            module_count_max=int(value["module_count_max"]),
            minimum_center_distance=_finite_float(value["minimum_center_distance"], label="minimum_center_distance"),
            wall_clearance=_finite_float(value["wall_clearance"], label="wall_clearance"),
            inlet_clearance=_finite_float(value["inlet_clearance"], label="inlet_clearance"),
            outlet_clearance=_finite_float(value["outlet_clearance"], label="outlet_clearance"),
            total_heat_range=_optional_pair(value.get("total_heat_range"), label="total_heat_range"),
        )

    def materialize_normalized(self, request: StructuredRequest) -> StructuredRequest:
        if not self.normalization:
            raise ValueError("Cannot materialize normalized requests without functional statistics.")
        tokens: list[RequestToken] = []
        for index, token in enumerate(request.tokens):
            stats = self.normalization[token.request_type]
            normalized_target = None if token.target is None else float(stats.normalize(token.target))
            normalized_tolerance = float(stats.normalize_width(token.tolerance))
            normalized_range = (
                None
                if token.target_range is None
                else tuple(float(item) for item in stats.normalize(token.target_range))
            )
            for label, supplied, expected in (
                ("normalized_target", token.normalized_target, normalized_target),
                ("normalized_tolerance", token.normalized_tolerance, normalized_tolerance),
            ):
                if supplied is not None and expected is not None and not np.isclose(
                    supplied, expected, atol=1.0e-5, rtol=1.0e-5
                ):
                    raise ValueError(f"request.tokens[{index}].{label} is inconsistent with physical values.")
            if token.normalized_target_range is not None and (
                normalized_range is None
                or not np.allclose(token.normalized_target_range, normalized_range, atol=1.0e-5, rtol=1.0e-5)
            ):
                raise ValueError(
                    f"request.tokens[{index}].normalized_target_range is inconsistent with physical values."
                )
            tokens.append(
                replace(
                    token,
                    normalized_target=normalized_target,
                    normalized_tolerance=normalized_tolerance,
                    normalized_target_range=normalized_range,
                )
            )
        return replace(request, tokens=tuple(tokens))

    def tensorize(self, request: StructuredRequest, *, max_tokens: int = MAX_REQUEST_TOKENS) -> RequestTensors:
        if max_tokens != MAX_REQUEST_TOKENS:
            raise ValueError(f"Request schema v1 uses exactly {MAX_REQUEST_TOKENS} tensor slots.")
        materialized = self.materialize_normalized(request) if any(
            token.normalized_tolerance is None for token in request.tokens
        ) else request
        result = {
            "type_id": np.full(max_tokens, -1, dtype=np.int8),
            "relation_id": np.full(max_tokens, -1, dtype=np.int8),
            "target_raw": np.zeros(max_tokens, dtype=np.float32),
            "target_normalized": np.zeros(max_tokens, dtype=np.float32),
            "target_mask": np.zeros(max_tokens, dtype=np.uint8),
            "tolerance_raw": np.zeros(max_tokens, dtype=np.float32),
            "tolerance_normalized": np.zeros(max_tokens, dtype=np.float32),
            "range_raw": np.zeros((max_tokens, 2), dtype=np.float32),
            "range_normalized": np.zeros((max_tokens, 2), dtype=np.float32),
            "range_mask": np.zeros(max_tokens, dtype=np.uint8),
            "priority": np.zeros(max_tokens, dtype=np.uint8),
            "weight": np.zeros(max_tokens, dtype=np.float32),
            "region": np.zeros((max_tokens, 4), dtype=np.float32),
            "region_mask": np.zeros(max_tokens, dtype=np.uint8),
            "active_mask": np.zeros(max_tokens, dtype=np.uint8),
        }
        active = [token for token in materialized.tokens if token.active]
        for slot, token in enumerate(active):
            result["type_id"][slot] = self.type_to_id[token.request_type]
            result["relation_id"][slot] = RELATION_TO_ID[token.relation]
            result["tolerance_raw"][slot] = token.tolerance
            result["tolerance_normalized"][slot] = float(token.normalized_tolerance)
            result["priority"][slot] = token.priority
            result["weight"][slot] = token.weight
            result["active_mask"][slot] = 1
            if token.target is not None:
                result["target_raw"][slot] = token.target
                result["target_normalized"][slot] = float(token.normalized_target)
                result["target_mask"][slot] = 1
            if token.target_range is not None:
                result["range_raw"][slot] = token.target_range
                result["range_normalized"][slot] = token.normalized_target_range
                result["range_mask"][slot] = 1
            if token.region is not None:
                result["region"][slot] = token.region
                result["region_mask"][slot] = 1
        return RequestTensors(**result)

    def summarize(self, request: StructuredRequest, *, context: Mapping[str, float] | None = None) -> str:
        lines = [f"{request.schema_name} v{request.schema_version}: {len(request.tokens)} token(s)"]
        lx = None if context is None else context.get("domain_length_x")
        ly = None if context is None else context.get("domain_length_y")
        for token in request.tokens:
            if not token.active:
                continue
            if token.relation == "target_range":
                objective = f"range {token.target_range} +/- {token.tolerance:g}"
            elif token.relation == "minimize":
                objective = "minimize"
            else:
                objective = f"{token.relation} {token.target:g} +/- {token.tolerance:g}"
            region = ""
            if token.region is not None:
                if lx is not None and ly is not None:
                    x0, y0, x1, y1 = token.region
                    region = f" region=({x0*lx:g},{y0*ly:g})-({x1*lx:g},{y1*ly:g})"
                else:
                    region = f" region_norm={token.region}"
            lines.append(
                f"- {token.request_type}: {objective}; priority={token.priority}, weight={token.weight:g}{region}"
            )
        geometry = request.geometry_constraints
        lines.append(
            "geometry: "
            f"count={geometry.module_count_min}..{geometry.module_count_max}, "
            f"min_center={geometry.minimum_center_distance:g}, wall={geometry.wall_clearance:g}, "
            f"inlet={geometry.inlet_clearance:g}, outlet={geometry.outlet_clearance:g}, "
            f"total_heat={geometry.total_heat_range}"
        )
        return "\n".join(lines)
