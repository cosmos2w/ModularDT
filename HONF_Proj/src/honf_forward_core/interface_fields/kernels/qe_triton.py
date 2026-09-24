"""Optional Triton selected-QE reader.

This module contains the first bounded fused kernel for the routed QE
operator.  It fuses projected query/key/value loading, the query-key score,
learned geometry bias, live log prior, stable normalization, and weighted
value accumulation.  The small geometry network is intentionally outside the
kernel: callers evaluate it on the retained scalar pairs and pass its
``[I, heads]`` result for CSR rows (or a bounded dense tile for complete rows).

    The custom autograd wrapper implements the corresponding first-order
    backward. It recomputes the score tiles and atomically reduces source-key,
    source-value, pair-bias, live-prior, and optional score-multiplier
    derivatives. ``grad_query`` is row-owned and written once. This is an execution primitive, so it returns
unprojected ``[B, heads, Q, head_dim]`` contexts; the caller retains the
historical output projection and any mixed complete/partial row assembly.

The scalar score path is evaluated in FP64, including ``log(prior)`` and the
softmax normalizer.  FP32 Q/K/V/bias remain FP32 neural tensors and FP32
outputs are restored after value accumulation.  No TF32, reduced precision,
probability floor, or fast-math switch is enabled here.
"""

from __future__ import annotations

from typing import Literal

import torch

try:  # Triton is optional; CPU imports must remain usable.
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except (ImportError, ModuleNotFoundError):  # pragma: no cover - host dependent
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False


def is_triton_qe_available(device: torch.device | str | None = None) -> bool:
    """Return whether the fused QE path can execute on ``device``.

    The function is intentionally cheap and side-effect free apart from the
    normal PyTorch CUDA availability query.  A CPU call is always false even
    when Triton is installed, allowing configuration code to select its
    readable Torch fallback without importing CUDA-only code.
    """

    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        return False
    if device is None:
        return True
    return torch.device(device).type == "cuda"


def _next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _zero_context(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    prior: torch.Tensor,
    score_multiplier: torch.Tensor,
    batch: int,
    heads: int,
    queries: int,
    head_dim: int,
) -> torch.Tensor:
    """Return an empty-reader result while preserving differentiable inputs."""

    # Include all floating inputs in a zero scalar.  This keeps the custom
    # reader's empty-row result connected to autograd while yielding exactly
    # zero gradients for every input, as the mathematical empty reduction
    # requires.
    connected = (
        query.sum() * 0.0
        + key.sum() * 0.0
        + value.sum() * 0.0
        + bias.sum() * 0.0
        + prior.sum() * 0.0
        + score_multiplier.sum() * 0.0
    ).to(dtype=value.dtype)
    return connected.reshape(1, 1, 1, 1).expand(batch, heads, queries, head_dim)


def _validate_common(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if not is_triton_qe_available(query.device):
        raise RuntimeError(
            "fused_qe_reader requires a CUDA device with an importable Triton; "
            "use qe_reader_reference or the Torch QE reader on this device."
        )
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B, heads, count, head_dim].")
    if query.dtype not in (torch.float32, torch.float64):
        raise TypeError("fused_qe_reader preserves FP32/FP64 neural inputs; other dtypes are unsupported.")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise TypeError("query, key, and value must use one common FP32 or FP64 dtype.")
    if key.device != query.device or value.device != query.device:
        raise ValueError("query, key, and value must share one CUDA device.")
    if bias.dtype != query.dtype or bias.device != query.device:
        raise TypeError("geometry bias must share the neural dtype and CUDA device with query.")
    batch, heads, queries, head_dim = (int(v) for v in query.shape)
    if tuple(key.shape[:2]) != (batch, heads) or tuple(value.shape[:2]) != (batch, heads):
        raise ValueError("query/key/value batch and head dimensions must match.")
    source_count = int(key.shape[2])
    if int(value.shape[2]) != source_count or int(key.shape[3]) != head_dim or int(value.shape[3]) != head_dim:
        raise ValueError("key/value source and head dimensions must match query.")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive.")
    return batch, heads, queries, source_count, head_dim


if TRITON_AVAILABLE:

    @triton.jit
    def _qe_forward_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        bias_ptr,
        prior_ptr,
        score_multiplier_ptr,
        row_offsets_ptr,
        source_indices_ptr,
        source_mask_ptr,
        output_ptr,
        lse_ptr,
        batch,
        heads,
        queries,
        sources,
        head_dim,
        max_items,
        score_scale,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        MODE_COMPLETE: tl.constexpr,
        HAS_SOURCE_MASK: tl.constexpr,
        HAS_SCORE_MULTIPLIER: tl.constexpr,
        USE_FP32_SCALAR: tl.constexpr,
        ACC_FP64: tl.constexpr,
    ):
        """One program computes one (batch, head, receiver) row."""

        pid = tl.program_id(0)
        head = pid % heads
        query_index = (pid // heads) % queries
        batch_index = pid // (heads * queries)
        row = batch_index * queries + query_index

        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim
        q_base = (batch_index * heads + head) * queries * head_dim + query_index * head_dim
        q = tl.load(query_ptr + q_base + d_offsets, mask=d_mask, other=0.0)
        if ACC_FP64:
            q_acc = q.to(tl.float64)
            accumulator = tl.zeros((BLOCK_D,), dtype=tl.float64)
        else:
            q_acc = q.to(tl.float32)
            accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

        if MODE_COMPLETE:
            row_start = 0
            row_end = sources
        else:
            row_start = tl.load(row_offsets_ptr + row)
            row_end = tl.load(row_offsets_ptr + row + 1)

        if USE_FP32_SCALAR:
            maximum = tl.full((), -float("inf"), tl.float32)
            denominator = tl.zeros((), dtype=tl.float32)
        else:
            maximum = tl.full((), -float("inf"), tl.float64)
            denominator = tl.zeros((), dtype=tl.float64)
        for start in tl.range(0, max_items, BLOCK_N):
            item_offsets = start + tl.arange(0, BLOCK_N)
            if MODE_COMPLETE:
                source = item_offsets
                active = item_offsets < sources
                if HAS_SOURCE_MASK:
                    active = active & tl.load(
                        source_mask_ptr + batch_index * sources + source,
                        mask=active,
                        other=0,
                    )
                prior_offset = (batch_index * queries + query_index) * sources + source
                bias_offset = ((batch_index * heads + head) * queries + query_index) * sources + source
            else:
                entry = row_start + item_offsets
                active = entry < row_end
                source = tl.load(source_indices_ptr + entry, mask=active, other=0)
                prior_offset = entry
                bias_offset = entry * heads + head

            source_mask = active & (source < sources)
            key_base = (batch_index * heads + head) * sources * head_dim + source * head_dim
            k = tl.load(
                key_ptr + key_base[:, None] + d_offsets[None, :],
                mask=source_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            dot = tl.sum(q_acc[None, :] * k.to(tl.float64 if ACC_FP64 else tl.float32), axis=1)
            bias_value = tl.load(bias_ptr + bias_offset, mask=source_mask, other=0.0)
            prior_value = tl.load(prior_ptr + prior_offset, mask=source_mask, other=1.0)
            if HAS_SCORE_MULTIPLIER:
                multiplier_value = tl.load(
                    score_multiplier_ptr + bias_offset,
                    mask=source_mask,
                    other=1.0,
                )
            else:
                multiplier_value = 1.0
            # Preserve the historical FP32 neural rounding before promoting
            # the scalar score to FP64.  The prior log and softmax remain
            # scalar FP64 operations.
            if ACC_FP64:
                neural_score = (
                    dot.to(tl.float64)
                    * score_scale
                    * multiplier_value.to(tl.float64)
                    + bias_value.to(tl.float64)
                )
            elif USE_FP32_SCALAR:
                neural_score = dot * score_scale * multiplier_value + bias_value
            else:
                neural_score = (
                    dot * score_scale * multiplier_value + bias_value
                ).to(tl.float64)
            if USE_FP32_SCALAR:
                score = neural_score + tl.log(prior_value)
            else:
                score = neural_score + tl.log(prior_value.to(tl.float64))
            score = tl.where(source_mask, score, -float("inf"))

            tile_maximum = tl.max(score, axis=0)
            new_maximum = tl.maximum(maximum, tile_maximum)
            both_empty = (maximum == -float("inf")) & (tile_maximum == -float("inf"))
            old_scale = tl.where(both_empty, 0.0, tl.exp(maximum - new_maximum))
            tile_weight = tl.where(source_mask, tl.exp(score - new_maximum), 0.0)
            denominator = old_scale * denominator + tl.sum(tile_weight, axis=0)

            v = tl.load(
                value_ptr + key_base[:, None] + d_offsets[None, :],
                mask=source_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            if ACC_FP64:
                accumulator = old_scale.to(tl.float64) * accumulator + tl.sum(tile_weight[:, None] * v.to(tl.float64), axis=0)
            elif USE_FP32_SCALAR:
                accumulator = old_scale * accumulator + tl.sum(
                    tile_weight[:, None] * v.to(tl.float32),
                    axis=0,
                )
            else:
                accumulator = old_scale.to(tl.float32) * accumulator + tl.sum(tile_weight[:, None].to(tl.float32) * v.to(tl.float32), axis=0)
            maximum = tl.where(both_empty, maximum, new_maximum)

        has_mass = denominator > 0.0
        if ACC_FP64:
            result = tl.where(has_mass, accumulator / denominator, 0.0).to(tl.float64)
        else:
            result = tl.where(has_mass, accumulator / denominator, 0.0).to(tl.float32)
        output_base = ((batch_index * heads + head) * queries + query_index) * head_dim
        tl.store(output_ptr + output_base + d_offsets, result, mask=d_mask)
        lse = tl.where(has_mass, maximum + tl.log(denominator), -float("inf"))
        tl.store(lse_ptr + (batch_index * heads + head) * queries + query_index, lse)


    @triton.jit
    def _qe_backward_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        bias_ptr,
        prior_ptr,
        score_multiplier_ptr,
        row_offsets_ptr,
        source_indices_ptr,
        source_mask_ptr,
        output_ptr,
        lse_ptr,
        grad_output_ptr,
        grad_query_ptr,
        grad_key_ptr,
        grad_value_ptr,
        grad_bias_ptr,
        grad_prior_ptr,
        grad_score_multiplier_ptr,
        batch,
        heads,
        queries,
        sources,
        head_dim,
        max_items,
        score_scale,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        MODE_COMPLETE: tl.constexpr,
        HAS_SOURCE_MASK: tl.constexpr,
        HAS_SCORE_MULTIPLIER: tl.constexpr,
        USE_FP32_SCALAR: tl.constexpr,
        ACC_FP64: tl.constexpr,
    ):
        """Recompute one row and reduce its first-order gradients."""

        pid = tl.program_id(0)
        head = pid % heads
        query_index = (pid // heads) % queries
        batch_index = pid // (heads * queries)
        row = batch_index * queries + query_index
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < head_dim
        q_base = (batch_index * heads + head) * queries * head_dim + query_index * head_dim
        q = tl.load(query_ptr + q_base + d_offsets, mask=d_mask, other=0.0)
        grad_output = tl.load(grad_output_ptr + q_base + d_offsets, mask=d_mask, other=0.0)
        output = tl.load(output_ptr + q_base + d_offsets, mask=d_mask, other=0.0)
        lse = tl.load(lse_ptr + (batch_index * heads + head) * queries + query_index)
        # Triton 3.2 has no ``tl.isfinite`` helper.  Empty rows are encoded
        # by the forward pass as exactly ``-inf``; positive-prior rows have a
        # finite log normalizer under the validated input contract.
        valid_row = lse != -float("inf")
        safe_lse = tl.where(valid_row, lse, 0.0)
        if ACC_FP64:
            grad_query = tl.zeros((BLOCK_D,), dtype=tl.float64)
        else:
            grad_query = tl.zeros((BLOCK_D,), dtype=tl.float32)

        if MODE_COMPLETE:
            row_start = 0
            row_end = sources
        else:
            row_start = tl.load(row_offsets_ptr + row)
            row_end = tl.load(row_offsets_ptr + row + 1)

        for start in tl.range(0, max_items, BLOCK_N):
            item_offsets = start + tl.arange(0, BLOCK_N)
            if MODE_COMPLETE:
                source = item_offsets
                active = item_offsets < sources
                if HAS_SOURCE_MASK:
                    active = active & tl.load(
                        source_mask_ptr + batch_index * sources + source,
                        mask=active,
                        other=0,
                    )
                prior_offset = (batch_index * queries + query_index) * sources + source
                bias_offset = ((batch_index * heads + head) * queries + query_index) * sources + source
            else:
                entry = row_start + item_offsets
                active = entry < row_end
                source = tl.load(source_indices_ptr + entry, mask=active, other=0)
                prior_offset = entry
                bias_offset = entry * heads + head
            source_mask = active & (source < sources) & valid_row
            key_base = (batch_index * heads + head) * sources * head_dim + source * head_dim
            k = tl.load(
                key_ptr + key_base[:, None] + d_offsets[None, :],
                mask=source_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            v = tl.load(
                value_ptr + key_base[:, None] + d_offsets[None, :],
                mask=source_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            bias_value = tl.load(bias_ptr + bias_offset, mask=source_mask, other=0.0)
            prior_value = tl.load(prior_ptr + prior_offset, mask=source_mask, other=1.0)
            if HAS_SCORE_MULTIPLIER:
                multiplier_value = tl.load(
                    score_multiplier_ptr + bias_offset,
                    mask=source_mask,
                    other=1.0,
                )
            else:
                multiplier_value = 1.0
            if ACC_FP64:
                query_key_dot = tl.sum(q.to(tl.float64)[None, :] * k.to(tl.float64), axis=1)
                neural_score = (
                    query_key_dot * score_scale * multiplier_value.to(tl.float64)
                    + bias_value.to(tl.float64)
                )
            else:
                query_key_dot = tl.sum(q.to(tl.float32)[None, :] * k.to(tl.float32), axis=1)
                if USE_FP32_SCALAR:
                    neural_score = (
                        query_key_dot * score_scale * multiplier_value + bias_value
                    )
                else:
                    neural_score = (
                        query_key_dot * score_scale * multiplier_value + bias_value
                    ).to(tl.float64)
            if USE_FP32_SCALAR:
                score = neural_score + tl.log(prior_value)
                weight = tl.where(source_mask, tl.exp(score - safe_lse), 0.0)
                value_difference = v.to(tl.float32) - output.to(tl.float32)[None, :]
                delta = weight * tl.sum(
                    value_difference * grad_output.to(tl.float32)[None, :],
                    axis=1,
                )
            else:
                score = neural_score + tl.log(prior_value.to(tl.float64))
                weight = tl.where(source_mask, tl.exp(score - safe_lse), 0.0)
                value_difference = v.to(tl.float64) - output.to(tl.float64)[None, :]
                delta = weight * tl.sum(
                    value_difference * grad_output.to(tl.float64)[None, :],
                    axis=1,
                )
            delta = tl.where(source_mask, delta, 0.0)
            if ACC_FP64:
                delta_acc = delta
                grad_query += tl.sum(
                    (delta[:, None] * multiplier_value.to(tl.float64)[:, None]
                     * k.to(tl.float64)) * score_scale,
                    axis=0,
                )
            else:
                delta_acc = delta.to(tl.float32)
                grad_query += tl.sum(
                    (delta_acc[:, None] * multiplier_value.to(tl.float32)[:, None]
                     * k.to(tl.float32)) * score_scale,
                    axis=0,
                )

            # Every source can be read by many receiver rows; atomics are
            # deliberate here.  The caller can choose the Torch reader when a
            # deterministic reduction is required by its execution policy.
            tl.atomic_add(
                grad_key_ptr + key_base[:, None] + d_offsets[None, :],
                (delta_acc[:, None]
                 * multiplier_value.to(tl.float64 if ACC_FP64 else tl.float32)[:, None]
                 * q.to(tl.float64 if ACC_FP64 else tl.float32)[None, :]).to(
                     tl.float64 if ACC_FP64 else tl.float32
                 ) * score_scale,
                mask=source_mask[:, None] & d_mask[None, :],
            )
            tl.atomic_add(
                grad_value_ptr + key_base[:, None] + d_offsets[None, :],
                (weight[:, None].to(tl.float64 if ACC_FP64 else tl.float32) * grad_output.to(tl.float64 if ACC_FP64 else tl.float32)[None, :]),
                mask=source_mask[:, None] & d_mask[None, :],
            )
            tl.atomic_add(
                grad_bias_ptr + bias_offset,
                delta,
                mask=source_mask,
            )
            if HAS_SCORE_MULTIPLIER:
                tl.atomic_add(
                    grad_score_multiplier_ptr + bias_offset,
                    delta * query_key_dot * score_scale,
                    mask=source_mask,
                )
            # dL/dPi = exp(neural_score - logsumexp(neural_score+log Pi))
            # * <g,V-o>.  This is algebraically delta/Pi but avoids an
            # unnecessary division by a tiny live prior.
            if USE_FP32_SCALAR:
                prior_gradient = tl.exp(neural_score - safe_lse) * tl.sum(
                    value_difference * grad_output.to(tl.float32)[None, :],
                    axis=1,
                )
            else:
                prior_gradient = tl.exp(neural_score - safe_lse) * tl.sum(
                    value_difference * grad_output.to(tl.float64)[None, :], axis=1
                )
            tl.atomic_add(grad_prior_ptr + prior_offset, prior_gradient, mask=source_mask)

        tl.store(grad_query_ptr + q_base + d_offsets, grad_query, mask=d_mask)


def _launch_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    prior: torch.Tensor,
    score_multiplier: torch.Tensor,
    *,
    mode: Literal["complete", "csr"],
    row_offsets: torch.Tensor | None = None,
    source_indices: torch.Tensor | None = None,
    source_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, queries, sources, head_dim = _validate_common(query, key, value, bias)
    if prior.device != query.device:
        raise ValueError("live QE priors must share the projected tensor device.")
    if prior.dtype not in (torch.float32, torch.float64):
        raise TypeError("live QE priors must use FP32 or FP64 scalar precision.")
    if score_multiplier.device != query.device or score_multiplier.dtype != query.dtype:
        raise TypeError("score multipliers must share the neural dtype and device.")
    has_score_multiplier = int(score_multiplier.numel()) > 0
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    bias = bias.contiguous()
    prior = prior.contiguous()
    score_multiplier = score_multiplier.contiguous()
    if mode == "complete":
        if tuple(prior.shape) != (batch, queries, sources):
            raise ValueError("complete prior must have shape [B,Q,N].")
        if tuple(bias.shape) != (batch, heads, queries, sources):
            raise ValueError("complete bias must have shape [B,heads,Q,N].")
        if has_score_multiplier and tuple(score_multiplier.shape) != tuple(bias.shape):
            raise ValueError("complete score_multiplier must match bias shape [B,heads,Q,N].")
        if source_mask is None or source_mask.numel() == 0:
            source_mask = torch.empty((0,), device=query.device, dtype=torch.bool)
        elif source_mask.device != query.device or source_mask.dtype != torch.bool or tuple(source_mask.shape) != (batch, sources):
            raise ValueError("complete source_mask must have shape [B,N] and bool CUDA dtype.")
        else:
            source_mask = source_mask.contiguous()
        # The routed compiler supplies the authoritative positive support.  Do
        # not scan a full dense prior on every receiver chunk just to repeat a
        # check that would synchronize the host; callers that construct raw
        # inputs should use ``qe_reader_reference`` for validating fixtures.
        if batch == 0 or queries == 0 or sources == 0:
            return _zero_context(
                query, key, value, bias, prior, score_multiplier, batch, heads, queries, head_dim
            ), prior.new_full(
                (batch, heads, queries), -torch.inf
            )
        row_offsets = torch.empty((0,), device=query.device, dtype=torch.long)
        source_indices = torch.empty((0,), device=query.device, dtype=torch.long)
        max_items = sources
    else:
        if row_offsets is None or source_indices is None:
            raise ValueError("CSR mode requires row_offsets and source_indices.")
        if source_mask is not None:
            raise ValueError("CSR mode does not accept a complete-row source mask.")
        source_mask = torch.empty((0,), device=query.device, dtype=torch.bool)
        if prior.ndim != 1 or bias.ndim != 2 or int(bias.shape[1]) != heads:
            raise ValueError("CSR prior must be [I] and CSR bias must be [I,heads].")
        if has_score_multiplier and tuple(score_multiplier.shape) != tuple(bias.shape):
            raise ValueError("CSR score_multiplier must match bias shape [I,heads].")
        if row_offsets.ndim != 1 or int(row_offsets.numel()) != batch * queries + 1:
            raise ValueError("CSR row_offsets must have shape [B*Q+1].")
        if source_indices.ndim != 1 or int(source_indices.numel()) != int(prior.numel()):
            raise ValueError("CSR source_indices and prior must have the same length.")
        if int(row_offsets[-1]) != int(prior.numel()) or bool(torch.any(row_offsets[1:] < row_offsets[:-1])):
            raise ValueError("CSR row_offsets must be nondecreasing and terminate at I.")
        # Source indices and priors are supplied by the exact CSR compiler.
        # Their positivity/range contract is deliberately trusted here to
        # avoid a second full-device scan before launching the kernel.
        row_offsets = row_offsets.contiguous()
        source_indices = source_indices.contiguous()
        max_items = int((row_offsets[1:] - row_offsets[:-1]).max()) if int(row_offsets.numel()) > 1 else 0
        if batch == 0 or queries == 0 or max_items == 0 or sources == 0:
            return _zero_context(
                query, key, value, bias, prior, score_multiplier, batch, heads, queries, head_dim
            ), prior.new_full(
                (batch, heads, queries), -torch.inf
            )

    output = torch.empty_like(query)
    use_fp32_scalar = bool(
        has_score_multiplier
        and query.dtype == torch.float32
        and prior.dtype == torch.float32
    )
    lse_dtype = torch.float32 if use_fp32_scalar else torch.float64
    lse = torch.empty((batch, heads, queries), dtype=lse_dtype, device=query.device)
    block_d = _next_power_of_two(head_dim)
    # Keep the source tile independent from the feature tile.  128 is large
    # enough for the current 192-source environmental banks while avoiding a
    # compile-time unrolled loop over the full receiver grid.
    block_n = 128 if max_items >= 128 else 64 if max_items >= 64 else _next_power_of_two(max_items)
    _qe_forward_kernel[(batch * heads * queries,)](
        query,
        key,
        value,
        bias,
        prior,
        score_multiplier,
        row_offsets,
        source_indices,
        source_mask,
        output,
        lse,
        batch,
        heads,
        queries,
        sources,
        head_dim,
        max_items,
        1.0 / (float(head_dim) ** 0.5),
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        MODE_COMPLETE=mode == "complete",
        HAS_SOURCE_MASK=source_mask.numel() != 0,
        HAS_SCORE_MULTIPLIER=has_score_multiplier,
        USE_FP32_SCALAR=use_fp32_scalar,
        ACC_FP64=query.dtype == torch.float64,
        num_warps=4 if block_d <= 64 else 8,
    )
    return output, lse


def _launch_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    prior: torch.Tensor,
    score_multiplier: torch.Tensor,
    row_offsets: torch.Tensor,
    source_indices: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    mode: Literal["complete", "csr"],
    source_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, queries, sources, head_dim = _validate_common(query, key, value, bias)
    if prior.device != query.device:
        raise ValueError("live QE priors must share the projected tensor device.")
    if prior.dtype not in (torch.float32, torch.float64):
        raise TypeError("live QE priors must use FP32 or FP64 scalar precision.")
    if score_multiplier.device != query.device or score_multiplier.dtype != query.dtype:
        raise TypeError("score multipliers must share the neural dtype and device.")
    has_score_multiplier = int(score_multiplier.numel()) > 0
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    bias = bias.contiguous()
    prior = prior.contiguous()
    score_multiplier = score_multiplier.contiguous()
    if has_score_multiplier and tuple(score_multiplier.shape) != tuple(bias.shape):
        raise ValueError("score_multiplier must match the complete or CSR bias shape.")
    if mode == "complete":
        if source_mask is None or source_mask.numel() == 0:
            source_mask = torch.empty((0,), device=query.device, dtype=torch.bool)
        elif source_mask.device != query.device or source_mask.dtype != torch.bool or tuple(source_mask.shape) != (batch, sources):
            raise ValueError("complete source_mask must have shape [B,N] and bool CUDA dtype.")
        else:
            source_mask = source_mask.contiguous()
    else:
        source_mask = torch.empty((0,), device=query.device, dtype=torch.bool)
    output = output.contiguous()
    lse = lse.contiguous()
    grad_output = grad_output.contiguous()
    if mode == "complete":
        max_items = sources
    else:
        max_items = int((row_offsets[1:] - row_offsets[:-1]).max()) if int(row_offsets.numel()) > 1 else 0
    grad_query = torch.zeros_like(query)
    grad_key = torch.zeros_like(key)
    grad_value = torch.zeros_like(value)
    grad_bias = torch.zeros_like(bias)
    grad_prior = torch.zeros_like(prior)
    grad_score_multiplier = torch.zeros_like(score_multiplier)
    if max_items == 0 or sources == 0:
        return grad_query, grad_key, grad_value, grad_bias, grad_prior, grad_score_multiplier
    use_fp32_scalar = bool(
        has_score_multiplier
        and query.dtype == torch.float32
        and prior.dtype == torch.float32
    )
    block_d = _next_power_of_two(head_dim)
    block_n = 128 if max_items >= 128 else 64 if max_items >= 64 else _next_power_of_two(max_items)
    _qe_backward_kernel[(batch * heads * queries,)](
        query,
        key,
        value,
        bias,
        prior,
        score_multiplier,
        row_offsets,
        source_indices,
        source_mask,
        output,
        lse,
        grad_output,
        grad_query,
        grad_key,
        grad_value,
        grad_bias,
        grad_prior,
        grad_score_multiplier,
        batch,
        heads,
        queries,
        sources,
        head_dim,
        max_items,
        1.0 / (float(head_dim) ** 0.5),
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        MODE_COMPLETE=mode == "complete",
        HAS_SOURCE_MASK=source_mask.numel() != 0,
        HAS_SCORE_MULTIPLIER=has_score_multiplier,
        USE_FP32_SCALAR=use_fp32_scalar,
        ACC_FP64=query.dtype == torch.float64,
        num_warps=4 if block_d <= 64 else 8,
    )
    return grad_query, grad_key, grad_value, grad_bias, grad_prior, grad_score_multiplier


class _CompleteQEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor,
        prior: torch.Tensor,
        score_multiplier: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        output, lse = _launch_forward(
            query,
            key,
            value,
            bias,
            prior,
            score_multiplier,
            mode="complete",
            source_mask=source_mask,
        )
        empty_offsets = torch.empty((0,), device=query.device, dtype=torch.long)
        empty_indices = torch.empty((0,), device=query.device, dtype=torch.long)
        # Keep the original tensors for the higher-order Torch fallback.  The
        # first-order CUDA path makes contiguous working copies internally,
        # while saving those copies here would sever a non-contiguous caller's
        # second-derivative graph.
        ctx.save_for_backward(
            query,
            key,
            value,
            bias,
            prior,
            score_multiplier,
            empty_offsets,
            empty_indices,
            source_mask,
            output,
            lse,
        )
        ctx.mode = "complete"
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query, key, value, bias, prior, score_multiplier, row_offsets, source_indices, source_mask, output, lse = ctx.saved_tensors
        if torch.is_grad_enabled():
            # The Triton primitive intentionally implements first-order
            # backward.  A create_graph request is routed through the
            # differentiable Torch oracle so inverse-design callers never
            # receive silently disconnected second derivatives.
            return (
                *_reference_backward(
                    query,
                    key,
                    value,
                    bias,
                    prior,
                    score_multiplier,
                    grad_output,
                    mode="complete",
                    source_mask=source_mask,
                ),
                None,
            )
        return (
            *_launch_backward(
                query,
                key,
                value,
                bias,
                prior,
                score_multiplier,
                row_offsets,
                source_indices,
                output,
                lse,
                grad_output,
                mode="complete",
                source_mask=source_mask,
            ),
            None,
        )


class _CSRQEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor,
        prior: torch.Tensor,
        score_multiplier: torch.Tensor,
        row_offsets: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> torch.Tensor:
        output, lse = _launch_forward(
            query,
            key,
            value,
            bias,
            prior,
            score_multiplier,
            mode="csr",
            row_offsets=row_offsets,
            source_indices=source_indices,
        )
        ctx.save_for_backward(
            query,
            key,
            value,
            bias,
            prior,
            score_multiplier,
            row_offsets,
            source_indices,
            output,
            lse,
        )
        ctx.mode = "csr"
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query, key, value, bias, prior, score_multiplier, row_offsets, source_indices, output, lse = ctx.saved_tensors
        if torch.is_grad_enabled():
            gradients = _reference_backward(
                query,
                key,
                value,
                bias,
                prior,
                score_multiplier,
                grad_output,
                mode="csr",
                row_offsets=row_offsets,
                source_indices=source_indices,
            )
            return (*gradients, None, None)
        gradients = _launch_backward(
            query,
            key,
            value,
            bias,
            prior,
            score_multiplier,
            row_offsets,
            source_indices,
            output,
            lse,
            grad_output,
            mode="csr",
        )
        return (*gradients, None, None)


def fused_qe_reader(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    prior: torch.Tensor,
    *,
    score_multiplier: torch.Tensor | None = None,
    mode: Literal["complete", "csr"] = "csr",
    row_offsets: torch.Tensor | None = None,
    source_indices: torch.Tensor | None = None,
    source_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the optional fused QE reader.

    Parameters use projected attention-head layouts:

    * ``query``: ``[B, heads, Q, D]``;
    * ``key``/``value``: ``[B, heads, N, D]``;
    * complete mode: ``bias=[B,heads,Q,N]``, ``prior=[B,Q,N]`` and no row
      index tensors.  An optional ``source_mask=[B,N]`` keeps padded/invalid
      source slots implicit while excluding them from every row;
    * CSR mode: ``bias=[I,heads]``, optional ``score_multiplier=[I,heads]``,
      ``prior=[I]``, ``row_offsets=[B*Q+1]`` and ``source_indices=[I]``.
      Rows are in flattened ``(B,Q)`` order; an empty row has equal adjacent
      offsets.

    The result is the unprojected ``[B,heads,Q,D]`` context.  Complete mode
    avoids any source-index list; CSR mode loads only its row's source index.
    The function raises an explicit ``RuntimeError`` on CPU or when Triton is
    unavailable so configuration code can record a Torch fallback.
    """

    if mode not in {"complete", "csr"}:
        raise ValueError("mode must be 'complete' or 'csr'.")
    if score_multiplier is None:
        score_multiplier = query.new_empty((0,))
    if mode == "complete":
        if row_offsets is not None or source_indices is not None:
            raise ValueError("complete mode does not accept CSR index tensors.")
        if source_mask is None:
            source_mask = torch.empty((0,), device=query.device, dtype=torch.bool)
        return _CompleteQEFunction.apply(
            query,
            key,
            value,
            bias,
            prior,
            score_multiplier,
            source_mask,
        )
    if row_offsets is None or source_indices is None:
        raise ValueError("CSR mode requires row_offsets and source_indices.")
    if source_mask is not None:
        raise ValueError("CSR mode does not accept a complete-row source mask.")
    if row_offsets.device != query.device or source_indices.device != query.device:
        raise ValueError("CSR indices must share the projected tensor device.")
    if row_offsets.dtype != torch.long or source_indices.dtype != torch.long:
        raise TypeError("CSR row_offsets and source_indices must be torch.long.")
    return _CSRQEFunction.apply(
        query,
        key,
        value,
        bias,
        prior,
        score_multiplier,
        row_offsets,
        source_indices,
    )


def qe_reader_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    prior: torch.Tensor,
    *,
    score_multiplier: torch.Tensor | None = None,
    mode: Literal["complete", "csr"] = "csr",
    row_offsets: torch.Tensor | None = None,
    source_indices: torch.Tensor | None = None,
    source_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Readable Torch oracle for fused-reader parity and CPU tests.

    The oracle intentionally uses a small Python row loop for CSR inputs.  It
    is a test/reference path and must not be used as a performance claim.
    Score and normalization precision follow the fused path: FP64 scalar
    scores, then value-dtype weighted accumulation.
    """

    if mode == "complete":
        batch, heads, queries, sources, _ = _validate_reference_shapes(query, key, value, bias, prior)
        if tuple(bias.shape) != (batch, heads, queries, sources):
            raise ValueError("complete bias must have shape [B,heads,Q,N].")
        if source_mask is None or source_mask.numel() == 0:
            source_mask = torch.ones((batch, sources), device=query.device, dtype=torch.bool)
        elif source_mask.dtype != torch.bool or source_mask.device != query.device or tuple(source_mask.shape) != (batch, sources):
            raise ValueError("complete source_mask must have shape [B,N] and bool dtype.")
        valid = source_mask[:, None, None, :]
        safe_prior = torch.where(valid[:, 0], prior, torch.ones_like(prior))
        scores = torch.einsum("bhqd,bhnd->bhqn", query, key) / (float(query.shape[-1]) ** 0.5)
        has_multiplier = score_multiplier is not None and int(score_multiplier.numel()) > 0
        if has_multiplier:
            if tuple(score_multiplier.shape) != tuple(bias.shape):
                raise ValueError("complete score_multiplier must match bias shape [B,heads,Q,N].")
            if (
                query.dtype == torch.float32
                and prior.dtype == torch.float32
            ):
                scores = scores * score_multiplier + bias
                scores = scores + safe_prior.log()[:, None, :, :]
                scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
                weights = torch.softmax(scores, dim=-1)
                weights = weights * valid.to(weights.dtype)
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(weights.dtype).tiny
                )
                return torch.einsum(
                    "bhqn,bhnd->bhqd", weights.to(value.dtype), value
                )
            scores = scores * score_multiplier
        # Historical QE forms the neural score in its original dtype, then
        # promotes the combined score before adding the FP64 live-prior log.
        # Keeping this order is observable in the last FP32 bits and makes
        # the oracle an exact reference for both the Torch and Triton paths.
        scores = (scores + bias).double() + safe_prior.double().log()[:, None, :, :]
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        weights = weights.to(value.dtype)
        return torch.einsum("bhqn,bhnd->bhqd", weights, value)
    if row_offsets is None or source_indices is None:
        raise ValueError("CSR mode requires row_offsets and source_indices.")
    batch, heads, queries, sources, _ = _validate_reference_shapes(query, key, value, bias, prior)
    if bias.ndim != 2 or tuple(bias.shape) != (int(prior.numel()), heads):
        raise ValueError("CSR bias must have shape [I,heads].")
    if row_offsets.ndim != 1 or int(row_offsets.numel()) != batch * queries + 1:
        raise ValueError("CSR row_offsets must have shape [B*Q+1].")
    if int(row_offsets[-1]) != int(prior.numel()) or int(source_indices.numel()) != int(prior.numel()):
        raise ValueError("CSR row offsets, source indices, and priors must agree on I.")
    if bool(torch.any(row_offsets[1:] < row_offsets[:-1])):
        raise ValueError("CSR row_offsets must be nondecreasing.")
    if int(prior.numel()) and (
        bool(torch.any(prior <= 0.0))
        or bool(torch.any(source_indices < 0))
        or bool(torch.any(source_indices >= sources))
    ):
        raise ValueError("CSR sources must be in range and priors must be strictly positive.")
    has_multiplier = score_multiplier is not None and int(score_multiplier.numel()) > 0
    if has_multiplier and tuple(score_multiplier.shape) != tuple(bias.shape):
        raise ValueError("CSR score_multiplier must match bias shape [I,heads].")
    use_fp32_scalar = bool(
        has_multiplier and query.dtype == torch.float32 and prior.dtype == torch.float32
    )
    output = value.new_zeros((batch, heads, queries, int(value.shape[-1])))
    for row in range(batch * queries):
        start = int(row_offsets[row])
        end = int(row_offsets[row + 1])
        if start == end:
            continue
        batch_index, query_index = divmod(row, queries)
        source = source_indices[start:end]
        scores = torch.einsum("hd,hnd->hn", query[batch_index, :, query_index], key[batch_index, :, source])
        scores = scores / (float(query.shape[-1]) ** 0.5)
        if has_multiplier:
            scores = scores * score_multiplier[start:end].transpose(0, 1)
        scores = scores + bias[start:end].transpose(0, 1)
        if use_fp32_scalar:
            scores = scores + prior[start:end].log()[None, :]
            weights = torch.softmax(scores, dim=-1).to(value.dtype)
        else:
            scores = scores.double() + prior[start:end].double().log()[None, :]
            weights = torch.softmax(scores, dim=-1).to(value.dtype)
        output[batch_index, :, query_index] = torch.einsum(
            "hn,hnd->hd", weights, value[batch_index, :, source]
        )
    return output


def _validate_reference_shapes(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    prior: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must have shape [B,heads,count,head_dim].")
    batch, heads, queries, head_dim = (int(v) for v in query.shape)
    if tuple(key.shape[:2]) != (batch, heads) or tuple(value.shape[:2]) != (batch, heads):
        raise ValueError("query/key/value batch and head dimensions must match.")
    sources = int(key.shape[2])
    if tuple(key.shape[3:]) != (head_dim,) or tuple(value.shape[2:]) != (sources, head_dim):
        raise ValueError("query/key/value dimensions must match.")
    del bias, prior
    return batch, heads, queries, sources, head_dim


def _reference_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    bias: torch.Tensor,
    prior: torch.Tensor,
    score_multiplier: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    mode: Literal["complete", "csr"],
    row_offsets: torch.Tensor | None = None,
    source_indices: torch.Tensor | None = None,
    source_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, ...]:
    """Differentiate the readable oracle for explicit higher-order requests."""

    original_inputs = (query, key, value, bias, prior, score_multiplier)
    tracked_indices = tuple(
        index for index, tensor in enumerate(original_inputs) if tensor.requires_grad
    )
    with torch.enable_grad():
        reference = qe_reader_reference(
            query,
            key,
            value,
            bias,
            prior,
            score_multiplier=score_multiplier,
            mode=mode,
            row_offsets=row_offsets,
            source_indices=source_indices,
            source_mask=source_mask,
        )
        if not reference.requires_grad:
            return tuple(torch.zeros_like(tensor) if tensor.requires_grad else None for tensor in original_inputs)
        tracked = tuple(original_inputs[index] for index in tracked_indices)
        tracked_gradients = torch.autograd.grad(
            reference,
            tracked,
            grad_outputs=grad_output,
            allow_unused=True,
            create_graph=True,
        )
    gradients: list[torch.Tensor | None] = [None] * len(original_inputs)
    for index, gradient in zip(tracked_indices, tracked_gradients, strict=True):
        gradients[index] = gradient
    return tuple(gradients)


__all__ = [
    "TRITON_AVAILABLE",
    "fused_qe_reader",
    "is_triton_qe_available",
    "qe_reader_reference",
]
