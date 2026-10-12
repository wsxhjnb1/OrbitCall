"""Requested-token logprobs with full-vocabulary FP32 normalization.

The Triton path changes storage and reduction layout, not vocabulary support.
It does not promise bitwise identity to PyTorch's different reduction tree.
This inference API deliberately performs no device-to-host validity checks.
Invalid target IDs, NaN logits, positive infinity, or an all-minus-infinity row
produce NaN. The calling scorer must reject NaN before using or saving scores.
Minus infinity for a requested, masked-out token is a legitimate logprob.
"""

from __future__ import annotations

from typing import Literal

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


Backend = Literal["auto", "torch", "torch_lse", "torch_ce", "triton"]
_TILE_SIZE = 4096
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


if triton is not None:

    @triton.jit
    def _tile_lse_stats(
        logits,
        tile_maxima,
        tile_sums,
        vocab_size: tl.constexpr,
        row_stride: tl.constexpr,
        vocab_stride: tl.constexpr,
        num_tiles: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1)
        columns = tile * BLOCK + tl.arange(0, BLOCK)
        in_vocab = columns < vocab_size
        values = tl.load(
            logits + row * row_stride + columns * vocab_stride,
            mask=in_vocab,
            other=-float("inf"),
        ).to(tl.float32)
        maximum = tl.max(values, axis=0)
        # A tile containing only -inf contributes zero to a valid row's sum.
        weights = tl.where(
            maximum == -float("inf"), 0.0, tl.exp(values - maximum)
        )
        total = tl.sum(weights, axis=0)
        # Explicitly propagate NaN even if the backend's max ignores NaN.
        invalid = tl.sum(
            (in_vocab & ((values != values) | (values == float("inf")))).to(
                tl.int32
            ),
            axis=0,
        )
        total = tl.where(invalid > 0, float("nan"), total)
        index = row * num_tiles + tile
        tl.store(tile_maxima + index, maximum)
        tl.store(tile_sums + index, total)

    @triton.jit
    def _merge_lse_stats(
        tile_maxima,
        tile_sums,
        row_maxima,
        row_log_sums,
        num_tiles: tl.constexpr,
        MERGE_BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        tiles = tl.arange(0, MERGE_BLOCK)
        valid_tiles = tiles < num_tiles
        maxima = tl.load(
            tile_maxima + row * num_tiles + tiles,
            mask=valid_tiles,
            other=-float("inf"),
        )
        sums = tl.load(
            tile_sums + row * num_tiles + tiles, mask=valid_tiles, other=0.0
        )
        maximum = tl.max(maxima, axis=0)
        rescaled = tl.where(
            valid_tiles & (maxima != -float("inf")),
            tl.exp(maxima - maximum) * sums,
            0.0,
        )
        # A NaN tile may also have max=-inf; never discard its invalid marker.
        bad_tiles = tl.sum((valid_tiles & (sums != sums)).to(tl.int32), axis=0)
        total = tl.sum(rescaled, axis=0)
        log_sum = tl.where(bad_tiles > 0, float("nan"), tl.log(total))
        tl.store(row_maxima + row, maximum)
        tl.store(row_log_sums + row, log_sum)

    @triton.jit
    def _gather_selected(
        logits,
        target_ids,
        row_ids,
        row_maxima,
        row_log_sums,
        output,
        num_rows: tl.constexpr,
        num_edges: tl.constexpr,
        vocab_size: tl.constexpr,
        row_stride: tl.constexpr,
        vocab_stride: tl.constexpr,
        target_stride: tl.constexpr,
        ids_stride: tl.constexpr,
        HAS_ROW_IDS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        edge = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        in_edges = edge < num_edges
        if HAS_ROW_IDS:
            row = tl.load(row_ids + edge * ids_stride, mask=in_edges, other=-1)
        else:
            row = edge.to(tl.int64)
        target = tl.load(target_ids + edge * target_stride, mask=in_edges, other=-1)
        valid_row = (row >= 0) & (row < num_rows)
        valid_target = (target >= 0) & (target < vocab_size)
        valid = in_edges & valid_row & valid_target
        selected = tl.load(
            logits + row * row_stride + target * vocab_stride,
            mask=valid,
            other=float("nan"),
        ).to(tl.float32)
        maximum = tl.load(row_maxima + row, mask=valid, other=float("nan"))
        log_sum = tl.load(row_log_sums + row, mask=valid, other=float("nan"))
        # Subtract max before log(sum); x - (max + log(sum)) loses precision
        # for very large shared offsets, even when all logits are finite.
        result = (selected - maximum) - log_sum
        tl.store(output + edge, result, mask=in_edges)


def _validate_shapes(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    row_ids: torch.Tensor | None = None,
) -> None:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, V]")
    if logits.shape[1] == 0:
        raise ValueError("the vocabulary dimension must be nonempty")
    if logits.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("logits must be float16, bfloat16, or float32")
    if target_ids.ndim != 1:
        raise ValueError("target_ids must be one-dimensional")
    if row_ids is None and target_ids.shape[0] != logits.shape[0]:
        raise ValueError("target_ids must have shape [N] when row_ids is omitted")
    if target_ids.dtype != torch.int64:
        raise TypeError("target_ids must have dtype int64")
    if target_ids.device != logits.device:
        raise ValueError("logits and target_ids must be on the same device")
    if row_ids is not None:
        if row_ids.shape != target_ids.shape:
            raise ValueError("row_ids must have the same shape as target_ids")
        if row_ids.dtype != torch.int64:
            raise TypeError("row_ids must have dtype int64")
        if row_ids.device != logits.device:
            raise ValueError("logits and row_ids must be on the same device")
    if logits.requires_grad:
        raise ValueError("selected_logprobs is an inference-only API")


def _torch_indices(logits, target_ids, row_ids):
    if row_ids is None:
        row_ids = torch.arange(logits.shape[0], device=logits.device)
    valid = (
        (row_ids >= 0)
        & (row_ids < logits.shape[0])
        & (target_ids >= 0)
        & (target_ids < logits.shape[1])
    )
    return (
        row_ids.clamp(0, max(logits.shape[0] - 1, 0)),
        target_ids.clamp(0, logits.shape[1] - 1),
        valid,
    )


def selected_logprobs_torch(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    row_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Materializing reference, using stable FP32 torch.log_softmax.

    Out-of-range targets return NaN rather than performing an invalid gather.
    Shape/type checks are metadata-only; no host synchronization is introduced.
    """
    _validate_shapes(logits, target_ids, row_ids)
    if logits.shape[0] == 0:
        return torch.full_like(target_ids, float("nan"), dtype=torch.float32)
    safe_rows, safe_ids, valid = _torch_indices(logits, target_ids, row_ids)
    full_logprobs = torch.log_softmax(logits.float(), dim=-1)
    selected = full_logprobs[safe_rows, safe_ids]
    return torch.where(valid, selected, float("nan"))


def selected_logprobs_torch_lse(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    row_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full-V logsumexp control without a full log-softmax output.

    Explicit max-centering avoids cancellation under large common offsets.
    Eager Torch still materializes a centered FP32 [N, V] temporary; this is
    a storage/control implementation, not a new logsumexp algorithm.
    """
    _validate_shapes(logits, target_ids, row_ids)
    if logits.shape[0] == 0:
        return torch.full_like(target_ids, float("nan"), dtype=torch.float32)
    safe_rows, safe_ids, valid = _torch_indices(logits, target_ids, row_ids)
    fp32 = logits.float()
    maxima = fp32.amax(dim=-1)
    shifted_lse = torch.logsumexp(fp32 - maxima[:, None], dim=-1)
    selected = (fp32[safe_rows, safe_ids] - maxima[safe_rows]) - shifted_lse[safe_rows]
    return torch.where(valid, selected, float("nan"))


def selected_logprobs_cross_entropy(
    logits: torch.Tensor, target_ids: torch.Tensor
) -> torch.Tensor:
    """Negative torch cross-entropy control, one distinct target per row.

    This native reduction is a required competing implementation. It is kept
    distinct from the repeated/scrambled parent-row API so that an apparent
    speedup cannot come from materializing repeated [E, V] input logits.
    """
    _validate_shapes(logits, target_ids)
    if logits.shape[0] == 0:
        return torch.empty(0, device=logits.device, dtype=torch.float32)
    valid = (target_ids >= 0) & (target_ids < logits.shape[1])
    safe_ids = target_ids.clamp(0, logits.shape[1] - 1)
    values = -torch.nn.functional.cross_entropy(
        logits.float(), safe_ids, reduction="none"
    )
    return torch.where(valid, values, float("nan"))


def selected_logprobs(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    row_ids: torch.Tensor | None = None,
    *,
    backend: Backend = "auto",
) -> torch.Tensor:
    """Return [E] float32 logprobs normalized over all V logits.

    With no ``row_ids``, E=N and edge i selects from logits row i. Otherwise
    each edge selects ``logits[row_ids[i], target_ids[i]]``; repeated parents
    are normalized once without constructing a repeated [E, V] logits tensor.

    ``auto`` uses Triton on CUDA when available, otherwise the Torch reference.
    Explicit ``triton`` fails if it is unavailable; it never silently changes
    the requested backend after a compilation/runtime error. Strided inputs
    are supported. The three-stage Triton scratch space contains two FP32 values
    per row and vocabulary tile, instead of a full [N, V] log-softmax buffer.

    The caller owns NaN rejection and any synchronization for wall timing.
    Import and use this function under ``torch.inference_mode()`` for the study.
    """
    _validate_shapes(logits, target_ids, row_ids)
    if backend not in ("auto", "torch", "torch_lse", "torch_ce", "triton"):
        raise ValueError("unrecognized selected-logprob backend")
    if backend == "torch_lse":
        return selected_logprobs_torch_lse(logits, target_ids, row_ids)
    if backend == "torch_ce":
        if row_ids is not None:
            raise ValueError("torch_ce requires one distinct target per logits row")
        return selected_logprobs_cross_entropy(logits, target_ids)
    available = triton is not None and logits.device.type == "cuda"
    if backend == "triton" and not available:
        raise RuntimeError("the Triton backend requires Triton and CUDA tensors")
    if backend == "torch" or (backend == "auto" and not available):
        return selected_logprobs_torch(logits, target_ids, row_ids)
    rows, vocab_size = logits.shape
    edges = target_ids.numel()
    output = torch.empty(edges, device=logits.device, dtype=torch.float32)
    if edges == 0:
        return output
    if rows == 0:
        return output.fill_(float("nan"))
    num_tiles = triton.cdiv(vocab_size, _TILE_SIZE)
    maxima = torch.empty((rows, num_tiles), device=logits.device, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    row_maxima = torch.empty(rows, device=logits.device, dtype=torch.float32)
    row_log_sums = torch.empty_like(row_maxima)
    # CUDA/Triton launch context must match the input tensors, including when
    # the caller's current default device differs from logits.device.
    with torch.cuda.device(logits.device):
        _tile_lse_stats[(rows, num_tiles)](
            logits,
            maxima,
            sums,
            vocab_size,
            logits.stride(0),
            logits.stride(1),
            num_tiles,
            _TILE_SIZE,
            num_warps=8,
        )
        _merge_lse_stats[(rows,)](
            maxima,
            sums,
            row_maxima,
            row_log_sums,
            num_tiles,
            triton.next_power_of_2(num_tiles),
            num_warps=4,
        )
        _gather_selected[(triton.cdiv(edges, 256),)](
            logits,
            target_ids,
            row_ids,
            row_maxima,
            row_log_sums,
            output,
            rows,
            edges,
            vocab_size,
            logits.stride(0),
            logits.stride(1),
            target_ids.stride(0),
            0 if row_ids is None else row_ids.stride(0),
            row_ids is not None,
            256,
            num_warps=4,
        )
    return output
