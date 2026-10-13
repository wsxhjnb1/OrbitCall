"""Explicit experimental FP32 Tensor-Core linear inference, reference by default.

TF32x3 is Triton's established three-product FP32 algorithm, not IEEE/bitwise
FP32 matrix multiplication. TF32x6 is a separately declared six-product control:
split each operand into three RNA-rounded TF32 residuals and retain terms with
part indices summing to at most two. Neither backend certifies model scores.
The caller must run the frozen finite/score/winner gates before accepting it.

Primary implementation reference (TF32x3/RNA conversion and analogous BF16x6):
https://github.com/triton-lang/triton/blob/v3.7.1/lib/Dialect/TritonGPU/Transforms/F32DotTC.cpp
Fixed launch: M32/N64/K32, four warps, three stages; no shape/result tuning.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


Precision = Literal["tf32x3", "tf32x6"]
Backend = Literal["torch", "triton"]
BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
NUM_WARPS, NUM_STAGES = 4, 3


if triton is not None:

    @triton.jit
    def _round_tf32(value):
        # Same RNA conversion/register constraints as the pinned compiler pass.
        return tl.inline_asm_elementwise(
            "cvt.rna.tf32.f32 $0, $1;", constraints="=r,r", args=[value],
            dtype=tl.float32, is_pure=True, pack=1,
        )

    @triton.jit
    def _linear_kernel(
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        HAS_BIAS: tl.constexpr, SIX_PRODUCTS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        columns = tl.program_id(1) * BN + tl.arange(0, BN)
        reductions = tl.arange(0, BK)
        accumulator = tl.full((BM, BN), 0.0, tl.float32)
        for block in range(tl.cdiv(K, BK)):
            k = block * BK + reductions
            left = tl.load(input_ptr + rows[:, None] * K + k[None, :],
                           mask=(rows[:, None] < M) & (k[None, :] < K), other=0.0)
            # nn.Linear stores contiguous weights as [out_features,in_features].
            right = tl.load(weight_ptr + columns[None, :] * K + k[:, None],
                            mask=(columns[None, :] < N) & (k[:, None] < K), other=0.0)
            if SIX_PRODUCTS:
                left_hi = _round_tf32(left)
                left_residual = left - left_hi
                left_mid = _round_tf32(left_residual)
                left_lo = _round_tf32(left_residual - left_mid)
                right_hi = _round_tf32(right)
                right_residual = right - right_hi
                right_mid = _round_tf32(right_residual)
                right_lo = _round_tf32(right_residual - right_mid)
                partial = tl.full((BM, BN), 0.0, tl.float32)
                partial = tl.dot(left_mid, right_mid, partial, input_precision="tf32", out_dtype=tl.float32)
                partial = tl.dot(left_lo, right_hi, partial, input_precision="tf32", out_dtype=tl.float32)
                partial = tl.dot(left_hi, right_lo, partial, input_precision="tf32", out_dtype=tl.float32)
                partial = tl.dot(left_mid, right_hi, partial, input_precision="tf32", out_dtype=tl.float32)
                partial = tl.dot(left_hi, right_mid, partial, input_precision="tf32", out_dtype=tl.float32)
                partial = tl.dot(left_hi, right_hi, partial, input_precision="tf32", out_dtype=tl.float32)
                accumulator += partial
            else:
                accumulator = tl.dot(left, right, accumulator,
                                     input_precision="tf32x3", out_dtype=tl.float32)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + columns, mask=columns < N, other=0.0)
            accumulator += bias[None, :]
        tl.store(output_ptr + rows[:, None] * N + columns[None, :], accumulator,
                 mask=(rows[:, None] < M) & (columns[None, :] < N))


def _validate(input, weight, bias, backend, precision):
    if backend not in ("torch", "triton") or precision not in ("tf32x3", "tf32x6"):
        raise ValueError("Unknown linear backend or precision")
    if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[1]:
        raise ValueError("Expected input [...,K] and weight [N,K]")
    if input.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError("Experimental linear requires FP32 inputs and weights")
    if input.device != weight.device:
        raise ValueError("Input and weight devices differ")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]
                             or bias.dtype != torch.float32 or bias.device != input.device):
        raise ValueError("Bias must be FP32 [N] on the same device")
    if backend == "triton":
        if triton is None or input.device.type != "cuda":
            raise RuntimeError("Explicit Triton linear requires CUDA and Triton")
        if not input.is_contiguous() or not weight.is_contiguous() or (bias is not None and not bias.is_contiguous()):
            raise ValueError("Triton linear requires contiguous input, weight and bias; no hidden copies")
        if torch.is_grad_enabled() and any(value.requires_grad for value in (input, weight, bias) if value is not None):
            raise ValueError("Triton linear is inference-only; use inference_mode")


def fp32_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
    *, backend: Backend = "torch", precision: Precision = "tf32x3",
) -> torch.Tensor:
    """Return [...,N] FP32 output; default is unchanged native F.linear.

    The explicit Triton path flattens contiguous leading row dimensions without
    copying. M/N/K tails are masked, including empty rows and zero K. Invalid
    metadata raises before launch. Compilation/runtime failures never silently
    fall back. No device-to-host finite checks occur inside this timed API.
    """
    _validate(input, weight, bias, backend, precision)
    if backend == "torch":
        return F.linear(input, weight, bias)
    k, n = weight.shape[1], weight.shape[0]
    m = input.numel() // k if k else 1
    # For zero K, reshape(-1,0) is ambiguous; leading dimensions define rows.
    if k == 0:
        m = 1
        for size in input.shape[:-1]:
            m *= size
    output = torch.empty((*input.shape[:-1], n), device=input.device, dtype=torch.float32)
    if m == 0 or n == 0:
        return output
    with torch.cuda.device(input.device):
        _linear_kernel[(triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))](
            input, weight, bias if bias is not None else input, output,
            m, n, k, bias is not None, precision == "tf32x6",
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps=NUM_WARPS, num_stages=NUM_STAGES,
        )
    return output


class ExperimentalLinear(nn.Module):
    """Inference wrapper retaining the original Parameter objects and ties."""

    def __init__(self, original: nn.Linear, *, backend: Backend, precision: Precision):
        super().__init__()
        if type(original) is not nn.Linear:
            raise TypeError("Only ordinary nn.Linear modules can be wrapped")
        self.in_features, self.out_features = original.in_features, original.out_features
        self.weight, self.bias = original.weight, original.bias
        self.backend, self.precision = backend, precision
        self.training = original.training

    def forward(self, input):
        return fp32_linear(input, self.weight, self.bias, backend=self.backend, precision=self.precision)


@contextmanager
def accelerated_linears(
    model: nn.Module, *, scope: Literal["head", "backbone", "all"] = "backbone",
    precision: Precision = "tf32x3", backend: Backend = "triton",
):
    """Temporarily opt in selected ordinary Qwen2 linear modules; always restore.

    Only model.* backbone and lm_head qualify. Unrelated modules and embedding
    modules are untouched. A tied head remains tied to its original embedding
    Parameter; the retrofit does not alter/decompose/transplant weight storage.
    Return the selected names for a receipt. Native reference wrappers can be
    requested with backend='torch'. No architecture/precision auto-detection or
    fallback silently enables this experimental kernel.
    """
    if scope not in ("head", "backbone", "all") or precision not in ("tf32x3", "tf32x6") or backend not in ("torch", "triton"):
        raise ValueError("Unknown retrofit scope, precision or backend")
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "qwen2" or getattr(config, "use_sliding_window", False):
        raise ValueError("Retrofit is scoped to the declared non-sliding Qwen2 models")
    if model.training:
        raise ValueError("Experimental retrofit requires an eval-mode model")
    plan, wrappers = [], {}
    for name, module in model.named_modules(remove_duplicate=False):
        selected = ((name == "lm_head" and scope in ("head", "all"))
                    or (name.startswith("model.") and scope in ("backbone", "all")))
        if not selected or not isinstance(module, nn.Linear):
            continue
        if type(module) is not nn.Linear:
            raise TypeError("Custom Linear subclasses are outside the retrofit contract")
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
            raise ValueError("Hooked Linear modules must be explicitly audited before retrofit")
        if module.weight.dtype != torch.float32 or (module.bias is not None and module.bias.dtype != torch.float32):
            raise TypeError("Retrofit requires FP32 weights and biases")
        if backend == "triton" and (module.weight.device.type != "cuda" or triton is None):
            raise RuntimeError("Explicit retrofit requires CUDA/Triton linear weights")
        if backend == "triton" and (not module.weight.is_contiguous() or (module.bias is not None and not module.bias.is_contiguous())):
            raise ValueError("Retrofit requires contiguous weights and biases")
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if id(module) not in wrappers:
            wrappers[id(module)] = ExperimentalLinear(module, backend=backend, precision=precision)
        wrapper = wrappers[id(module)]
        plan.append((name, parent, attribute, module, wrapper))
    if not plan:
        raise ValueError("No eligible linears in the requested Qwen2 scope")
    try:
        for _, parent, attribute, _, wrapper in plan:
            setattr(parent, attribute, wrapper)
        yield tuple(name for name, *_ in plan)
    finally:
        for _, parent, attribute, original, _ in reversed(plan):
            setattr(parent, attribute, original)
