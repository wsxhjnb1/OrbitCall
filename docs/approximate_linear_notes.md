# Approximate FP32-storage linear acceleration

This note describes the new approximate complete-scoring study in
[APPROXIMATE_SCORING_PROTOCOL.md](APPROXIMATE_SCORING_PROTOCOL.md). The user now
permits small numerical differences. The earlier `3e-5`-nat Tensor Core gates
remain failed under their original contract; their source, outputs and costs
are preserved. A newly declared error budget is a separate experiment, not a
retroactive passing result. This note asserts no new speed or accuracy result.

## What changes

[linear_acceleration.py](../src/orbitcall/linear_acceleration.py) stores weights,
inputs and outputs as `torch.float32`. The original Parameter objects are
retained, including tied embedding/head weights. Embedding lookup modules are
not replaced, weights are not trained or compressed, and no vocabulary logits
are discarded. The complete EOS-terminated serialization family and the
full-vocabulary native cross-entropy normalizer remain the same.

**FP32 storage does not imply IEEE FP32 multiplication.** The optional kernel
uses Tensor Core products with a different operand decomposition and reduction
order. Triton's established `tf32x3` mode approximates a float32 product with
high/residual cross-products; it is not a promise of bitwise agreement with
native IEEE GEMM. The local `tf32x6` control splits each operand into three
RNA-rounded TF32 residual components and retains six products:

```text
A ~= A0 + A1 + A2;  B ~= B0 + B1 + B2
AB ~= A0B0 + A0B1 + A1B0 + A0B2 + A2B0 + A1B1
```

The omitted residual products and FP32 accumulation still introduce error.
More retained terms do not guarantee that every complete-model score is
closer to the IEEE reference, because GEMM reductions and later model layers
also matter. The decomposition is credited to established Tensor Core
multi-product techniques, not claimed as a new numerical identity.
The pinned [Triton compiler implementation](https://github.com/triton-lang/triton/blob/v3.7.1/lib/Dialect/TritonGPU/Transforms/F32DotTC.cpp)
documents TF32x3 and the analogous multi-part BF16 algorithms.

Six configurations are frozen: `tf32x3`/`tf32x6` crossed with
`head`/`backbone`/`all`. Every configuration uses M32/N64/K32 tiles, four warps
and three stages, without autotuning from observed results. Head-only changes
the LM-head linears; backbone-only changes the ordinary Qwen2 linears under
`model.*`; all changes both. Attention score/value matmuls remain native IEEE
FP32 in these custom-linear configurations. Explicit opt-in, unsupported-input
rejection and exception-safe module restoration remain part of the API.

## Accuracy contract and retained evidence

The new protocol compares against the frozen full-prompt IEEE-FP32 reference
on the original 119 inputs for each pinned Qwen2.5 checkpoint. It declares:

| Quantity | Prospective acceptance limit |
| --- | ---: |
| Maximum absolute token logprob error | 0.01 nats |
| Pooled, token-weighted mean absolute error | 0.0005 nats |
| Maximum absolute complete-sequence error | 0.1 nats |
| Winner agreement for each of the eleven aggregators | At least 99% of scored records |
| Joint agreement across all eleven aggregators | At least 99% of scored records |

The maximum errors cover every requested token and every complete sequence,
including EOS, not only the eventual winner. Finite values, exact path lengths,
token-sum consistency, unchanged prompts/candidate families and source/model
receipts are required. Preserve both winners and margins, all changed IDs,
per-item errors and development/test subgroup results. Agreement is evaluated
with integer counts rather than rounded percentages. A small timing pool can
therefore require all its winners to agree even though the complete population
contract permits a small disagreement rate.

Zero-candidate records stay in the original population and are reported
separately. In the retained 1.5B pool, the sixteen zero-family inputs must not
inflate the numerical/winner denominator or disappear. Agreement with the
reference is not agreement with gold, and these limits do not certify
downstream task accuracy or the ideal real-valued model's probabilities.
No new candidates, labels, sampling or failure-based input exclusions are
introduced. The development choice is frozen before held-out observations
and is applied unchanged to both checkpoints.

## A strong native approximate control

The seventh configuration is native Torch's global `high` float32 matmul
policy, with CUDA TF32 allowed and cuDNN TF32 disabled. It uses the same model,
FP32 storage, full input computation and score-stage workload. This ordinary
control is eligible for selection: if it wins, the result supports a native
setting rather than a custom-kernel advantage. A gain over TF32-disabled native
GEMM alone would not establish that the custom decomposition is necessary.

The global native policy has a **broader arithmetic scope** than a head-only
or backbone-only retrofit: it can affect attention's `torch.matmul` calls as
well as `nn.Linear`. It must be labeled as global native approximate matmul,
with its own measured error and winner agreement. `high` is an implementation
policy, not proof that a particular Tensor Core kernel was dispatched; actual
kernel identity requires a retained runtime trace. Its internal algorithm can
depend on device, shape and library version.

### Precision switching on the pinned runtime

A flag-only check in an isolated Spark process confirmed the following on
Torch **2.13.0+cu130**, without creating tensors, initializing CUDA or running
any GPU operation:

The same flag-only check was repeated on 2026-10-13 at 04:17:58 UTC solely to
retain an auditable receipt. Its [source](../metadata/torch_precision_flagcheck_source.py),
[complete JSON observations](../metadata/torch_precision_flagcheck_20261013_v1.json)
and [log](../logs/torch_precision_flagcheck_20261013_v1.log) are preserved. The
receipt contains the source SHA256 and runtime identity; transferred copies
matched the Spark SHA256 values. All six snapshots retained
`cuda_initialized=False`, and both complete flag snapshots restored exactly.
These are configuration checks, not an additional numerical or timing study.

```sh
.venv/bin/python -u metadata/torch_precision_flagcheck_source.py \
  --receipt metadata/torch_precision_flagcheck_fresh.json \
  --log logs/torch_precision_flagcheck_fresh.log
```

The script refuses to overwrite existing outputs. Use a fresh isolated process
and new output names when verifying a different runtime.

| Policy | `get_float32_matmul_precision()` | CUDA `allow_tf32` | CUDA/MKLDNN raw `fp32_precision` |
| --- | --- | --- | --- |
| Fresh process | `highest` | `False` | `none` / `none` |
| Set `highest` | `highest` | `False` | `ieee` / `ieee` |
| Set `high` | `high` | `True` | `tf32` / `tf32` |

`set_float32_matmul_precision` also changes the MKLDNN matmul policy. Restoring
only its legacy getter value, or only `allow_tf32`, does not necessarily restore
the original raw `none` state. The tested round-trip saves these expressions:

```python
saved = (
    torch.get_float32_matmul_precision(),
    torch.backends.cuda.matmul.fp32_precision,
    torch.backends.mkldnn.matmul.fp32_precision,
)
try:
    torch.set_float32_matmul_precision("high")
    # Execute the explicitly declared native approximate control here.
finally:
    torch.set_float32_matmul_precision(saved[0])  # Restore legacy policy first.
    torch.backends.cuda.matmul.fp32_precision = saved[1]
    torch.backends.mkldnn.matmul.fp32_precision = saved[2]
```

If the experiment additionally changes cuDNN `allow_tf32`, save and restore
that setting as well; changing the matmul policy alone does not switch cuDNN.
Do not set `high` and then disable CUDA `allow_tf32`, which cancels the intended
fast matmul policy. Read and record settings inside each arm and after restoration.
Serialize these process-wide policy changes; they are not thread-local controls.

This is a compatibility recipe verified for this pinned environment, not a
portable endorsement of arbitrary mixed legacy/new settings. PyTorch warns
against mixing its precision APIs. Code using the newer precision API exclusively
should stay within that API, and a dependency upgrade requires a fresh flag
round-trip check. See the [pinned setter documentation/source](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/__init__.py)
and [official precision guidance](https://docs.pytorch.org/docs/main/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).
The CPU-only check confirms configuration/restoration, not numerical accuracy,
CUDA dispatch, throughput or compatibility with captured strict-FP32 graphs.

## What a runtime result would mean

The primary measurement is **complete scoring wall time**: reparsing generated
calls, orbit construction, tokenization, packing, H2D transfer, full backbone,
LM-head, full-vocabulary normalization, D2H score extraction and aggregation.
Checkpoint loading, candidate generation, correctness checks and artifact I/O
are outside the timed body. First-call/JIT setup and failed attempts are charged
and reported separately from warm repeated scoring.

Use the frozen held-out pool and complete randomized paired blocks, preserve
every raw window before validity evaluation, and report both models and all
eligible comparators. Peak allocated and reserved memory must accompany their
baselines; FP32 weights do not imply model-memory reduction. A failed bound or
incomplete timing plan cannot produce an accepted speed claim.

Any accepted gain is restricted to this score stage, checkpoint/runtime,
candidate pool and declared numerical budget. It is not a measurement of
candidate-generation speed, HTTP serving throughput, overall tool-use latency,
training speed or task-accuracy improvement. Empirical approximate agreement
is separate from the optional exact supplied-table certificate.
