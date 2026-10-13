# Complete-scorer acceleration: chronology and separated studies

This document was saved on 2026-10-13 **after the first development discovery
had started**. It is a faithful record of the already declared source/plan and
its negative disposition, followed by requirements for a separate future
study. Its Markdown hash is not evidence of a prelaunch freeze. The original
`plan.json`, archived source and execution logs are authoritative for what the
completed screen actually ran.

For the executed follow-ups, the specialized `TENSORLINEAR_PROTOCOL.md` and
`GRAPH_SCORING_PROTOCOL.md` are authoritative. In the graph study only the full
backbone is captured, so exact(batch,width) keys suffice: positions always
0..width-1, the actual native4D mask is recopied every batch, and variable
per-sequence hidden gathering/head work remains eager. The generic key advice
below concerned capture of the entire score body. Specialized stages permit
per-model whole119 audit then timing; a combined result requires both models
to pass. Both specialized timing configurations and plans freeze prospectively.

The completed OrbitCall quality/normalizer study, public evidence and all
failures remain unchanged. Existing quality labels are already known. None of
these acceleration stages may read answers or evaluation reports, resample
candidates, add gold calls, change the finite family or produce new quality
claims. A timing holdout is held out from acceleration selection; it is not a
new unseen-label task-quality test.

## Completed initial development screen

Run: `runs/full_scoring_discovery_3b_20261013_v1` (complete, retained).

The initial screen was implemented by `scripts/screen_full_scoring.py` and
`src/orbitcall/full_acceleration.py`, with the actual versions archived in its
run directory before execution. The source creates the plan and references
before the configuration loop. That plan, not an earlier draft of this document,
records its checkpoint receipts, candidate/input hashes and source inventory.

The fixed grid comprised12 configurations in this exact order:

1. Native eager32, eager128, eager256.
2. Native SDPA32, SDPA128, SDPA256.
3. Packed eager32, eager128, eager256.
4. Packed SDPA32, SDPA128, SDPA256.

All arms use full-prompt FP32 backbone computation, deterministic settings and
TF32 disabled. Native arms use direct Torch cross-entropy; packed arms use the
existing Triton full-vocabulary requested-token reducer. Thus the initial
packing comparison also changes the reducer; it is not an isolated estimate
of packing alone. The12-arm grid was coordinated before its outcomes.

The packed mechanism combines bulk row construction/H2D transfer and omission
of a redundant right-padding mask. It retains ordinary causal attention,
absolute positions and every real token/EOS. A scored query precedes padding,
so causal attention cannot see a right-padding key. This rule does not apply
to left/interior padding, cached non-square suffix attention or ancestor-tree
masks. `_tree` and `TokenTrie.ancestor_mask` must not lose their masks.

Fixtures are the source's first up-to-four lexicographic3B development IDs
with at least two distinct valid primary-seed calls. The screen invokes the
entire finite family of each question in one sequence batch. It does not use
batch8 as its timing policy and does not select fixtures by quality, margin,
runtime or numerical error.

For every arm, the actual execution records **one profiler-warmed full-pool
numerical call** and its CPU/CUDA trace before qualification. Qualification
uses maximum per-token error <=3e-5 nats and all eleven semantic winner
agreements against freshly recorded native eager32/CE whole-family references.
Invalid arms are retained and are not timed. Qualifying arms then receive
**two unprofiled, warmed complete-pool measured calls**, in fixed configuration
order. They are not three randomized rounds, six ABBA blocks or confirmatory
held-out measurements. `plan.json` declares1800 seconds for this screen.

Only native eager32 and packed eager32 passed the development gate. Wider
heads and native SDPA failed its unchanged3e-5 numerical criterion. Reported
warmed complete-pool times were approximately6.82s native versus6.88s packed;
raw observations, rather than these rounded coordination values, are the
source for any exact statistic. The screen establishes no packing benefit.

**Disposition:** retire this initial bulk-packing/mask-omission utility attempt
as a negative development screen. Preserve all12 gate files, profiler traces,
source, plan, both qualifying arms' measurements, failures and process cost.
Do not pick a surviving small-head arm and relabel it as a positive result,
loosen the tolerance, drop a failed fixture or replace the original run.
No whole119 audit, selected-config freeze, eight-test-fixture confirmation or
six-block ABBA result is attributed to this completed screen.

The previously drafted9000-second discovery/audit/confirmation schedule was
not an executed allocation or a completed protocol. It is superseded by this
chronology correction; it must not be used to describe the negative run's
sampling, timing, source-freeze time or cost.

## Different prospective mechanisms

TF32x3/TF32x6 matrix multiplication and exact CUDA Graph replay are different
execution mechanisms. They require new source, new run IDs and a separately
saved prospective supplement **before their first observations**. They do not
repair or rename the retired bulk-packing result.

The parent-owned supplement must freeze the exact configuration list, native
controls, numerical reference, selection rule, fixtures, source/checkpoint
hashes, repetitions, random seed and stage ceilings. Proposed stage allowances
are1200s for a new development screen and separately declared3000s allowances
for audits/timing; the supplement must state whether those latter allowances
are per model or shared. Do not infer an approved aggregate budget from this
proposal. Retain setup, cold compilation/capture and failed work separately.

A tensor-matmul supplement may predeclare head-only, backbone-only and combined
TF32x3/x6 configurations, with strict native FP32 controls. It must explicitly
state that its internal matmul arithmetic differs, even if weights and outputs
are FP32. Numerical failure remains failure at3e-5; no result-dependent layer
mixture, precision switch or larger tolerance is allowed. A passed empirical
gate does not prove bitwise or mathematical identity in floating point.

For a CUDA Graph follow-up, preserve native eager32 full-prompt execution and
its actual GEMM/attention shapes. Capture the GPU computation only after fixed
shape-specific warmup. Use identical preallocated input/work/output buffers
and ordinary packing in the eager control as well; otherwise allocation or
input-preparation improvements could masquerade as graph replay benefit.

Tokenization, CPU construction, H2D input copies, result D2H extraction,
shape/finite checks and all-aggregator ranking remain outside capture but inside
complete scorer wall timing. Store cold capture time, warmup, graph workspace
and graph reuse count. Do not pad all records to a global geometry if that
changes the historical arithmetic shapes. A cache key must cover model/dtype,
prompt length, batch geometry, each sequence length and LM-head chunk layout.
No stale masks, target IDs, result pointers or changing Python metadata may be
silently reused.

Graph replay is a standard CUDA mechanism, not claimed as an algorithmic
novelty. It may reduce host launch gaps; it does not remove full backbone FLOPs,
reduce the legal vocabulary or improve task quality. Any actual gain and its
cold-start break-even point remain unmeasured until the new study completes.

## Shared requirements for a fresh accepted study

Use the same two pinned public Qwen checkpoints, candidate receipts and input
SHA256 `1e8edf278caa6066c1b4c1fab07599cc8acb03fdf0037d3447603ff75916060c`.
No new candidates, labels, permutations, filtering or answer-dependent choice.

Development-only discovery is confined to3B. Freeze its selected configuration
and strongest predeclared native comparator before any held-out observation,
then apply those same settings unchanged to1.5B. A source/configuration that
fails on test is retained; do not select a different development arm afterward.

Whole-population numerical validation retains **all119 inputs:20 development
and99 test**, with audit sequence batch8 and original order. Compare against
the stored FULL/eager/FP32 batch8 vectors and compare the selected path with
its native control. Require exact path/EOS shapes, finite vectors, <=3e-5 nats
per token, sequence error <=EOS-inclusive length*3e-5+1e-8, token-sum checks and
all eleven semantic winners. Preserve actual margins/errors. Zero-family
records remain explicit null failures and are excluded only from the nonempty
numerical denominator; null agreement is not a numeric comparison. The16 zero
primary families in1.5B cannot disappear.

Freeze timing fixtures before development outcomes: first up-to-eight
lexicographic test IDs per model with >=2 distinct valid primary calls, or all
qualifying IDs if fewer. No fallback to better-performing IDs, new samples or
another dataset. This choice is unrelated to known task-quality labels.

Discovery/timing uses the **whole finite family in one batch**. Immediately
before confirmatory timing, use fresh native and fresh legacy references at
that same whole-family batch and apply the strict3e-5/EOS/shape/all-winner gates.
A batch8 audit must never be relabeled as whole-family stored-vector equality.

After both full-population audits pass, use six randomized ABBA blocks per
predeclared pair on the entire fixed timing fixture pool, with a frozen seed
and plan. Validate every measured window; report all six raw block ratios,
median/range, memory and block-level uncertainty. No deleted outlier, adaptive
repeat, early success stop or selected best block. Empirical resampling is not
additional independent experimental trials.

Primary utility is complete-scoring wall time against the strongest frozen
native baseline. The timing includes parsing, transfers, full computation,
extraction and aggregation. CUDA brackets can contain host gaps and are not
GPU-busy time. A score-stage gain is not full generation, serving throughput,
overall tool-use latency or broad inference acceleration.

Snapshot whole-host occupancy before stages and around measured windows. Never
stop unrelated work. Preserve interference, incomplete plans, controller logs,
source/raw hashes and all failed/successful costs. Write exclusive new outputs.
A ceiling-only retry is prospective and keeps scientific choices unchanged.

## Primary implementation references

Native SDPA dispatch and mask semantics are documented by
[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
and the [HF attention interface](https://huggingface.co/docs/transformers/main/attention_interface).
Actual kernel names, including a math fallback, must come from retained runtime
traces. Do not assume a faster backend from the name `sdpa`.

[PyTorch numerical-accuracy guidance](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
explains why changing batch/GEMM/reduction geometry can change floating-point
results. That explains a possible failure source but never permits a tolerance
increase. The initial larger-head/SDPA failures remain negative evidence.
