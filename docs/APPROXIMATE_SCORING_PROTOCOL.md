# Approximate complete-scoring study, 2026-10-13

The user explicitly permits small numerical differences and does not require
100% equality. This is a new study, frozen before its development measurements.
Earlier 3e-5 Tensor Core failures remain failures under their original protocol.
The original public results and model/input freezes are immutable.

## Contract

All numerical work runs on the same authorized DGX Spark GB10, pinned public
Qwen2.5-3B/1.5B-Instruct assets, original 119 input-only BFCL records and fixed
generated candidates. No labels are read and no generation is repeated.
Scores include every serialization and explicit EOS. FP32 weights, activation
storage, eager attention, full prompt computation, head chunk32 and native CE
are shared. The optimization changes matrix arithmetic, not scoring semantics.

Prospective bounds relative to the frozen IEEE-FP32 reference: maximum absolute
token logprob error0.01 nats, pooled mean absolute token error0.0005 nats,
maximum absolute complete sequence error0.1 nats. Every one of11 aggregators
must retain at least99% winner agreement per model among scored records;
zero-family records are reported separately. Joint all11 agreement must also
be at least99%, evaluated by integer counts without percentage rounding.
Development/test subgroup rates, union-changed IDs and per-item error means
are reported separately; the acceptance denominator remains the scored119
population. Numerical bounds are applied to
all token paths, not only the winner. Full errors, changed IDs and both margins
are retained. These are empirical acceptance limits, not a formal certificate
of downstream accuracy. A0.01-nat token error corresponds to a probability
ratio bounded by exp(0.01), about1.01; sequence0.1 is about1.105.

## Fixed development selection

Six unchanged Triton configs: TF32x3/x6 times head/backbone/all. The seventh arm
is native PyTorch matmul precision high with CUDA TF32 enabled and cuDNN TF32
disabled. Native-high is a global matmul policy, affecting attention matmuls
as well as linear modules; PyTorch may choose TF32 or another allowed internal
implementation. Record actual effective flags, without asserting a particular
decomposition. This approximate control is eligible for selection as well: a simple
native option winning means no custom-kernel advantage. Native IEEE FP32 with
TF32 off is the exact baseline. Each temporary policy restores its settings.

Use the original first four eligible3B development fixtures, whole-family
batches. Freeze source hashes, fixtures, configs and limits before inference.
Each candidate receives an untimed accuracy check, then three timed full-score
calls if valid. Development requires all winners unchanged and the same
numerical bounds. Select the eligible config with minimum median wall seconds,
lexical JSON tie break. Fixed order is retained; no held-out speed inference
comes from discovery. Native timing is retained for context. Budget1800s.
Do not alter tile sizes, scopes, limits or inputs from these observations.

## Confirmation

Freeze selection before all119 audits and the pre-frozen first8 eligible test
timing fixtures. Use exactly that config on both models without reselection.
Per model, audit all119 against original stored FP32 scores at batch8 and also
audit the PyTorch-high control under identical limits. Failed audits remain
retained and are ineligible for accepted timing claims. Selected failure ends
that model confirmation. Native-control failure excludes only that comparator.

Then freshly check each8-fixture whole-family shape against current IEEE native
CE results. Whole-family and every subsequent measured window must satisfy the
same bounds and per-aggregator99% threshold; because the pool has8 records this
implies all8 winners agree. This is a gate, not an assertion that100% agreement
is always required in the119 population. No changed family/prompt/token path is
allowed. If selection equals the native approximate control, do not duplicate
the same comparator or claim a custom implementation gain.

Each eligible comparison has6 paired blocks, randomized ABBA or BAAB with
seed271828. Each window scores the whole8-fixture pool once. Compare original
IEEE native versus selected, and native-high versus selected when distinct and
eligible. Freeze the complete block order before model work. Synchronize CUDA,
record wall/device interval and actual allocated/reserved/peak memory. Include
reparsing, orbit construction, tokenization, packing, transfers, backbone,
head, CE, host score extraction and aggregation. Exclude loading, candidate
generation, file writes and correctness validation. Retain first-call/JIT
setup separately from warm repeated scores. Budget3600s per model, no adaptive
extension. Preserve all raw windows before validity evaluation. Report each
block ratio, median/range and descriptive paired bootstrap interval; no
serving-throughput, overall inference or task-accuracy improvement claim.

Verify source/candidate/reference hashes again after each run. Model file
hashes are verified before loading and sizes/mtimes rechecked after. All failed
attempts/cost, baseline and optimized outputs, original snapshots and model
identity receipts are retained and synchronized before publication.
