# Independent exact-shape CUDA Graph scoring study

Declared before Graph kernel tests/model observations, 2026-10-13. The initial
12-arm packing screen and six-arm Tensor Core screen are retired negatives.
Their numerical thresholds and raw failures are unchanged. CUDA Graph execution
is a different mechanism: replay the unchanged IEEE-FP32 full backbone with the
same actual batch/width, eager causal/padding masks and ordinary native linears.
No extra padding, approximation, KV sharing or changed family is introduced.

One fixed configuration: backbone-only capture, head32/native cross-entropy
outside capture, three side-stream warmups per new exact shape, explicit
fallback='raise', retention ceiling24GiB. Development cachemax8; whole119 audit
max2; after clearing its bounded cache, confirmation max16 for the fixed test
pool. Capacity is a predeclared lifecycle choice, not an outcome-selected shape
bucket. Report every capture, warmup, eviction, cold setup, retained bytes and
replay. Capture reservations are bounded after capture, not a transient process
memory guarantee. Model weights/allocator caches/eager head memory are separate.

The strong native comparator uses graphs_enabled=False with identical bulk
CPU packing, static GPU input layout, mask, positions, model and normalization.
The unchanged original full eager32 scorer remains a numerical anchor. Compare
both alternative paths against fresh original whole-family development vectors
at <=3e-5 nats/token, length-scaled EOS sequence bounds and all eleven winners.
Use the same first-four 3B development fixtures. A valid configuration receives
three warm measurements per arm, paired alternating native/graph order; these
are discovery only. Freeze the single valid configuration before held-out
observations, even if its descriptive development runtime is slower.

Then reuse it unchanged on both models. Each model receives all119 batch8 audit
against frozen full eager32/LSE vectors, with explicit zero-family dispositions.
Any failed nonempty path fails that model and produces no accepted speed claim.
Fresh timing gates compare the graph and strong native against original native
at whole-family shapes before six ABBA/BAAB blocks (seed271828). Test IDs are
the immutable input-only first-up-to-eight eligible IDs frozen before these new
discovery stages in metadata/acceleration_input_freeze_v1.json. No answers or
new sampling, item exclusion or tolerance changes. Both models must pass before
a combined result is accepted; per-model audit precedes per-model timing.

Complete timing includes reparsing, enumeration, prompt/suffix tokenization,
CPU packing, H2D, all full computation, head and normalization, D2H and ranking.
Checkpoint load, artifact I/O, validation and candidate generation are outside
the timed body. Pre-capture fixed shapes during setup. Measured windows must
have zero new captures/evictions/fallbacks: accepted timing is specifically
**warm exact-shape reuse**, with cold setup and first-call costs separately
reported. An unseen shape incurs capture, not an automatic universal speedup.
Native control preallocated buffers are also warmed. Retain all24 raw windows,
outputs, errors, shape/work counts, CUDA brackets, memory and six block ratios;
the brackets include host gaps and are not GPU-busy time. Report median/range
and block bootstrap uncertainty without treating resamples as new trials.

Discovery1200 seconds; audit+timing3600 seconds/model, all-inclusive stage
ceilings. All measured wall costs including failed attempts remain cumulative.
Only the authorized Spark GB10 performs tests/model work/analysis. Whole-host
occupancy checks cannot stop unrelated jobs. Snapshot/archive exact source and
verify model/input/candidate/historical reference receipts before each stage.

Primary reference: PyTorch's established CUDA Graph execution API and memory
pool semantics, https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graphs.
This is an execution optimization, not a new likelihood algorithm, quality gain
or proof of bitwise equivalence. Empirical all-population gates remain required.

## Infrastructure correction before checkpoint discovery

The tiny-random-network smoke `graph_runtime_smoke_v1` failed before a replay:
the pinned HF eager mask builder creates a CUDA scalar from a non-pinned CPU
tensor, which the CUDA Graph capture rejects. Preserve its failure/source/log.
The corrected implementation invokes the exact native mask builder outside
capture, owns a static4D mask and copies its values before every forward/replay.
The graph and disabled native control share this same setup. Full-score timing
includes mask construction/copy; only the unchanged backbone is captured.
Masks retain exact0/FP32-min values, absolute positions and padding semantics.
No numerical tolerance, weight, shape, sample or discovery fixture changes.
`graph_runtime_smoke_v2` is an infrastructure retry on a tiny random network;
checkpoint discovery has not yet launched or produced an observation.

## Stronger confirmatory control before held-out work

While the single graph configuration remains fixed, confirmation additionally
times the unchanged original full eager32 scorer. This comparison is declared
before the first held-out numerical/timing observation. Two independent pairs:
disabled-graph native/graph and original legacy/graph, six ABBA/BAAB blocks per
pair (48 complete windows/model), all frozen in the run manifest before model
scoring. Retain both direct ratios; do not infer improvement over the original
pipeline from the disabled-graph comparison alone. This stronger control does
not change the configuration or select fixtures from any new output.
