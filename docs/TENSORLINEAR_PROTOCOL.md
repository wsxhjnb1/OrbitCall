# Separate FP32 Tensor Core development study

Declared before the first kernel tests or model observations, 2026-10-13.
The failed initial bulk/SDPA/chunk screen is retired and retained. This is a
different mechanism: FP32 input/weight/output matrix multiplication using the
established three-product TF32 algorithm or a fixed six-product residual
decomposition. Neither is bitwise IEEE GEMM or a new probability algorithm.

Run `tensorlinear_discovery_3b_20261013_v1`. Fixed six configurations in this
order: precision `{tf32x3, tf32x6}` x scope `{head, backbone, all}`. One fixed
Triton tile M32/N64/K32, four warps, three stages, no autotuning. Weights are
unchanged and FP32; eager attention, original row packing, head32, batch policy
and native cross-entropy are identical in both arms. All ordinary Torch GEMMs
retain TF32 disabled. The explicit Tensor Core module is an approximation and
must pass empirical score gates; no automatic low-precision fallback exists.

Discovery uses the same answer-free first-four 3B development fixtures and
whole-family batching as the initial screen. Record a fresh IEEE native CE
reference, each exact score vector, <=3e-5 nats/token, EOS-inclusive
sequence-error length bounds and all eleven winners. Invalid arms are retained
and not timed. A valid arm receives three complete pool measurements after its
gate (which includes compilation). Lowest median wins, lexicographic config
breaks exact ties. This is discovery, not a confirmatory speed estimate. Freeze
one configuration and its source inventory before any held-out observation.
Apply it unchanged to 1.5B; no tuning after a held-out failure.

Before discovery the Spark CPU freeze `acceleration_input_freeze_v1.json`
retains first up-to-eight lexicographic eligible test IDs separately per model.
The selection uses primary-candidate count only, not gold, score, numerical
failure, length or runtime. Each model receives a whole119 batch8 audit against
its immutable full-FP32/eager/LSE reference, retaining zero-family records,
complete token arrays, all eleven winners and the unchanged 3e-5 gate. A failed
audit accepts no timing for that model. Fresh same-whole-batch references
precede six randomized ABBA blocks (seed271828) on its fixed test pool; all
windows are checked again. Both models must pass before any two-model claim.
No tolerance increase, changed failed configuration or removed input is allowed.

Discovery ceiling1800 seconds; audit plus timing ceiling3600 seconds/model,
including load/hash/compile/gates/setup. All stage wall costs and failures are
retained. This measures complete parsing/tokenization/scoring/extraction/ranking,
excluding checkpoint loading, validation, artifact I/O and candidate generation.
It cannot establish serving throughput or full generation-pipeline acceleration.
Whole-host occupancy checks must pass; unrelated jobs are never interrupted.

Every stage archives its own exact source. Historical references retain their
original source; the sole released tree-to-full default change is independently
verified. Publication retains the initial negative screen and this study's full
failures, source/weight/input hashes, all windows and actual cost.
