# Model-dtype compatibility validation

Frozen before execution on 2026-10-14. This is a compatibility check, not a new
performance study. The original timing protocol, raw arrays, charts and reported
ratios remain unchanged.

- Host: authorized DGX Spark GB10; no unrelated GPU work may be interrupted.
- Runtime: the retained vLLM 0.29.0 / CUDA environment.
- Checkpoint: public Qwen2.5-1.5B-Instruct, revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; recheck every model-file hash.
- Two fresh engines: `dtype=auto` (expected to resolve to the checkpoint's
  bfloat16) and `dtype=float16`. All other engine options are shared.
- Use synchronous legacy scheduling, ordinary unquantized KV, no LoRA,
  speculation, or KV connector. Keep compilation and CUDA Graphs enabled.
- Compare each engine to its own native full scorer. Do not compare FP16 values
  to BF16 values or infer equal performance across formats.
- Fixed locked families: `add` and `add3` tools with two and three integer arguments, including
  EOS, scored under both policies. Validate all suffix token probabilities,
  sequence scores and all eleven decision rules. Also exercise a single batch
  containing marked and unmarked prompt-score requests.
- Complete Agent check: the existing first catalog and calendar DEVELOPMENT
  fixtures, four fixed candidate seeds, actual two-tool observations and final
  generation, once per policy in each engine. No held-out answers are read.
- Gate: every compared suffix has finite values and complete EOS coverage;
  maximum token error <=0.05 nats and sequence error <=0.5 nats, with all eleven
  locked decisions equal. Complete Agent candidate token IDs, candidates,
  selected calls, observations and final token IDs must match within each dtype.
  The same numerical/decision checks apply to every nonempty Agent-round
  candidate family, with explicit suffix length and final EOS validation.
- CPU hook and demo routing tests plus existing regression tests run on Spark.
  CUDA tests also run there. Save their complete logs separately.
- Freeze source hashes in the fresh run directory, preserve all failures and
  outputs, and refuse to overwrite an existing run. Terminate only owned engine
  processes. Any infrastructure retry requires its own amendment and run ID.

This gate establishes the tested compatibility path. It does not establish an
FP16 speedup, universal model support, task-quality equivalence between formats,
or a new held-out quality score.
