# OrbitCall frozen study protocol v1

Declared 2026-10-13 before model observations. This is a completed-project
evaluation, not a claim that the underlying marginal-likelihood or prefix-tree
principles are new. All computation and analysis runs on DGX Spark GB10.

## Research questions

1. On a fixed inference-generated candidate pool, does summing probabilities
   over a finite JSON argument-key permutation family improve call selection
   over strong aggregators using the same scores?
2. Can a packed shared-prefix tree score the same finite token sequences faster
   than independently batched and prompt-cache batched teacher forcing, with
   measured numerical error and selected-call agreement?
3. Does a tiled requested-token logprob reducer remove full log-softmax
   materialization without changing legal vocabulary support or normalization?

For candidate a, M(a) = sum over distinct token sequences s in S(a) of
P_model(s | prompt). S is compact JSON with fixed name/arguments envelope,
all permutations of 2 or 3 required flat scalar argument keys, and one explicit
end token. This is finite-family probability mass, not a complete semantic
posterior. Missing/null, scalar types and array order remain distinct. Duplicate
JSON keys and duplicate token sequences are rejected. Sum and arithmetic mean
have identical rankings when family sizes match; they are not separate controls.

## Models and inputs

- Public Qwen/Qwen2.5-3B-Instruct and Qwen/Qwen2.5-1.5B-Instruct, exact revisions
  and weight hashes recorded before inference. BF16, eval mode, no training.
- Gorilla/BFCL input revision 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8,
  BFCL_v4_simple_python and BFCL_v4_live_simple. Input-only eligibility: one
  function, 2–3 flat scalar parameters, all required, no optional/extra keys.
- Keep every eligible item, including parse/no-candidate failures. Expected
  input-only population: 84 + 36 = 120. Freeze IDs and a deterministic schema-
  grouped development/test split before any answer file is accessed. Aim for
  approximately 20 development and 100 test items; report actual counts.
- Four candidate samples per item: seeds 17, 29, 43, 71, temperature 0.7,
  top-p 0.9, maximum 128 output tokens. Greedy is a separate control. Retain
  all generated tokens/text, counts and invalid outputs. No external model API.

## Task-quality controls

Use the same deduplicated pool and all recorded sequence scores for canonical
schema order, development-selected fixed order, maximum over orders, arithmetic
mean logprob, length-normalized mean logprob, and finite-family probability sum.
Include semantic-majority voting on the same four samples, native greedy,
and additional-sampling majority as a compute-allocation control. Report its
actual time/token budget rather than claiming it is equal-cost automatically.
Ground truth is read only by a separate evaluation stage after candidate and
prediction hashes have been frozen. Use BFCL AST/value semantics where supported;
document unsupported cases, do not silently treat JSON validity as task accuracy.

Primary comparison is test accuracy versus the strongest equal-score control;
also report candidate coverage, JSON validity, model/task category breakdown,
paired differences and uncertainty at the question/schema level. Do not select
the best model or subset after seeing results. No universal quality guarantee.

## System comparisons and validity

Compare independent full-prompt batching, immutable prompt-cache batched suffix
scoring, and packed ancestor-masked prefix-tree scoring on identical inputs,
candidate sequences, weights, dtype and score normalization. Do not omit the
prompt-cache control. Preserve per-edge and sequence logprobs, maximum errors,
winner margins, call agreement and memory. A real-arithmetic tree identity does
not imply bitwise equality across GPU batch/attention shapes.

Freeze timing inputs from development without answers. Warm every method, then
use randomized ABBA paired windows with at least four independent blocks.
CUDA-event kernel times and synchronized complete wall times are separate.
Time parsing, packing, H2D transfers and result extraction for complete scorer
cost. Also report the complete candidate-generation plus ranking pipeline;
score-only savings are not claimed as full generation speedup.

Correctness gates precede timing: randomized small-tree enumeration; packed
tree vs full/cached sequence scoring on actual checkpoints; requested-token
logprob kernel vs torch full log-softmax, including extreme and odd vocabulary
sizes. Fail closed on NaN, unsupported masks, output mismatch or interference.

## Budget and disposition

Initial budget is 4 Spark GPU-hours for correctness, development, both models
and fixed test population, excluding download/setup. Record actual GPU/wall
cost; amend prospectively if infrastructure requires more. No result-driven
extra permutations, extra samples or reruns. Preserve negative quality results.
Publish the implementation, theorem/scope, baseline evidence and failures. Resume
claims use only verified project outcomes with explicit hardware/model scope.

## Setup record before generation, 2026-10-13

The exact input-side contract produced 119 eligible records (83 Python simple,
36 live simple), with 20 development and 99 test IDs. Freeze SHA256 is
1e8edf278caa6066c1b4c1fab07599cc8acb03fdf0037d3447603ff75916060c.
No answers were accessed, and no quality-dependent selection occurred.
The first real-checkpoint correctness launch stopped on the tokenizer's new
BatchEncoding return type before computing real-model comparisons. Its log is
retained; an API normalization repair uses a fresh scoring_gate_3b_v2 result.

### Precision amendment before any candidate generation or label access

The BF16 scoring gate did not satisfy numerical consistency: even independent
full scoring versus prompt-cache scoring differed by up to 0.75 nats/token on
the fixed unlabeled checkpoint fixture. The packed path was not accepted as an
exact scorer. An earlier custom-mask dtype API failure is also retained.
No candidate pool, quality evaluation, timing conclusion or gold access occurred.

Keep candidate generation at BF16/SDPA. Define all likelihood-scoring comparisons
in FP32/eager attention with TF32 disabled, at identical precision in every arm.
Use a new scoring_gate_3b_fp32_v1 artifact and do not relabel BF16 evidence.
This is a precision/implementation validity repair, not a positive quality
selection. Gate tolerance is 1e-3 nats per edge on the real fixture and 3e-5
on the randomized tiny FP32 model; retain measured errors and winner margins.
Claims will explicitly distinguish generation dtype and score dtype. Sampling
uses batch size 8, seed base+1000003*batch_index. The additional-sampling control
uses four extra seeds 101/113/127/139. Freeform-to-reformat uses four 64-token
analysis stages and four 64-token format stages, matching the direct four-sample
maximum generation-token budget but not automatically its measured time.

### Cache baseline amendment before any ranking, timing, or answer access

Source review of Transformers 5.15 found DynamicCache(ddp_cache_data=...) copies
the supplied expanded prefix through an empty-tensor concatenation. Initialize
fresh non-sliding DynamicLayer wrappers with immutable expanded prefix views,
including dtype/device/initialization metadata, for BOTH cached and tree arms.
Suffix update still uses the same HF concatenation. Reject other cache classes.
Revalidate with scoring_gate_3b_fp32_v3 and every actual candidate family. The
benchmark uses the entire family as the full/cache batch, not a smaller batch
that could artificially favor the packed path. These decisions precede any
scoring quality labels or timing observations; candidate generation is unchanged.

### Reference-quality amendment after a retained numerical failure, before gold/timing

The initial rank_qwen3b_20261013_v1 stopped at item0081/simple_python_279.
Full/cache maximum edge error was0.0014152527 nats and full/tree0.0013637543,
exceeding the unchanged1e-3 gate. All11 call winners agreed. Preserve that
failed run and its full cost; it is not a successful global fast-path gate.

Run new rank_*_20261013_reference_v2 studies on BOTH unchanged candidate pools,
all119 inputs, with every task-quality aggregator using full-prompt FP32 scores.
No extra samples, labels, permutation families or test-driven rules are added.
Fast-path error and winner comparisons remain mandatory diagnostics across the
whole population; their failures remain false and are reported separately from
valid reference prediction provenance. No failed item is dropped. Do not call
the packed path globally exact or the offline reference audit a cheap online
certificate. Frozen development timing rules, all benchmark arms and strict
numerical tolerances remain unchanged. This disposition separates the finite
statistical objective from a backend whose numeric gate failed; it does not
repair the failure by enlarging a tolerance.

### Retire packed timing; isolate the originally declared normalization mechanism

The first fixed3B development benchmark also failed its unchanged edge gate
before any timing: the retained simple_python_279 fixture had0.0015125275-nat
maximum error. No packed-versus-cached speed result is accepted. Preserve the
original benchmark directory, raw fixtures/source and all cost. Do not remove
the failed fixture, change dtype/tolerance, or relabel the failed study.

The separate RQ3 mechanism is now tested in new normalizer_*_20261013_v1 runs:
keep the SAME input-only fixture rule and every selected fixture, but use the
identical full FP32 backbone for Torch log-softmax, LSE, native cross-entropy and
Triton in every arm. Compare five direct pairs in four randomized ABBA blocks.
Capture the actual full-backbone first LM-head chunk for an independent reducer
study; native CE operates directly on one target per row. Whole-population full
Triton versus retained full-LSE scores receives a separate3e-5-nat audit using
the same stored token arrays, batch8 andhead32. No labels or extra samples are
accessed/added. This tests a different originally declared execution mechanism;
it supplies no repaired packed-path guarantee, packed speed, or full generation
speedup. Both reference-quality populations and all failed evidence remain.

### Infrastructure budget retry, 2026-10-13

The first3B full-normalizer study reached its600-second runtime ceiling after
78 of80 numerically validated scoring windows; it did not finish the planned
blocks or reducer stage. Retain that incomplete directory and cost, accepting
no final speed ratio from it. The fixed workload, rather than a quality/error
gate, exhausted the runtime allocation. No labels have been accessed.

Use new normalizer_*_20261013_budget_v2 directories with an explicitly declared
1200-second ceiling. Keep ALL four selected fixtures, four arms, five pairs,
four blocks, repeat counts, warmups, random seed, numerical tolerances, full
backbone precision and candidate/input hashes unchanged. Do not choose better
ratios from the retained incomplete attempt. This is an infrastructure-only
retry, charged cumulatively within the initial project allocation; it adds no
candidate samples, adjusts no scientific method and repairs no numerical gate.
