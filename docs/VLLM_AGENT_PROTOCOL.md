# BF16 vLLM Agent integration protocol, 2026-10-13

This is a new prospective study, separate from the historical FP32/Triton
prefix-tree results. No historical score-stage ratio is an Agent speedup.
The implementation and all numerical work run on the authorized DGX Spark
GB10, with the two unchanged pinned Qwen2.5-3B/1.5B-Instruct checkpoints.

## Runtime and three same-policy systems arms

Use the locally provisioned public vLLM **0.29.0** package, Torch 2.13.0+cu130,
Transformers 5.15.0, native BF16 model and KV storage, raw full-vocabulary
logprobs, TP1 and the legacy V1 runner. Async scheduling is disabled for the
declared compatibility hook; eager-only execution is not the timing baseline.
Record actual runtime source hashes, selected attention backend, compilation
and CUDA Graph settings, pinned checkpoint receipts and all source snapshots.

| Arm | Probabilities requested | Prefix reads during scoring |
| --- | --- | --- |
| `native_full` | Native prompt_logprobs=0 over the full prompt plus suffix; use all suffix targets | Native default bypass |
| `suffix_no_cache` | Plugin computes only causal rows predicting suffix tokens | Explicitly bypassed; strong primary control |
| `bounded_apc` | Identical native BF16 suffix extraction | Read at most P-1 tokens, rounded down to available complete blocks |

The plugin is an explicit `vllm.general_plugins` entry point, enabled only by
ORBITCALL_VLLM_SCORING=1/VLLM_USE_V2_MODEL_RUNNER=0 and marked requests. Untagged
generation delegates native code. It uses native compute_logits, full-vocabulary
logprob normalization, gathers every actual suffix target, and retains EOS.
Unscored prompt positions are explicit None. If P is prompt length, hidden
row P-1 predicts the first suffix token. Reading KV beyond that row is forbidden.
Native generation in every arm has ordinary APC enabled. Scoring reuses only
the KV genuinely populated earlier in that same online episode, within the
timer. There is no uncounted warmup or prompt preparation in online windows.

Exact engine kwargs are recorded before launch: BF16/auto KV, memory fraction
0.65, context4096, max sequences64, max batched tokens8192, chunked prefill and
APC enabled, async schedulingFalse, enforce_eagerFalse, no remote model code,
seed20261013 and raw_logprobs. Do not substitute the Torch FP32 implementation,
TF32 kernels, packed JSON-prefix tree, quantized KV, a new checkpoint or runtime.

## Input-only task freeze and actual workflow

48 original MIT local test inputs are frozen in data/agent_inputs_v2.json:
12 development and36 test, grouped by catalog item or region/year policy.
The earlier v1 input freeze is preserved as a preflight source-version record;
its inputs were not evaluated. The new v2 records the final task-module hash.
Fixture prices and closure dates are original local data, not live merchant
prices or public holidays. No network service or artificial tool sleep exists.

- Catalog: catalog_lookup(item_id,store) executes a lookup, then
  order_total(unit_price_cents,quantity,shipping_cents) executes arithmetic.
- Calendar: calendar_policy(region,year) returns actual fixture policy, then
  business_date(start_date,business_days,calendar_id) executes a date loop.
- Each stage exposes both tools available in that round. Candidate parsing
  validates current-stage schemas. Round2 receives the actual round1 result;
  final response receives both executed observations. No oracle supplies calls.
- Two rounds and final generation always run, including explicit null/tool
  errors. All such cases remain in quality and timing denominators.

Each tool round generates four fresh samples with seeds17/29/43/71,
temperature0.7, top_p0.9, at most128 tokens. Strict parsing and semantic
deduplication retain invalid raw outputs. Each valid call is scored in every
two/three-key permutation, compact name/arguments envelope, plus one EOS.
Selection is finite-family probability mass with semantic-key tie breaking.
Eleven diagnostics use alphabetical canonical argument order and the six
fixed index permutations, not the historical schema-order policy. No paths
are pruned and every vocabulary logit remains in the denominator.
Final generation is greedy, seed17, at most64 tokens. Ordinary vote4 and
greedy1 (one greedy tool sample, cap128) are separately reported utility controls.
They have different compute allocations and are not same-workload speed arms.

## Development, numerical audit and fixed timing

The actual-runtime Add-call smoke is an API development check, with all source,
raw arrays and failures retained. Then the first two lexicographic development
inputs validate the entire Agent loop with the fixed implementation. No tuning
from held-out observations or gate-threshold changes is allowed.

New BF16 fidelity limits are declared BEFORE inference: maximum absolute token
logprob error0.05 nats; pooled token-weighted mean0.005 nats; maximum complete
sequence error0.5 nats; at least99% agreement for each of eleven rules and
their joint agreement over nonempty locked pools. These are empirical bounds,
not bitwise equality, task-accuracy guarantees or an inherited FP32 certificate.

All36 test inputs run the actual Agent online under five policies. Native-full
candidate families at both rounds are frozen before separate locked audits of
the two suffix arms. Audits seed a fresh prompt cache explicitly and record
that work outside the online timers; this is numerical validation only.
Every suffix token/EOS, error array, score sum, winner and null pool is retained.
Numerical failure prevents accepted confirmation timing. Policy predictions
freeze before the audit so task outcomes can still be evaluated after a failure.

Eight timing IDs are frozen from inputs before any outputs: first four
lexicographic test IDs in each family. Process them in fixed cohorts of four,
with batched candidate/final inference, actual local tool execution and separate
scoring calls per task. A cache reset and fresh per-window/per-episode salt
prevent cross-window reuse. Within-episode salts are shared by generation,
scoring and final response. Each window reruns the complete online workflow.

For each model, compare native_full and suffix_no_cache to bounded_apc in six
ABBA/BAAB blocks each, randomization seed271828;48 total windows. Freeze the
schedule before engine launch. The primary metric is completion wall time of
the entire eight-episode workload, including generation, parsing, packing,
scoring, tools, histories, final inference and counted cache resets. Record
cohort service time and queued pool latency separately. This is a bounded
four-episode concurrency workload, not steady-state serving throughput.
Model startup/loading/graph compilation are recorded separately.

Preserve all raw windows before checks. Report six block mean(A)/mean(B)
ratios, median and range, component times, tokens, candidate/trajectory/final
agreement, and repeated A/A and B/B consistency. Adaptive divergences are not
filtered. Exact-workload comparability and task-success parity determine
whether a result may be promoted; otherwise report the observed online
time with its workload/quality limitations, without a matched speed claim.

Budgets are1800 process seconds/development run and4200/model confirmation,
including initialization, audit, online policies, timing and artifact writing.
Check the process watchdog, retain every failure, and use fresh run IDs for
justified infrastructure retries. Do not overwrite or silently extend budgets.

## Separate evaluation and publication

Freeze prediction JSON and SHA256 receipt before evaluate_after_freeze may
compute reference outcomes. Report exact task success, per-round correctness,
invalid/null/truncated output counts and final-answer validity over all36 test
inputs for every policy. Local fixtures do not establish a public leaderboard
score or general Agent quality. Numerical winner fidelity is not correctness.

An independent Spark CPU analyzer rechecks raw arrays, hashes, schedule and
ratios; publication includes raw evidence, retained failed attempts and cost.
Figures are generated from that retained analysis on Spark. Update README and
resume only with supported Agent/BF16 results, with historical FP32 measurements
clearly separated. The adapted plugin retains its Apache-2.0 attribution.
