# Prospective stock-BF16 normalization-only Agent study

Declared 2026-10-13 before any held-out inference. This replaces the failed
APC mechanism, without overwriting the original protocol, refinement or runs.
Development v2 preserved an APC maximum token deviation of 0.1837387085 nats,
above the unchanged 0.05 limit. FP32-output suffix/no-cache scored exactly on
its two nonempty development pools; it is not a stock-BF16 acceleration result.

## A materially different execution path

`native_shape_suffix` uses stock BF16 weights, activations, KV and output head.
It bypasses prefix reads. Native per-request/chunk LM-head row counts and
hidden-state slice strides are preserved, including prefix-only chunks. Only
after the entire native BF16 head projection are logits cropped to rows that
predict suffix targets. Native full-vocabulary normalization, actual-target
gather and compact D2H follow. Every suffix token and EOS is scored; prefix
positions are explicit unscored None. No restricted vocabulary, approximate
probability, reduced serialization family, FP32 head or model sidecar is used.

The request markers are `orbitcall_score_start=P-1` and
`orbitcall_score_mode=native_shape_suffix`, with explicit
`skip_reading_prefix_cache=True`. The prior APC/FP32 modes remain experimental
code only. Ordinary generation and `native_default` delegate unmodified vLLM.

The optimization can save log-softmax, rank calculation, device transfer and
frontend objects. It does not claim to avoid transformer or LM-head work.
Numerical equality is an empirical check, not a theorem about CUDA kernels.

## Frozen controls, tasks and decision gates

Two system policies are `native_default` and `native_shape_suffix`, both with
stock BF16 score outputs. `vote4` and `greedy1` remain quality/cost controls.
Use the unchanged 48 input-only fixtures and 12-dev/36-test group split from
`data/agent_inputs_v2.json`, with two local tool rounds and final generation.
Original development failures and their elapsed costs remain retained.

Development uses one first-lexicographic ID per family. Retained v2 outputs
showed calendar calls with parameters at the top level or under `parameters`,
not the required `arguments` object, and stopped normally (no truncation).
The new generic last-user directive explicitly illustrates only the structural
envelope {"name":"<available_tool>","arguments":{}} and requires every tool
parameter nested inside arguments. It supplies no task-specific value, actual
tool observation or gold answer. No output repair is performed. All policies
use this identical `explicit_nested_arguments_v3` interface. No held-out
answers or outputs are consulted.

Locked score audits compare the suffix arm directly with stock BF16 arrays
over the identical complete candidate/serialization/token/EOS families.
Limits stay 0.05 maximum token nats, 0.005 pooled mean token nats, 0.5 maximum
sequence nats and at least 99% agreement for each of eleven rules and jointly.
Empty pools are reported and excluded only from nonempty score-agreement
denominators; they remain in every task-quality and online-timing denominator.
The population must contain a nonempty pool. A failed gate blocks accepted
timing; no threshold is changed based on observed results.

## Complete online timing

For each unchanged pinned Qwen2.5-3B/1.5B checkpoint, run all four policies on
the 36 held-out inputs. Freeze actual calls, observations, generated text/token
IDs and final responses before deterministic gold evaluation. No repairs,
oracle argument insertion or output filtering are allowed.

Then run six paired ABBA/BAAB blocks, 24 full online windows per model, using
the same eight preselected fixture IDs. Seed271828 fixes order before inference.
Each window starts a counted cache reset and runs actual candidate generation,
strict parsing, complete-family scoring/selection, real local tool execution,
observation feedback, second round and final generation. Cohorts contain four
requests; each scoring policy samples four candidates per tool round. Tools
have no artificial latency, sleep, network or reference-answer injection.
The whole-window wall clock is primary; nested scoring times are secondary.
Loading, first compilation and offline locked audits are retained as costs but
excluded from steady online timing. Adaptive branches are retained in full;
same work/quality and A/A, B/B, A/B text/token/trajectory equality are reported.

Engine settings remain BF16/auto KV, TP1, memory fraction0.65, context4096,
max sequences64, max batched tokens8192, chunked prefill/APC for generation,
synchronous legacy runner, normal torch.compile/CUDA Graphs and raw logprobs.
Record package versions, runtime source and native-library hashes, actual
attention backend from retained logs, model identities, source snapshots and
all failures. Budgets stay1800s per development and4200s per confirmation.
Independent CPU analysis and plots run on Spark. This is a reproducible local
closed-loop workflow benchmark, not a production Agent leaderboard or broad
deployment-throughput claim.
