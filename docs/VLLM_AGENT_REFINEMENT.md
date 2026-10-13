# Prospective scoring-head and Agent-interface refinement, 2026-10-13

Declared after initial development, before any held-out Agent inference.
The initial protocol and all raw failures remain unchanged. BF16 API smoke
v1 stopped before model load due to an offline dependency priority error;
v2 stopped before inference because Transformers5.15 returns a structured
chat-template tokenization result; v3 validated API alignment but its original
BF16-output score head exceeded the 0.05/0.005 limits. The initial two-calendar
Agent development produced only final-answer JSON, no valid calls. Its empty
pools failed the nonempty validation gate. None is an accepted speed result.

## A different numerical execution mechanism

New scoring requests explicitly opt into **native FP32-output logits**, using
the existing vLLM0.29 LogitsProcessor head_dtype path. Torch multiplies BF16
hidden states and BF16 stored weights with out_dtype=float32. The entire model,
KV cache, attention/backbone and all ordinary candidate/final generation remain
native BF16. There is no FP32 model sidecar, FP32 weight copy, TF32 replacement
or batch-invariant backbone setting in this refinement.

The worker hook temporarily changes the existing logits processor only around
an explicitly marked prompt-logprob calculation and restores it in finally.
Mixed precision markers in a scoring batch are rejected. Untagged requests
and generation keep the stock BF16 output head. This changes arithmetic
execution; it does not relax any of the frozen error limits.

| Policy | Score head output | Score rows | Prefix reads |
| --- | --- | --- | --- |
| native_default | Stock BF16 | Native full prompt | Bypassed |
| native_full | Explicit native FP32 output, BF16 inputs/weights | Native full prompt | Bypassed |
| suffix_no_cache | Same FP32 output | All suffix predictors | Bypassed |
| bounded_apc | Same FP32 output | All suffix predictors | Capped before first predictor |

The unchanged 0.05 token-max /0.005 pooled-mean /0.5 sequence-max /99% eleven-rule
and joint decision gates compare the two suffix arms to the matched native_full
FP32-output score reference. Stock native_default errors are reported separately;
its old BF16-output gate failure is never relabeled. Stock BF16 remains an
actual full online Agent timing and quality control, rather than a paper number.

Markers: orbitcall_score_logit_dtype='float32' requests the precision path;
orbitcall_score_start=P-1 additionally requests suffix extraction. Default stock
uses neither. All model/storage/cache settings and complete family/EOS tokens
stay fixed. Independent audits retain all four score arrays on locked candidates.

## Correct the tool-loop interface before confirmation

The generic stage-specific user instruction now follows the original request
and real observations: perform the current tool round, return exactly name and
arguments, choose a currently available tool, and defer the final answer.
This instruction contains no argument values, lookup results or reference
answer. It is identical across policies. Final response instructions are
unchanged. Development uses the first lexicographic development input of each
family, two total, so both executable workflows are covered. The previous
two-calendar development is retained rather than repurposed.

The input-only48-task records and split are unchanged. data/agent_inputs_v2.json
records the final task-library source and is the confirmation authority.
The driver/interface changes are separately snapshotted in every new run.

## Confirmation changes declared now

Compare bounded_apc with **all three** systems controls in six ABBA/BAAB blocks
each:72 windows/model, not the earlier48. All four systems policies, vote4 and
greedy1 run the whole36-input test set. The primary production comparison is
stock native_default; same-output-precision and suffix/no-cache controls identify
the remaining contributions. Numeric gates precede accepted timing, and every
quality/workload difference is retained. No favorable held-out comparator is
selected. The same configuration is used for both checkpoints.

Budgets remain1800 seconds/new development and4200/model confirmation. If the
new fixed numerical gates fail, retain that result; do not change thresholds
or silently accept timings. New native implementation corrections use fresh
development run IDs and preserved source, and never use held-out outcomes.
