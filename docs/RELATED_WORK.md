# Attribution, related work, and scope

Reviewed from public primary sources on 2026-10-13. This is a bounded research
and engineering study, not an assertion that its mathematical ingredients are
new. No private research source, method, checkpoint, or log is used here.

## What OrbitCall studies

Given a candidate call generated without gold answers, OrbitCall sums model
probabilities over a declared finite family: compact JSON, fixed `name` then
`arguments` envelope, every ordering of two or three required scalar argument
keys, and one explicit tokenizer-supplied terminator. All argument values are
unchanged. This family omits other spellings, whitespace, envelopes, numerical
representations, and semantically equivalent argument values. Its probability
mass is not the complete posterior over a program's meaning.

The statistical question is whether this finite-family score selects better
calls than strong aggregators using exactly the same candidate pool and scores.
The original systems question compared shared-prefix tree scoring with
independent and immutable prompt-cache batching on a single GB10. Its unchanged
numerical gates failed, so the cache/tree performance study is retired and
supplies no accepted speed claim. The separate final systems study keeps the
same full FP32 backbone and compares Torch log-softmax, LSE, direct native CE
and Triton requested-token normalization on unchanged development fixtures.
Task-quality predictions use full FP32 reference scores; cache/tree outputs
remain reported diagnostics. Quality and complete-scorer performance conclusions
require completed, reviewed reports and are not assumed here.

A finite-family sum and arithmetic mean probability rank identically when all
candidates have the same family size; they are not two independent controls.
Task accuracy, resident-reducer runtime, complete scorer runtime and full
pipeline runtime must be reported separately. The offline full-population
Triton audit is empirical numerical agreement, not an online certificate or
a replacement for task-quality evaluation.

## Nearby work and the boundary of our claims

| Public primary source | Relevant established idea | OrbitCall boundary |
| --- | --- | --- |
| [Long et al., *LLMs Are Biased Towards Output Formats!* (NAACL 2025)](https://aclanthology.org/2025.naacl-long.15/) | LLM outputs depend on formatting; format bias can be evaluated and mitigated. | We do not claim to discover format bias. We test a fixed inference-only argument-order family with same-score controls. |
| [*Structure Tax: How Structured Output affects LLMs Performance*](https://arxiv.org/abs/2610.12056) | Structured field ordering can affect model performance. | Field-order sensitivity is prior work. Our orbit permutes unchanged tool arguments, rather than changing reasoning placement or training. |
| [*Increasing Probability Mass on Answer Choices Does Not Always Improve Accuracy* (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.522/) | More answer probability mass need not yield better answer accuracy. | No quality guarantee follows from the finite mass identity. The frozen test comparison can be negative. |
| [Park et al., *Grammar-Aligned Decoding* (NeurIPS 2024)](https://arxiv.org/abs/2405.21047) | Grammar constraints and conditioning on valid outputs require attention to the underlying distribution. | OrbitCall reranks an inference-generated finite pool; it neither samples the complete grammar-conditioned distribution nor replaces grammar alignment methods. |
| [GenLM, *Syntactic and Semantic Control for Large Language Models via Sequential Monte Carlo* (ICLR 2025), official implementation](https://github.com/genlm/control-iclr-2025) and [GenLM Control](https://github.com/genlm/genlm-control) | Probabilistic inference with programmable constraints is an existing approach. | Summing alternative strings and exploiting common prefixes are established ideas. OrbitCall is a restricted finite-family benchmark/implementation with explicit same-budget controls. |
| [Cai et al., *Medusa* (2024)](https://arxiv.org/abs/2401.10774) | Tree attention can evaluate multiple candidate continuations in one packed structure. | Ancestor-masked attention is not a new primitive here. We use it for teacher-forced orbit scoring without extra decoding heads or training. |
| [Liger Kernel, official repository](https://github.com/linkedin/Liger-Kernel) | Optimized cross-entropy and fused linear cross-entropy kernels already reduce intermediate memory/compute. | Exact target-token logprob reduction is established mathematics. Avoiding a full log-softmax temporary is not, by itself, an algorithmic novelty claim; compare a gathered-logit plus `torch.logsumexp` control where applicable. |

This review is not an exhaustive novelty proof. Similar exact-string reranking,
semantic marginalization, constrained inference, or tree scoring may exist in
other applications. Any publication claim must be narrower than these known
ingredients and supported by the completed experiments.

## Model licenses remain separate

The pinned
[Qwen2.5-3B-Instruct checkpoint](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE)
uses the Qwen Research License Agreement with noncommercial-use terms. The
pinned
[Qwen2.5-1.5B-Instruct checkpoint](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE)
uses Apache-2.0. Public availability does not make these permissions identical.
OrbitCall's MIT license applies to its original code and documentation; it
does not relicense model weights or BFCL materials. The release retains public
revision/license provenance and verification receipts without distributing
checkpoint weights or downloaded answer files.

## BFCL provenance and licensing

The inputs are from the [official Gorilla repository](https://github.com/ShishirPatil/gorilla),
frozen at commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. Although their
suffix is `.json`, the files are JSONL. The
[pinned dataset card](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data/README.md)
declares Apache-2.0, consistent with the pinned repository root license. The
download script retains that license in `metadata/gorilla/LICENSE` and writes
revision, URL, license, citation, and byte-hash provenance to
`metadata/bfcl_inputs_download.json`.

| Pinned asset | SHA256 |
| --- | --- |
| [`BFCL_v4_simple_python.json`](https://raw.githubusercontent.com/ShishirPatil/gorilla/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_simple_python.json) | `82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991` |
| [`BFCL_v4_live_simple.json`](https://raw.githubusercontent.com/ShishirPatil/gorilla/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_simple.json) | `1af2ac87dca47556db7b7e37e51e28b459a38b594e3c7b3c792b4903598ca0c4` |
| [Gorilla root `LICENSE`](https://raw.githubusercontent.com/ShishirPatil/gorilla/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/LICENSE) | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

Cite the benchmark paper:
[Patil et al., *The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models*, ICML 2025](https://proceedings.mlr.press/v267/patil25a.html).
This project is not the full BFCL benchmark or an official leaderboard submission.

The staging script's network allowlist contains only these two input assets and
the license. It does not download answers. Input eligibility uses one function,
two/three supported flat scalar parameters, all required, and no optional/extra
keys. Unsupported schema constraints are excluded with a recorded input-only
reason. Group identity uses the named structural scalar schema, ignoring
descriptions and property order. A public fixed salt orders groups; deterministic
subset-sum selects the development size closest to 20 while keeping schemas
intact, with smaller sizes winning ties. The input freeze records the actual
counts, every eligible ID, group allocation, source hashes, and every exclusion.
Expected counts in the protocol are a pre-experiment expectation, not a result.

## Parsing and semantic identity

The parser accepts a bare JSON object or one complete native Qwen `<tool_call>`
wrapper. It does not repair Markdown, extract JSON from explanations, accept
stringified argument dictionaries, or choose one call from multiple outputs.
Duplicate keys at any depth, nonfinite numbers, and unpaired Unicode surrogates
are invalid. The only envelope fields are `name` and `arguments`.

Identity ignores object-key order, preserves array order, and distinguishes
missing, null, booleans, integers, floating-point literals, and strings. Decimal
format variants that parse to the same finite float are canonicalized by Python
JSON encoding; signed floating zero remains distinct. This typed identity is a
declared conservative choice, not universal programming-language equivalence.
Integer literals may satisfy a float/number schema without being coerced. Every
valid scored candidate must still contain precisely the required schema keys.
The complete string orbit has 2! or 3! members. The tokenizer must append one
terminator and check token-sequence uniqueness before scoring.

## Separate scalar AST/value evaluator

The public reference is the
[pinned BFCL AST checker](https://github.com/ShishirPatil/gorilla/blob/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/eval_checker/ast_eval/ast_checker.py).
Our independently implemented evaluator supports one exact function and all
required flat literal scalar arguments. It checks allowed values, with BFCL's
scalar-string normalization and integer acceptance under float schemas.
`number` and `bool` schema aliases are explicit extensions. It does not implement
variable references, optional parameters, containers, multiple/parallel calls,
language conversion, or executable evaluation. Unsupported gold is retained
and reported, never used to revise the input split or candidate pool. Do not
label this restricted metric as the official full BFCL score.

`load_gold_after_freeze` performs no network requests and verifies exact SHA256
receipts for candidates and predictions before opening answer files. The caller
must retain the independently saved pre-gold receipts; hashes computed after
labels are available would not establish isolation. Gold cannot add candidates,
argument values, orders, or additional samples. A valid JSON call with wrong
values is incorrect. An eligible item with zero candidates is retained as a
failure; it is not dropped from the task-quality denominator.

Windows is used for editing and public asset downloads only. Unit tests, input
eligibility/split creation, numerical checks, inference, timing, and statistical
analysis must execute on the authorized DGX Spark under `PROTOCOL.md`.
