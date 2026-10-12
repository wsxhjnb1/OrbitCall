# Numerical validity and benchmark scope

OrbitCall computes the declared finite-family objective with full-vocabulary
normalization. The packed tree has a real-arithmetic identity to independent
teacher forcing under the assumptions in [THEORY.md](THEORY.md). The GPU
implementation is evaluated with measured floating-point agreement and selected
call agreement; a failed gate remains a failure even when selected calls agree.
Neither statement promises bitwise equality or universally correct tool
execution. The optional probability-table certificate is a third, separate
guarantee with a supplied-table scope.

## Precision decision and retained evidence

The initial BF16 native scoring fixture failed before candidate generation or
answer access: full-prompt and prompt-cache scores differed by up to **0.75 nats
per token**. The failed path is not accepted as an exact scorer and is not
relabeled as passing. Earlier tokenizer/mask API failures and their logs are
also retained. The prospective amendment in [PROTOCOL.md](../PROTOCOL.md) keeps
candidate generation at BF16/SDPA but defines every likelihood-scoring arm as
FP32/eager attention, with TF32 disabled. FP32 scoring is part of the measured
pipeline cost; it is not a free correction to BF16 generation.

The retained pre-generation Qwen2.5-3B fixture record
`runs/scoring_gate_3b_fp32_v1.json` reports:

| Comparison with full-prompt scoring | Maximum token error, nats | Maximum sequence error, nats | Top sequence |
| --- | ---: | ---: | --- |
| Prompt-cache FP32 | 6.866e-5 | at most 1.553e-4 | Same |
| Packed-tree FP32 | 6.294e-5 | at most 1.553e-4 | Same |

The maximum sequence value above is the reported maximum across the two
alternatives, not an independently recomputed value for each row. The tiny
FP32 fixture reported a maximum token error of approximately **4.77e-7**.
Recorded Spark tests passed 56 serializer/BFCL/trie cases and 159
logprob/certificate cases, including CUDA reducer tests and finite random-tree
controls. All numerical execution occurred on Spark. Source review on Windows
does not add a numerical observation.

These are initialization fixtures, not a guarantee covering every generated
family, checkpoint, sequence length, precision, attention backend, or library
version. The initial fixture script recorded top-sequence agreement without
requiring it and lacked explicit shape checks. The reviewed script now checks
shape, finite values, sequence sums, top-sequence agreement, and the two real
fixture families' mass winner. The earlier artifact does not retroactively
inherit those stronger checks. Fresh actual-family gates and their source
hashes are required for final evidence.

## What the three scorers compute

The probability event is defined on exact token arrays: the rendered prompt
tokens followed by each independently tokenized compact JSON serialization and
one explicit EOS. These arrays are identical in every arm. Fragment
tokenization is not generally equal to tokenization of concatenated raw text;
the project does not infer equality from a string prefix. Duplicate token
paths and interior configured stop tokens are rejected.

For suffix `s` of length `T` including EOS and prompt length `P`:

| Path | Backbone input | Hidden states projected to vocabulary |
| --- | --- | --- |
| Full | `prompt + s[:-1]`, causal right-padded batches | Prompt's last position, then each nonterminal suffix position |
| Cached | One prompt prefill, then `s[:-1]` with a new cache wrapper | Shared prompt last position, then each nonterminal suffix position |
| Tree | One prompt prefill, then unique predictive suffix parents | Prompt last position plus one row per unique predictive parent |

The cached/tree implementations remain documented and their failed diagnostic
evidence is preserved. The final normalization study uses the **full** path in
every arm; it does not use branch packing, prompt-cache reuse or deduplicated
parent projection as an accepted performance mechanism.

The prompt's last hidden state predicts the first suffix token. A suffix node
predicts each of its child tokens. EOS is a requested target at its parent's
logits; a terminal EOS node with no children is not fed into the backbone.
Thus EOS contributes to every complete sequence probability even though its
unused hidden state is omitted. All arms omit the same dead-leaf work.

`TokenTrie` assigns one node to each distinct token prefix. A packed node at
depth `d`, starting at one, receives position ID `P + d - 1`. Siblings therefore
can have equal rotary position IDs while occupying different packed columns.
This is necessary: packed column index is not semantic sequence position.
The active-node mask is sliced in both its query and key dimensions to the
same active-node order. It admits the entire prompt and only the node's own
ancestors, including itself; sibling and unrelated-branch keys are excluded.

The pinned [Transformers 5.15.0 Qwen2 implementation](https://github.com/huggingface/transformers/blob/v5.15.0/src/transformers/models/qwen2/modeling_qwen2.py)
uses explicit position IDs for rotary embeddings and updates a supplied cache
even when `use_cache=False`. Its tokenwise normalization and feed-forward
operations meet the tree identity's assumptions. In the pinned
[mask implementation](https://github.com/huggingface/transformers/blob/v5.15.0/src/transformers/masking_utils.py),
an already constructed four-dimensional mask is returned unchanged. These
implementation details need rechecking after dependency upgrades. Current
experimental support concerns the declared non-sliding Qwen2 checkpoints;
arbitrary architectures, hybrid/sliding caches, or custom attention interfaces
are not covered by the fixture evidence.

The eager attention operation still forms dense scores against the packed
keys, including entries later masked out. A lower unique-node count is not a
proof of proportionally lower attention work. Dense mask construction costs
`O(Q * (P + Q))` for `Q` active suffix nodes, and branch shape can make packed
attention expensive. Reusing predictive parent rows reduces projection work
in the packed implementation, but its retired numerical gates do not support
an accepted runtime benefit. The full-backbone normalization study retains
all independent predictive rows and the same output-projection work.

## Cache immutability and its cost

Each cached suffix group receives a fresh `DynamicCache` wrapper around the
saved prompt pairs; no wrapper with a previous suffix is reused. Each layer is
required to have exactly the pinned non-sliding `DynamicLayer` type and the
wrapper's layer count must match the saved pairs. The layer's keys and values
are initialized directly with expanded immutable prompt views, with dtype,
device, and initialization metadata set. In the pinned
[DynamicCache/DynamicLayer source](https://github.com/huggingface/transformers/blob/v5.15.0/src/transformers/cache_utils.py),
`update` concatenates tensors and reassigns layer references. It does not
overwrite the retained prompt tensor storage. This preserves the original
prompt pairs across groups. The behavior must not be assumed for an in-place
static cache, subclass, sliding layer, or changed dependency implementation.

The original `ddp_cache_data` constructor copied the expanded prompt KV tensors
through concatenation with an empty tensor. Before actual-family scoring,
timing, or answer access, the declared cache-baseline amendment removed that
redundant initialization copy for **both cached and tree** arms. A fresh stronger
fixture gate, `scoring_gate_3b_fp32_v3`, and actual-family checks are required;
the earlier numerical table above predates this amendment. Suffix update still
concatenates the prompt views and suffix KV into new storage. Those allocations
and copies were part of the original complete-scorer workload. Both cached and
tree arms use the same initialization and suffix-allocation policy. This is an
**immutable prefix-view
wrapper over the pinned Hugging Face DynamicLayer implementation**, not an
ideal paged engine, persistent cross-request prompt cache, or a zero-copy
attention operation. Accessing these layer fields is version-specific and must
fail closed when the pinned layout is unsupported.

## Normalization and invalidity

For parent `i` and requested target `j`, every reducer uses the complete model
vocabulary in the denominator:

```text
m_i = max_v logits[i, v]
logprob(i, j) = (logits[i, j] - m_i)
                - log(sum_v exp(logits[i, v] - m_i))
```

Requested edges may share a parent. Parent logits and normalizers are computed
once; selecting several targets does not restrict vocabulary support. The
Triton implementation uses FP32 tile statistics, an FP32 row merge, and target
gathers. Its reduction tree and exponential implementation can differ from
Torch's. Removing a full log-softmax output is a storage optimization, not a
new probability identity or an accuracy improvement.

The final controls are full Torch `log_softmax`, max-centered Torch
`logsumexp`, negative native cross-entropy and Triton. The full backbone supplies
one requested target for each independent predictive row. Native CE therefore
operates **directly on those targets**, with no shared-parent anchor adapter.
The retained generic benchmark source still supports an anchor adapter for the
retired tree workload; that adapter is not the CE execution path in the final
full-normalizer study. Torch LSE and native CE are strong full-vocabulary
controls: avoiding a complete log-softmax output is not by itself a novel
reduction identity. These controls follow established
[PyTorch reduction APIs](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html).

Every full-normalizer arm still computes the same dense LM-head projection
and vocabulary logits in the same chunks. The Triton change concerns selected
logprob normalization and temporary output storage. It does not avoid the
output projection, prune vocabulary support, or establish an accuracy gain.

Kernel metadata errors raise. Invalid target/parent IDs, NaN or positive-
infinity logits, and all-negative-infinity rows produce NaN. A selected
negative-infinity logit in a valid row can mathematically return negative
infinity, but the current scorer deliberately rejects every non-finite score
before ranking. No score is clipped, renormalized over requested tokens, or
silently repaired. Finite output alone is insufficient: reference validity also
checks shape and sequence sums. Fast-backend comparisons additionally require
the declared per-edge and sequence errors and selected-candidate agreement.

## Retained actual-family failures and reference-quality scope

The initial generated-population FP32 gate,
`rank_qwen3b_20261013_v1`, stopped at `item0081/simple_python_279`.
That item's full/cache maximum edge error was **0.0014152527 nats**, and its
full/tree maximum was **0.0013637543 nats**, both above the unchanged **1e-3**
threshold. All 11 same-score control winners agreed on that item. Winner
agreement does not rescue its failed numerical gate. The run, partial items,
failure and elapsed cost are retained. This was an interrupted population
attempt: these values are not maxima from a completed comparison of all 119
inputs, and the successful initialization fixtures do not establish population
validity.

Before any answer access or timing, the prospective reference-quality amendment
in [PROTOCOL.md](../PROTOCOL.md) defines new
`rank_*_20261013_reference_v2` runs for **both checkpoints**. Every task-quality
likelihood aggregator uses **full-prompt FP32 reference scores on all 119
inputs**. Candidate pools, primary seeds, complete permutation families,
development/test assignments and scoring tolerances stay fixed. No additional
samples or failure-based item exclusions are permitted. Sampling controls
continue to use their original frozen samples. Full-reference provenance,
shape, finite-value and sequence-sum checks must pass before reference
predictions are frozen and labels become available.

Every usable actual family still receives full/cache/tree diagnostic
comparisons, with exact arrays, per-edge logprobs, sequence sums, family
membership, errors, selected calls and winner margins retained. The reference
prediction source is explicitly `full`; reference validity and
`fast_path_valid` are separate fields with separate meanings. Failed cache/tree
comparisons remain false and are reported even when the independent full
reference output is valid. Near-tie winner disagreement also fails the fast
gate when its numerical error is small. No globally exact packed implementation
or inexpensive online certificate follows from the offline reference audit.
Completion and quality of these new reference runs remain separate from the
already observed backend failure.

The `1e-3`-nat edge threshold is an empirical gate, not a bound on error to the
ideal real-valued model. A sequence of length `T` can accumulate up to `T` times
an edge error; family winners need a separate check. Reference agreement does
not establish correctness against gold labels. Conversely, a full-reference
quality result cannot establish the failed fast backend's numerical validity
or runtime benefit.

The original fixed-development Qwen2.5-3B scorer benchmark also failed its
unchanged edge gate **before timing**: `simple_python_279` had a maximum error
of **0.0015125275 nats**. The original RQ2 cache/tree performance study is
retired, with its fixtures, source, raw gate results and all cost retained.
No cache-versus-tree throughput or scoring speed claim is accepted. The failed
fixture is not removed, and this failure is not repaired by changing precision,
tolerance or family membership.

## Matched full-backbone normalization study

The separately declared RQ3 normalization mechanism uses new normalizer run
IDs, with the same candidates and input-only development fixture selection.
[benchmark_scoring.py](../scripts/benchmark_scoring.py) prospectively freezes
the first **up to four** lexicographic development IDs with at least two
distinct valid calls from the four primary seeds. It uses all qualifying IDs
if fewer than four exist and requires at least one. It never promotes test IDs
or additional-seed candidates. Eligibility, source schema order, complete
orbits, fixture IDs, and hashes are retained before timed observations. This
small fixed development pool is a systems screen rather than a representative
population throughput estimate.

In `--normalizer-only` mode, all four complete-scorer arms use **the same full
backbone**, FP32/eager, TF32 disabled, the same checkpoint, head chunk 32 and a
whole-family batch. The arms differ only in the selected-token normalization:
Torch log-softmax, Torch LSE, direct native CE, or Triton. Neither cache nor
tree scoring is part of this accepted timing comparison.

The five predeclared pairs are Torch/LSE, Torch/CE, Torch/Triton, LSE/Triton and
CE/Triton. Each receives four randomly oriented ABBA blocks, with the frozen
random seed and shuffled block order. All arms must pass shape, finite-value,
EOS-inclusive sequence-sum, `<1e-3`-nat token, top-sequence and all 11 semantic
winner gates before warming and timing; measured-window outputs are checked
again. Failed validation invalidates the benchmark. Four paired blocks remain
a small descriptive screen, not a population speed guarantee or a strong
confidence claim. All raw windows and all direct pairs are retained.

Complete scorer wall time includes reparsing primary sample text, orbit
enumeration, prompt rendering/tokenization, packing, H2D transfers, model work,
result extraction, and ranking. Model loading and candidate generation are
outside that timed body, as are artifact I/O and validation. Here packing means
the full arm's ordinary input batching; it does not introduce a token tree.
Peak allocated memory and its starting baseline are reported together;
allocator-reserved memory is separate.

The complete scorer's CUDA-event bracket includes host-induced idle gaps;
it is not a profile of active GPU compute. The separate reducer benchmark
captures the **first actual full-backbone LM-head chunk** of the first frozen
fixture and preserves its logits, targets and identity row IDs. Each row has
one requested target; parent rows are not shared or deduplicated. It compares
the same four full-vocabulary reducers in the same five pairs and four ABBA
blocks after shape, finite-value and `<3e-5`-nat output checks. A legacy artifact
label such as `torch_ce_anchor` does not imply use of an anchor adapter in this
full mode: the recorded `ce_adapter` identifies direct native CE.

Model projection and H2D fixture loading are outside the reducer loop;
allocations and launches are inside. Repeated launch bodies have no host reads
or synchronizations. The actual logits fixture is not synthetic, but its timing
scope is only that captured shape. Neither isolated reducer timing nor complete
scorer timing measures candidate-generation speed or HTTP serving throughput.

## Independent full-population normalization audit

[audit_full_triton.py](../scripts/audit_full_triton.py) compares fresh full-path
Triton scores against the retained **full Torch-LSE** vectors on all **119 inputs
per checkpoint**, including zero-family dispositions. It requires the same
scoring source hashes, verifies the pinned model/tokenizer files against the
public revision receipt, and uses stored prompt/suffix token arrays. Re-encoding
is an integrity check and never substitutes new arrays into the scored arm.
Every scored path must retain its complete finite family and exactly one final
EOS with no interior configured stop token.

This audit uses **batch 8 and head chunk 32**, matching the frozen reference
ranking, rather than the timing screen's whole-family batch. FP32/eager and
TF32-disabled execution stay fixed. It checks shape and finite values,
EOS-inclusive lengths and sequence sums, **3e-5 nats per token**, length-scaled
sequence-error bounds and every one of the 11 aggregators' winners. Raw score
vectors, all winner margins and failed item IDs are retained. Zero-family null
predictions are counted separately and are not numerical comparisons.

Numerical failure can be recorded while auditing the remaining population, but
any failed item makes the completed audit invalid and exits with status 1.
An execution/integrity failure or its 1200-second budget stops the audit with
partial evidence. Passing this empirical audit would establish agreement only
in its declared checkpoint/input/runtime scope; it would not prove bitwise
identity, kernel-only error attribution, speed or cheap online verification.
Both quality studies continue to use their frozen full-reference predictions,
independently of this diagnostic audit's outcome.

## Runtime budgets and immutable attempts

The driver checks Spark identity and whole-host compute occupancy, creates a
fresh non-overwriting run, records source/dependency/model receipts, and retains
partial artifacts on failure. The first Qwen2.5-3B full-normalizer run reached
the default **600-second** ceiling after **78 of 80 numerically validated
scoring windows**. It did not complete the planned scoring blocks or the
independent reducer stage. Its raw windows, failure and elapsed cost are
retained; no final speed ratios are accepted from that incomplete attempt.

The explicit infrastructure amendment defines fresh
`normalizer_*_20261013_budget_v2` directories with `--max-seconds 1200`.
The same selected fixtures (four for 3B, three for 1.5B), four arms, five pairs, four ABBA blocks, repeat
counts, warmups, random seed, candidate/input hashes, precision and numerical
tolerances remain unchanged. `--max-seconds` records the declared runtime
ceiling in each run's manifest; its default remains 600. The new attempt does
not select favorable ratios from the old partial run, overwrite its evidence,
repair a numerical failure, or add candidate samples. All failed-attempt and
retry costs count cumulatively toward the original project allocation. An
unfinished retry still supplies no completed benchmark claim.

## Completed artifact disposition

Both budget-v2 normalizer studies completed all 80 scorer and 80 reducer windows
with their original numerical gates. Both all-119 FULL/Triton audits passed:
3B scored 119 nonempty families with maximum token error 9.5367432e-7 nats;
1.5B scored 103 families and retained 16 zero-family items, maximum 1.9073486e-6.
Every one of the 11 aggregators agreed on every scored item. The same 3e-5 audit
threshold was unchanged. These audits do not validate the failed cache/tree arm.

Complete scorer ratios are near one. The isolated resident reducer benefits
must retain their actual four-block ranges and strong native-CE baseline.
Finite-family mass ties the development-selected same-score control on both
held-out populations. See the generated [complete report](../evidence/RESULTS.md)
and [machine-readable evidence](../evidence/summary.json) for all controls,
numerical failures, zero/unsupported dispositions and cost limits.

## Table certificates remain separate

`certificates.py` accepts a supplied finite conditional-probability table and
uses exact rational arithmetic to bound each class's finite-family mass. Its
lower/upper winner certificate is valid for that table. A GPU float converted
to a rational denotes its stored value, not the underlying ideal model
probability. The implementation does not query hidden model probabilities or
bound floating-point error against the real-valued model.

All table probabilities have already been supplied before the audit. Fewer
certificate expansions therefore do not establish fewer model forwards,
less GPU work, or online acceleration. Successful table enumeration controls,
floating score agreement, task accuracy, score-stage runtime, and complete
generation-plus-ranking cost must be reported as separate evidence.
