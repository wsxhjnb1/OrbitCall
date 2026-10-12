# OrbitCall: finite-family scores and numerical scope

OrbitCall studies a restricted inference-time decision rule and its systems
implementation. It does not claim to introduce semantic marginalization,
prefix-tree scoring, stable logsumexp, or cross-entropy reduction. Task accuracy,
score-stage runtime, and complete pipeline runtime are separate empirical
questions. Mathematical identities below do not establish any measured gain.

## 1. What probability is summed

Let `p` be a fixed prompt token sequence and `a` a candidate tool call. Its
declared family `S(a)` consists of distinct complete token sequences for compact
JSON serializations with the same tool name and scalar argument values. The
study enumerates all argument-key permutations for two or three required flat
scalar arguments and appends one explicit end token. The name/arguments
envelope, JSON scalar serialization policy, and model template are fixed.

For a sequence `s = (s_1, ..., s_T)`, define

```text
P(s | p) = product over t of P_model(s_t | p, s_1, ..., s_(t-1))
log P(s | p) = sum over t of full-vocabulary token logprobs
M(a) = sum over distinct s in S(a) of P(s | p)
log M(a) = logsumexp over s in S(a) of log P(s | p)
```

The sum concerns the declared finite family only. It omits other whitespace,
escaping, tokenization, key orders outside scope, reasoning traces, and all
other ways a model could express the same action. It is not the model's complete
semantic posterior. Token-sequence duplicates must not contribute twice.
Explicit termination makes candidate events complete and prefix-free; omitting
it instead measures prefixes with unspecified continuations.

The representative invariance is restricted: reconstructing the same family
from another member's JSON key order leaves the sum unchanged. Reordering tool
descriptions or keys in the input prompt changes `p` and need not preserve the
score or prediction. Missing versus null arguments, scalar types, and array
order remain distinct. Duplicate JSON keys are rejected rather than interpreted
with a last-key-wins parser. The supported finite equivalence relation does not
prove that a tool execution is correct for the user's intent.

When every candidate in an item has family size `K`, `M(a)` and `M(a)/K` have
identical rankings. Arithmetic mean probability is therefore not an independent
control. Mean logprob and length-normalized logprob are different objectives.
Higher model probability, even after aggregation, does not guarantee higher
task accuracy or calibration.

Semantic probability aggregation has substantial prior art, including
[Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in
Natural Language Generation](https://arxiv.org/abs/2302.09664) and
[From Language to Programs: Bridging Reinforcement Learning and Maximum
Marginal Likelihood](https://arxiv.org/abs/1704.07926). These motivate attribution
of the principle; OrbitCall evaluates a small, exactly enumerated serialization
family rather than asserting general semantic-equivalence discovery.

## 2. Packed ancestor attention identity

Build a token trie over the finite suffix sequences. A node corresponds to one
unique token prefix. Its attention keys are exactly the fixed prompt positions
and the suffix nodes on its own ancestor path, including itself. Siblings and
other branches are masked out. At suffix depth `d`, starting at one, the absolute
position is `len(p) + d - 1`; packed array indices are not rotary positions.

For a causal decoder with tokenwise layer normalization and feed-forward
operations, no cross-row operations, and unchanged model parameters, induction
over layers establishes the real-arithmetic identity:

1. Embeddings and position-dependent transformations match independent
   teacher forcing for the same token prefix.
2. At an attention layer, the query and its permitted keys/values match that
   independent sequence. The denominator includes exactly the same ancestors.
3. The attention result, residual update, normalization, and feed-forward result
   consequently match for the node.

Repeated nodes can therefore be computed once and used by several paths.
This is causal prefix reuse, not semantic KV reuse: different token histories
cannot share a hidden state merely because their completed JSON objects agree.

The final prompt position predicts the first suffix token. Each suffix parent
predicts its child token. The end token is scored at its parent; its own hidden
state is unnecessary if it has no children. Output projection may run once per
unique predictive parent and its logits serve several requested child IDs.

Attention masks must describe these edges, and model position IDs must use
depth. Unsupported architectures/masks fail closed. The equality is not a
claim for arbitrary encoder-decoder, multimodal, recurrent, state-space,
cross-token normalization, or sliding-window implementations without separate
verification. Model-specific position offsets and cache conventions still
require actual-checkpoint tests.

Independent and packed GPU executions can differ through reduction order,
GEMM shapes, attention kernels, padding, and floating-point rounding. Real-
arithmetic identity does not imply bitwise equality. Retain per-edge numerical
error, sequence-score error, winner margins, and selected-call agreement. A
near-tie disagreement must not be relabeled as exact equality.

Tokenize with the declared model/template convention and preserve the exact
token arrays used by all methods. In general, tokenizing text fragments
separately is not identical to tokenizing their concatenation. Common-prefix
reuse must be justified at the token-array boundary, not by a string prefix
alone.

[TreeRanker](https://arxiv.org/abs/2508.02455) is relevant prior work for candidate
tries and exhaustive likelihood ranking. Its greedy-path ranking heuristic is
not the exact finite-family objective here; its exhaustive control is relevant.
Use immutable prompt-cache batched teacher forcing as a systems control, rather
than comparing only against repeated full prompts or an inefficient Python loop.

## 3. Requested-token logprob reduction

For predictive parent row `i` with full vocabulary size `V`, and requested token
`j`, the API computes

```text
logprob(i, j) = (logit(i, j) - m_i)
                - log(sum over every v in [0, V) of exp(logit(i, v) - m_i))
m_i = max over every v in [0, V) of logit(i, v)
```

All vocabulary entries participate in the denominator, including unrequested
tokens. Requested target IDs are gathered only after normalization. This is
neither a requested-only softmax nor a sparse-vocabulary approximation.

The optional Triton implementation uses three launches:

1. Each vocabulary tile records its FP32 maximum `m_(i,b)` and shifted
   exponential sum `z_(i,b)`.
2. Each row computes `m_i = max_b m_(i,b)` and
   `z_i = sum_b exp(m_(i,b) - m_i) * z_(i,b)` in FP32. These row statistics are
   computed once even when several requested edges share the parent row.
3. Each requested edge gathers its row/token logit and writes
   `(logit - m_i) - log(z_i)` as FP32.

For `B = ceil(V/4096)`, scratch storage is `2*N*B + 2*N` FP32 values plus `E`
output values. There is no full `[N,V]` log-softmax output and no repeated
`[E,V]` logits tensor. Full vocabulary reads remain necessary. Scratch
allocation and three-launch overhead can dominate small shapes; no speedup is
implied by the memory layout alone.

The Torch controls are full `log_softmax` plus indexing, max-centered
`logsumexp` plus indexing, and negative native cross-entropy for one target per
distinct parent row. Max-centering before subtracting the log sum avoids
cancellation under very large shared offsets. All controls use full vocabulary
normalization. Cross-entropy is established prior art and a required strong
control; removing a full log-softmax buffer is not a new mathematical result.
See the primary [PyTorch logsumexp documentation](https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html),
[PyTorch cross-entropy documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html),
and [Triton reduction documentation](https://triton-lang.org/main/python-api/generated/triton.language.sum.html).

The API accepts FP16/BF16/FP32 logits and performs reductions in FP32. Different
reduction trees and exponential implementations require tolerance checks;
"full-vocabulary" describes support and denominator, not bitwise arithmetic.
Negative-infinity logits in an otherwise valid row have zero probability. A
requested negative-infinity entry can legitimately return negative infinity.
NaN, positive infinity, an all-negative-infinity row, an invalid target, or an
invalid parent row produces NaN. Metadata shape/type errors raise exceptions.
The scorer must reject NaN before scores are used or persisted. The kernel API
does not add host synchronization or a second full-vocabulary validity scan.

## 4. Optional finite-table semantic winner certificate

`certificates.py` receives a complete supplied probability table for the finite
trie; it never calls the model or infers missing probabilities. Probabilities
are converted to exact rational `Fraction` values. A float is interpreted as
its exact stored binary value. Every used conditional row must have nonnegative
entries and total at most one. No clipping, normalization, or numerical
tolerance is silently applied. Decimal strings or rational entries can declare
an exact reference table explicitly. That table must not be confused with raw
GPU scores or the underlying ideal real-valued model.

Let `F` be a frontier of pairwise incomparable unresolved prefixes, `D(a)` the
completed leaves of class `a`, and `q(u)` the product of supplied conditional
probabilities along prefix `u`. Define

```text
L(a) = sum over completed leaves s in D(a) of q(s)
U(a) = L(a) + sum over u in F containing an unresolved class-a leaf of q(u)
```

Because each outgoing conditional sum is at most one, induction over each
finite subtree gives `sum descendant leaf masses <= q(u)`. A prefix is counted
once per semantic class, even if it has several leaves of that class. Different
frontier nodes are disjoint prefix events. Thus the exact finite-family mass
satisfies `L(a) <= M(a) <= U(a)`.

Expanding a frontier prefix into all its finite-trie children increases lower
bounds and cannot increase upper bounds. If some class `a` has
`L(a) > U(b)` for every other class `b`, then `M(a) > M(b)` for every other
class, so it is the exhaustive finite-table winner. Exact ties require
exhaustion; declared class order determines the tie result. An expansion limit
returns no certified winner. The history retains rational bounds and frontier
prefixes for every step.

This is a pure table-level audit. The supplied table has already been computed;
fewer audit expansions are not fewer model forwards or GPU time saved. Loose
bounds, query batching costs, and floating-point differences can prevent any
practical online advantage. No online acceleration claim follows from the
certificate. CPU tests compare every intermediate bound and final winner to
independent full enumeration, including random trees, zero probabilities,
unrepresented vocabulary mass, early stopping, exact ties, and invalid tables.

## 5. Evaluation claims and attribution boundary

The frozen protocol separates model benefit from runtime. Candidate generation
reads only inputs and schemas; answer files belong exclusively to the later
offline evaluator after predictions freeze. JSON validity is not task accuracy.
Candidate coverage is an upper bound on what any reranker of that pool can
recover; finite-family mass cannot repair an absent correct candidate.

[The Format Tax](https://arxiv.org/abs/2604.03616) provides a competing account of
structured-output degradation and motivates decoupled reasoning/formatting
controls. Format effects, candidate sampling, and score aggregation must not be
conflated. Report full costs for additional samples rather than assuming that
equal sample counts, equal score tensors, or equal floating-point operations
mean equal runtime.

Public release should describe implemented behavior and measured outcomes with
hardware/model scope, preserve failures, and cite these prior principles.
Task-quality improvement, score-stage speedup, full-pipeline speedup, numerical
correctness, and an optional probability-table certificate are different claims.
