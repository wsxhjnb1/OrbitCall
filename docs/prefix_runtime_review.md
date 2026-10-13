# Static review of the approximate prefix scorer

Reviewed on 2026-10-13. This review inspected source, frozen source copies,
protocol text and already recorded discovery completion fields. It did not
run inference, numeric tests, timing, or statistical calculations. Independent
numerical acceptance belongs to the separately retained Spark analysis receipt;
this document does not substitute for that analysis.

## Reviewed identities and disposition

The final [confirmation driver](../scripts/run_approximate_scoring.py) has
SHA256 `967bec71eb429291e5783d4a81257e6e68110bc4cf07d42b8326ffae760ba87a`.
Its retained 3B confirmation source copy matches that hash. The reviewed
[prefix discovery protocol](PREFIX_APPROXIMATE_PROTOCOL.md) has SHA256
`3c67f24f2cde7b28256d9c81f0bec590d8a1189e2846e6f5250e30117c179a87`.
The separate [confirmation control amendment](PREFIX_CONFIRMATION_CONTROL_AMENDMENT.md)
has SHA256 `621516e4775d15906b41c72de6a612533b12eb350e8a3f9d02e685ca3c33c281`;
its retained confirmation copy matches. The original discovery protocol is
preserved, and the additional control is documented in this separate amendment.

No material scoring, aliasing, denominator, or ratio-definition blocker was
found in the final source. The driver's opening documentation now correctly
describes the selected pipeline and its declared strong controls, rather than
claiming that every prefix study audits the native Torch-high arm.

## What the controls identify

The selected configuration is tree scoring with the fixed TF32x3 retrofit on
all supported linears. The final driver constructs three distinct controls:

| Control label | Complete scorer | Linear arithmetic | Comparison scope |
| --- | --- | --- | --- |
| `native` | Independent full-prompt batches | Native IEEE FP32 | Combined prefix reuse plus optional arithmetic |
| `cached` | One prompt prefill, independent suffix batches | Native IEEE FP32 | Gain over an ordinary prompt-cache control |
| `structure` | Same cached/tree structure as selected | Native IEEE FP32 | Arithmetic contribution with structure held fixed |

The same-structure control was added after development selection and before
the prefix audit/held-out confirmation, as declared by the amendment. It does
not change the discovery choice, candidate budget, frozen input fixtures,
numerical limits, or model weights. It strengthens the comparator set without
choosing a preferred baseline after held-out timings.

Aliasing uses equality of the entire configuration dictionary, including
method and arithmetic, not just a shared label. The static cases are:

| Selected configuration | Alias receipts | Distinct comparison labels |
| --- | --- | --- |
| Native cached | `cached` and `structure` point to `optimized` | `native` |
| Native tree | `structure` points to `optimized` | `native`, `cached` |
| Cached + TF32x3 | `structure` points to `cached` | `native`, `cached` |
| Tree + TF32x3 | None | `native`, `cached`, `structure` |

An alias references retained raw/check artifacts and declares zero extra
inference calls. It does not manufacture a duplicate independent measurement
or a speed comparison with the same implementation. Every distinct control
must pass its own whole-population and whole-family gates before its pair is
eligible. The selected/native windows remain retained if a control fails.
Measured-window control failures invalidate that control's pair and are
reported separately from the primary pair's status. The completion object's
primary validity must not be presented as universal acceptance of every pair.

## Full families, EOS, and normalization

`method_for` dispatches the actual cached/tree method, and `validate_layout`
checks the returned method rather than relabeling optimized scores as full.
The original four primary samples are reparsed into the complete compact-JSON
key-permutation family. Prompt IDs, suffix IDs, family ordering, complete token
paths and all eleven aggregators are checked. Every suffix contains the
explicit terminal EOS; interior stop tokens and duplicate token paths are
rejected. A terminal EOS hidden state is not fed back into the backbone because
it predicts no further requested token, but its parent predicts EOS and that
logprob remains in every complete sequence sum.

Cached scoring performs prompt prefill inside each complete score call and
uses a fresh non-sliding DynamicCache wrapper over immutable prefix views for
each suffix batch. Tree scoring includes only predictive parent nodes, uses
semantic-depth positions and ancestor-only attention, then reconstructs every
original complete path. There is no persistent cross-request KV reuse. The
layout checks distinguish independent suffix-token counts from unique tree
edges and predictive parents; fewer repeated computations do not mean fewer
candidate sequences or scored EOS events.

The shared-parent CE adapter remains a full-vocabulary normalizer. It computes
native CE for one real target at every projected parent row and obtains other
requested logprobs by adding their target-logit differences. Every projected
parent has at least one requested outgoing edge by construction. Anchor
planning/gathers are timed. Full/cached distinct-row paths use native CE
directly. No vocabulary pruning, reduced candidate support, Triton-reducer
speed advantage or uncharged repeated-logits expansion is introduced. The
anchor/logit-delta formulation can have floating-point rounding differences;
the declared empirical score bounds apply to its complete returned paths.

## Denominators and timing ratios

The population fidelity denominator is the number of scored, nonempty records,
not all 119 inputs with empty pools counted as automatic successes. Empty
candidate pools remain separate recorded dispositions. Each aggregator and
joint all-eleven agreement use exact integer counts at the 99% threshold.
For an eight-record timing pool, that threshold requires all eight records to
agree; it does not assert a universal requirement of 100% population fidelity.
Pooled mean absolute error is token-weighted over the full EOS-inclusive paths,
not averaged over only winners or selected orders.

Each distinct comparator gets a frozen six-block ABBA/BAAB plan. Within a block,
the ratio is the mean of the two complete control wall times divided by the
mean of the two optimized wall times. The reported median is over these six
paired block ratios. A value above one means the selected scorer is faster.
All distinct eligible comparators are reported, and bootstrap limits are
explicitly descriptive of these six blocks. They do not establish serving
throughput or broad task-distribution generalization.

Timed complete calls include reparsing, orbit construction, tokenization,
packing, prefill/backbone, full-vocabulary head/CE, transfers and final ranking.
Loading, first-call/JIT setup, artifact writes and validation are separate.
Both full and cached controls use the whole serialization-family batch for
timing. The population audit retains its prescribed batch-eight shapes for
full/cached and whole-family tree scoring. These different shapes have their
own numerical gates; they are not silently treated as bitwise equivalent.

## Historical trace and discovery budget finding

The retained full-reference ranking source is verified from its own archived
hashes. Static comparison of the current scorer with the retained reference
copy found only the documented public default change from `method="tree"`
to `method="full"`. The measured paths explicitly specify their methods. The
cached/tree numerical implementation is not rewritten to disguise an earlier
strict failure. The current fixed Tensor Core module also matches its earlier
strict-discovery snapshot hash
`1984b5d967ed481b6fc0c31093e1c494c2f17edcae56b6d4789599680de64b97`.
Old strict failures remain failures, while the user-authorized approximate
study has a separately declared tolerance and acceptance contract.

Two static reviewers identified a discovery budget-enforcement weakness:
`screen_prefix_approximate.py` checks 1800 seconds before each configuration,
but not before every timed repeat or final selection write. The retained
completion fields establish that this weakness did not cause either completed
approximate discovery to exceed its declared budget:

| Retained discovery run | Recorded `elapsed_seconds` | Declared budget |
| --- | ---: | ---: |
| `approximate_discovery_3b_20261013_v1` | 211.47203468182124 | 1800 seconds |
| `prefix_approximate_discovery_3b_20261013_v1` | 50.75266453297809 | 1800 seconds |

These fields are quoted from the existing completion JSON, not recomputed on
Windows. Their SHA256 values are respectively
`bb8fb7baa87c551552fe4c9f1a86d578091be5885312ae433a88024ccf82c298` and
`2c6aee0bc8a1125e2aece0d77d8bd89e494741177cff247aa326bc4c1faad0d7`.
The frozen discovery source was not edited retrospectively. Future runs should
enforce that budget at each repeat and before accepting a selection. The final
confirmation driver checks its 3600-second budget throughout and again before
writing completion, freezes/verifies source copies, checks final identity
hashes and records an integrity check on failure where available.

This static review makes no new inference-speed, numerical-accuracy or
task-quality measurement. Any accepted runtime claim is restricted to complete
scoring on the retained checkpoints, candidate pools, runtime, hardware and
declared approximation budget.
