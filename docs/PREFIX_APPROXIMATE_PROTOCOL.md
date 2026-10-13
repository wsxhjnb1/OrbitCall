# Approximate prefix-sharing complete scorer, 2026-10-13

This separate follow-on study is declared after the approximate linear
development screen: its fastest valid arm took6.593s, whereas native timing
was similar, and native-high exceeded the fixed numerical budget. No held-out
confirmation of that selection is claimed. The original linear/strict studies
remain immutable. The user now permits slight differences; previously failed
cache/tree strict gates remain failed under their earlier thresholds.

The mechanism is different: avoid recomputing a common prompt across the
serialization family. Cached scoring computes the prompt once then batches
independent suffixes; tree scoring also merges identical token prefixes and
uses the existing ancestor-only attention mask. All candidate spellings,
token paths and EOS are still scored. Neither reduces the candidate budget.
TF32x3-all is an optional combination with this structural reuse. Established
prefix caching/trie algorithms are credited, without a novelty claim.

Inherit every identity, answer-isolation,119-population, same error limits,
joint/per-aggregator99% scored-record fidelity, source/hash preservation and
6-block held-out ABBA rule in APPROXIMATE_SCORING_PROTOCOL.md. No tolerance is
relaxed further. The new fixed development grid has four configurations:

1. Native-FP32 cached.
2. Native-FP32 tree.
3. Cached with TF32x3 applied to all ordinary linears.
4. Tree with TF32x3 applied to all ordinary linears.

All use eager attention, head32, and native Torch CE. Tree scoring uses the
existing benchmark.ce_selected_adapter: compute one native full-vocabulary CE
anchor per parent row, then add the requested-logit delta for other edges. The
adapter is installed for every arm; ordinary full/cached rows are passed to
the same native CE implementation. Its parent planning and gathers are part
of timed tree scoring. No Triton reducer advantage is mixed into this study.
Tree retains one FP32 score per requested edge and reconstructs complete paths.

Freeze source/limits/four original3B development fixtures before launch.
One accuracy/setup call and three timed full-family calls per eligible arm;
all four development winners must agree under all11 aggregators. Choose the
fastest valid median, lexical JSON tie break. Budget1800s. This is engineering
discovery after earlier observations, not a wholly unseen method screen.

Confirmation uses that unchanged configuration on both models and the original
pre-frozen eight test fixtures. Audit selected and the strong native-FP32
cached control on all119 stored reference records, batch8 for cached paths;
tree always scores the entire supplied family. Retain processed prompt/query
positions, edges, memory and all error/winner arrays. Cached-control acceptance
is required for its own speed comparison; if it fails, retain the failure and
report the omitted comparator. Alias the control only if selected exactly
equals native cached, with an explicit receipt.

Then run6 randomized ABBA/BAAB blocks for IEEE-full versus selected and, when
distinct/eligible, another6 for native-cached versus selected. Freeze both
potential schedules before any model work. Measure the complete eight-family
score pool, including prefill each call, suffix/ancestor planning and scoring,
head/normalization, transfers and final ranking. No cross-request KV cache
survives a call. Cold loading/setup and warmed repeated calls remain separate.
All8 timing fixtures must meet the same99% rate, which requires8/8 on this
small pool; report this pool separately from the119 audit. Preserve every raw
window before gates. Budget3600s per model. No serving/generation/accuracy or
new-attention-kernel claim follows from score-stage speed.

Do not run the retired linear selection's confirmatory study merely to produce
a redundant tiny gain. Retain its discovery selection as unconfirmed. If this
prefix study fails, preserve its failure without tuning against held-out data.
