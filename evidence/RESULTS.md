# OrbitCall completed study results

This report is generated from frozen artifacts on DGX Spark. It reports both models and all controls; no best-model or test-subset selection is used.

Candidate generation uses BF16/SDPA. All likelihood comparisons use FP32/eager with TF32 disabled. Earlier BF16 full/cache disagreement (up to 0.75 nats/token) failed the numerical gate and remains a failure, not a positive result.

The initial all-item FP32 fast-path run also failed the unchanged 1e-3 nats/token gate. A prospective amendment before labels or timing makes full FP32 reference scoring the quality-prediction source for both models. Cached/tree comparisons retain their failures as diagnostics; no all-item exact fast-path or cheap production guard is claimed.

The original cache/tree development benchmark then failed its strict numerical gate before any timing. RQ2 has no accepted cache/tree speed result. A separate RQ3 normalizer study keeps the same complete full FP32 backbone, frozen development fixtures and four paired blocks while comparing Torch log-softmax, LSE, direct native cross-entropy and Triton requested-token normalization.

The score is mass over a restricted 2!/3! JSON argument-order family including one explicit EOS. It is not a complete semantic posterior or a universal accuracy guarantee.

## Task quality

Accuracy below uses the supported scalar BFCL AST/value scope, not the full official BFCL leaderboard. Unsupported labels and zero-candidate questions remain reported in the frozen population.

### Qwen2.5-3B-Instruct

Frozen population 119: 20 development and 99 test; unsupported labels 1. Development-selected order: `fixed_102`.

| Method | Correct / supported test | Accuracy | Question Wilson 95% interval |
| --- | --- | --- | --- |
| Finite-family mass | 91 / 98 | 92.86% | [85.98, 96.5] percentage points |
| Schema order | 91 / 98 | 92.86% | [85.98, 96.5] percentage points |
| Max order | 91 / 98 | 92.86% | [85.98, 96.5] percentage points |
| Mean logprob | 91 / 98 | 92.86% | [85.98, 96.5] percentage points |
| Length-normalized mean | 90 / 98 | 91.84% | [84.71, 95.81] percentage points |
| Dev-selected fixed | 91 / 98 | 92.86% | [85.98, 96.5] percentage points |
| fixed_012 | 91 / 98 | 92.86% | [85.98, 96.5] percentage points |
| fixed_021 | 91 / 98 | 92.86% | [85.98, 96.5] percentage points |
| fixed_102 | 91 / 98 | 92.86% | [85.98, 96.5] percentage points |
| fixed_120 | 91 / 98 | 92.86% | [85.98, 96.5] percentage points |
| fixed_201 | 91 / 98 | 92.86% | [85.98, 96.5] percentage points |
| fixed_210 | 91 / 98 | 92.86% | [85.98, 96.5] percentage points |
| Vote 4 | 91 / 98 | 92.86% | [85.98, 96.5] percentage points |
| Vote 8 | 91 / 98 | 92.86% | [85.98, 96.5] percentage points |
| Native greedy | 91 / 98 | 92.86% | [85.98, 96.5] percentage points |
| Freeform + reformat vote | 89 / 98 | 90.82% | [83.46, 95.09] percentage points |

The development-selected held-out comparator is `dev_selected_fixed`. Mass-minus-control accuracy difference: 0.00%; paired question-bootstrap interval [0, 0] percentage points, schema-bootstrap interval [0, 0] percentage points.
The strongest observed test same-score control is `canonical_schema` (92.86%); this oracle test comparison is descriptive. All five paired equal-score comparisons and within-model Holm p values are preserved in `summary.json`.

Primary candidate coverage: 119/119 with a valid call; 0 zero-candidate items. JSON-and-schema valid samples: 476/476. Oracle-correct primary pool coverage: 111/118 supported items.

### Qwen2.5-1.5B-Instruct

Frozen population 119: 20 development and 99 test; unsupported labels 1. Development-selected order: `fixed_012`.

| Method | Correct / supported test | Accuracy | Question Wilson 95% interval |
| --- | --- | --- | --- |
| Finite-family mass | 77 / 98 | 78.57% | [69.45, 85.54] percentage points |
| Schema order | 77 / 98 | 78.57% | [69.45, 85.54] percentage points |
| Max order | 77 / 98 | 78.57% | [69.45, 85.54] percentage points |
| Mean logprob | 76 / 98 | 77.55% | [68.34, 84.68] percentage points |
| Length-normalized mean | 77 / 98 | 78.57% | [69.45, 85.54] percentage points |
| Dev-selected fixed | 77 / 98 | 78.57% | [69.45, 85.54] percentage points |
| fixed_012 | 77 / 98 | 78.57% | [69.45, 85.54] percentage points |
| fixed_021 | 76 / 98 | 77.55% | [68.34, 84.68] percentage points |
| fixed_102 | 76 / 98 | 77.55% | [68.34, 84.68] percentage points |
| fixed_120 | 76 / 98 | 77.55% | [68.34, 84.68] percentage points |
| fixed_201 | 76 / 98 | 77.55% | [68.34, 84.68] percentage points |
| fixed_210 | 76 / 98 | 77.55% | [68.34, 84.68] percentage points |
| Vote 4 | 73 / 98 | 74.49% | [65.05, 82.08] percentage points |
| Vote 8 | 83 / 98 | 84.69% | [76.27, 90.5] percentage points |
| Native greedy | 53 / 98 | 54.08% | [44.25, 63.61] percentage points |
| Freeform + reformat vote | 53 / 98 | 54.08% | [44.25, 63.61] percentage points |

The development-selected held-out comparator is `canonical_schema`. Mass-minus-control accuracy difference: 0.00%; paired question-bootstrap interval [0, 0] percentage points, schema-bootstrap interval [0, 0] percentage points.
The strongest observed test same-score control is `canonical_schema` (78.57%); this oracle test comparison is descriptive. All five paired equal-score comparisons and within-model Holm p values are preserved in `summary.json`.

Primary candidate coverage: 103/119 with a valid call; 16 zero-candidate items. JSON-and-schema valid samples: 278/476. Oracle-correct primary pool coverage: 92/118 supported items.

## Numerical agreement and descriptive full-pool costs

Null predictions on zero-family items are not counted as actual numerical scoring comparisons. Every nonempty item was compared. The numerical gate remains 1e-3 nats/token; failed fast paths are reported and are not required for accepting the independent full reference output. These one-pass costs are not randomized speedup evidence.

### Qwen2.5-3B-Instruct

Scored nonempty items: 119; retained zero-family items: 0.

| Alternative vs full | Maximum token error (nats) | Maximum sequence error (nats) | Failed family gates | Mass winner agreement |
| --- | --- | --- | --- | --- |
| cached | 0.001415252685546875 | 0.0015499941655434668 | 1/119 | 119/119 (100.00%) |
| tree | 0.0013637542724609375 | 0.0014813286834396422 | 1/119 | 119/119 (100.00%) |

Combined cached/tree gate failures: 1/119 nonempty families. IDs: simple_python_279. Quality selection uses full FP32 reference scores for every item.

| Scorer | All-item measured seconds | Prompt positions | Query positions | Scored edges |
| --- | --- | --- | --- | --- |
| full | 79.895 | 113100 | 14948 | 15372 |
| cached | 45.085 | 24266 | 14948 | 15372 |
| tree | 38.845 | 24266 | 9066 | 9592 |

All aggregator winner counts, raw query/token counts, reducer choice, peak allocations and source/hash receipts are retained in the JSON and raw item evidence.

### Qwen2.5-1.5B-Instruct

Scored nonempty items: 103; retained zero-family items: 16.

| Alternative vs full | Maximum token error (nats) | Maximum sequence error (nats) | Failed family gates | Mass winner agreement |
| --- | --- | --- | --- | --- |
| cached | 0.00016164779663085938 | 0.0001802346669137478 | 0/103 | 103/103 (100.00%) |
| tree | 0.00019025802612304688 | 0.0001953986065927893 | 0/103 | 103/103 (100.00%) |

Combined cached/tree gate failures: 0/103 nonempty families. IDs: none. Quality selection uses full FP32 reference scores for every item.

| Scorer | All-item measured seconds | Prompt positions | Query positions | Scored edges |
| --- | --- | --- | --- | --- |
| full | 52.999 | 105450 | 14204 | 14643 |
| cached | 21.013 | 21070 | 14204 | 14643 |
| tree | 18.009 | 21070 | 8496 | 8998 |

All aggregator winner counts, raw query/token counts, reducer choice, peak allocations and source/hash receipts are retained in the JSON and raw item evidence.

## Whole-population FULL/Triton numerical audit

This separate answer-free audit compares stored FULL/Torch-LSE vectors with fresh FULL/Triton vectors at batch8/chunk32, FP32/eager and TF32 disabled. It uses every frozen item, the same complete EOS-inclusive family and an unchanged3e-5 nats/token threshold. Null families remain in the population and are excluded from numerical agreement counts.

| Model | Complete population (dev/test) | Nonempty / zero families | Maximum token error (nats) | Maximum sequence error (nats) | Mass winner agreement | Audit elapsed seconds |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-3B-Instruct | 119 (20/99) | 119 / 0 | 9.5367432e-07 | 1.6539979e-06 | 119/119 (100.00%) | 125.785 |
| Qwen2.5-1.5B-Instruct | 119 (20/99) | 103 / 16 | 1.9073486e-06 | 2.6113235e-06 | 103/103 (100.00%) | 67.335 |

All eleven aggregator agreement counts, verified public checkpoint receipts, raw-item and retained-source hashes are preserved under each study's `audit` in [summary.json](summary.json). Audit elapsed cost is counted once in the cumulative subtotal. Its alternative scoring windows are descriptive and are not compared as a randomized timing pair against stored reference costs. This empirical full-path audit establishes neither bitwise identity, an online production guard, nor cache/tree correctness.

## Full-reference normalizer randomized ABBA timing

Each pair has **four paired blocks**, with two windows per arm per block. Ratios are A time / B time, so values above one favor B. We enumerate the 256 empirical resamples of those four blocks for a descriptive bootstrap interval; these are not 256 independent trials. The observed block range is also reported. Fixtures are fixed development IDs selected without answers, with no test fallback.

### Qwen2.5-3B-Instruct

Fixtures: simple_python_259, simple_python_279, simple_python_283, simple_python_393; selected 4 from 4 eligible development items.

| Stage / pair (A / B) | Wall median ratio | Wall block range | Descriptive block-bootstrap 95% interval | CUDA interval median ratio |
| --- | --- | --- | --- | --- |
| complete full scorer: FULL Torch / FULL LSE | 1.0016 | [0.9965, 1.007] | [0.9965, 1.007] | 1.0016 |
| complete full scorer: FULL Torch / FULL native CE | 1.0003 | [0.9925, 1.005] | [0.9925, 1.005] | 1.0003 |
| complete full scorer: FULL Torch / FULL Triton | 0.9998 | [0.9986, 1.001] | [0.9986, 1.001] | 0.9998 |
| complete full scorer: FULL LSE / FULL Triton | 1.0043 | [1.002, 1.006] | [1.002, 1.006] | 1.0043 |
| complete full scorer: FULL native CE / FULL Triton | 1.0004 | [0.9982, 1.002] | [0.9982, 1.002] | 1.0004 |
| resident full-logit reducer: Torch / LSE | 0.4694 | [0.4583, 0.5383] | [0.4583, 0.5383] | 0.4691 |
| resident full-logit reducer: Torch / Native CE | 1.2846 | [1.142, 1.298] | [1.142, 1.298] | 1.2851 |
| resident full-logit reducer: Torch / Triton | 1.6206 | [1.52, 2.363] | [1.52, 2.363] | 1.6224 |
| resident full-logit reducer: LSE / Triton | 4.5539 | [3.95, 4.721] | [3.95, 4.721] | 4.5619 |
| resident full-logit reducer: Native CE / Triton | 1.3028 | [1.255, 1.646] | [1.255, 1.646] | 1.3034 |

Complete-full-scorer wall time includes parsing/tokenization, packing, transfers, unchanged full-backbone model execution, extraction and aggregation. Its CUDA bracket includes host-induced gaps; it is not active GPU time. Resident full-logit reducer timing uses one target per unique parent row and excludes LM-head projection, backbone computation and generation. Native CE and LSE are strong direct paired controls. The legacy raw arm name `torch_ce_anchor` invokes direct native CE in this full-only study; plots use its actual operation name. No cache/tree speed, serving-throughput or full generation-pipeline speedup is inferred.

Benchmark process span: 627.612s; summed timed windows: 558.346s; non-timed setup/gates/warmup/validation/I/O/other: 69.266s. Cold model load was not separately measured.

### Qwen2.5-1.5B-Instruct

Fixtures: simple_python_259, simple_python_279, simple_python_284; selected 3 from 3 eligible development items.

| Stage / pair (A / B) | Wall median ratio | Wall block range | Descriptive block-bootstrap 95% interval | CUDA interval median ratio |
| --- | --- | --- | --- | --- |
| complete full scorer: FULL Torch / FULL LSE | 0.9971 | [0.9965, 0.9986] | [0.9965, 0.9986] | 0.9971 |
| complete full scorer: FULL Torch / FULL native CE | 0.9989 | [0.9977, 1] | [0.9977, 1] | 0.9989 |
| complete full scorer: FULL Torch / FULL Triton | 1.0022 | [0.9986, 1.003] | [0.9986, 1.003] | 1.0022 |
| complete full scorer: FULL LSE / FULL Triton | 1.0034 | [1.002, 1.004] | [1.002, 1.004] | 1.0034 |
| complete full scorer: FULL native CE / FULL Triton | 1.0021 | [1.002, 1.002] | [1.002, 1.002] | 1.0021 |
| resident full-logit reducer: Torch / LSE | 0.4951 | [0.4404, 0.5553] | [0.4404, 0.5553] | 0.4948 |
| resident full-logit reducer: Torch / Native CE | 1.1476 | [1.007, 1.224] | [1.007, 1.224] | 1.1478 |
| resident full-logit reducer: Torch / Triton | 1.4530 | [1.419, 2.279] | [1.419, 2.279] | 1.4539 |
| resident full-logit reducer: LSE / Triton | 4.1730 | [2.64, 4.643] | [2.64, 4.643] | 4.1805 |
| resident full-logit reducer: Native CE / Triton | 1.6310 | [1.242, 5.861] | [1.242, 5.861] | 1.6315 |

Complete-full-scorer wall time includes parsing/tokenization, packing, transfers, unchanged full-backbone model execution, extraction and aggregation. Its CUDA bracket includes host-induced gaps; it is not active GPU time. Resident full-logit reducer timing uses one target per unique parent row and excludes LM-head projection, backbone computation and generation. Native CE and LSE are strong direct paired controls. The legacy raw arm name `torch_ce_anchor` invokes direct native CE in this full-only study; plots use its actual operation name. No cache/tree speed, serving-throughput or full generation-pipeline speedup is inferred.

Benchmark process span: 340.341s; summed timed windows: 298.067s; non-timed setup/gates/warmup/validation/I/O/other: 42.275s. Cold model load was not separately measured.

## Generation cost and scope

The fixed generator records synchronized `model.generate` windows. These omit prompt preprocessing/H2D, decoding/parsing, checkpoint setup and disk I/O. First-to-last spans include intervening work but omit initial/final tails. The control costs are actual executed allocations, not randomized generation comparisons; direct4 is a subset of direct8.

### Qwen2.5-3B-Instruct

Retained generation batches: 255 across 17 mode/seed stages.

| Control allocation | Batch windows | Generated stage samples | Measured generate seconds | Actual output tokens incl. EOS | Maximum output-token budget | Logical prompt tokens |
| --- | --- | --- | --- | --- | --- | --- |
| greedy | 15 | 119 | 29.157 | 3314 | 15232 | 24266 |
| direct4 | 60 | 476 | 120.425 | 13297 | 60928 | 97064 |
| direct8 | 120 | 952 | 241.046 | 26625 | 121856 | 194128 |
| freeform_plus_reformat | 120 | 952 | 316.447 | 35699 | 60928 | 216080 |

All-stage measured generate sum: 586.650s. Observed complete generation span between recorded forward windows: 599.026s. Cold checkpoint loading/setup is unmeasured. Summing generation windows and descriptive scoring costs is a measured-component total, not a randomized end-to-end speedup.

### Qwen2.5-1.5B-Instruct

Retained generation batches: 255 across 17 mode/seed stages.

| Control allocation | Batch windows | Generated stage samples | Measured generate seconds | Actual output tokens incl. EOS | Maximum output-token budget | Logical prompt tokens |
| --- | --- | --- | --- | --- | --- | --- |
| greedy | 15 | 119 | 23.039 | 5321 | 15232 | 24266 |
| direct4 | 60 | 476 | 99.303 | 21024 | 60928 | 97064 |
| direct8 | 120 | 952 | 197.975 | 41772 | 121856 | 194128 |
| freeform_plus_reformat | 120 | 952 | 210.899 | 46128 | 60928 | 219154 |

All-stage measured generate sum: 431.914s. Observed complete generation span between recorded forward windows: 444.002s. Cold checkpoint loading/setup is unmeasured. Summing generation windows and descriptive scoring costs is a measured-component total, not a randomized end-to-end speedup.

## Retained failed attempts and cumulative cost

Failed history contributes measured cost only. It supplies no task-quality predictions and does not change the numerical threshold or final candidate pool.

### rank_qwen3b_20261013_v1

Retained checkpoints: 82; successful items before stopping: 81; failed items: 1; unattempted or unretained population: 37. Failed IDs: simple_python_279.

| Method | Completed measurement windows | Recorded wall seconds |
| --- | --- | --- |
| full | 82 | 54.044 |
| cached | 82 | 31.156 |
| tree | 82 | 27.090 |

| Failed item | Alternative | Actual maximum token error (nats) | Actual maximum sequence error (nats) | Winner agreements |
| --- | --- | --- | --- | --- |
| simple_python_279 | cached | 0.0014152527 | 0.0015499942 | 11/11 |
| simple_python_279 | tree | 0.0013637543 | 0.0014813287 | 11/11 |

The attempt stopped at the original 0.001 nats/token numerical/winner gate. Per-item errors, comparator values and winner margins remain in `summary.json` and raw checkpoints. Its measured scorer subtotal is 112.290s; initial loading and unrecorded work are excluded.

### benchmark_qwen3b_20261013_v1

Study kind: `packed_scorer_study`; failure cause: `packed_numerical_gate_failure`; declared runtime allowance: 600s. Failed process interval: 10.983s. Frozen fixtures: simple_python_259, simple_python_279, simple_python_283, simple_python_393. Failed before any timed window: True. Accepted pair ratios: 0; no quality output is produced.
original cache/tree numerical gate failed at the unchanged1e-3 tolerance; preserve the negative and process cost; RQ2 is retired with no cache/tree speed claim

Retained scorer windows: 0; strictly validated: 0; total windows in saved plan: None.
Retained reducer windows: 0; strictly validated: 0; total windows in saved plan: None.
Actual retained maximum token error: 0.0015125274658203125 nats; maximum sequence error: 0.0016826727078296244 nats. The original 1e-3 nats/token threshold remains unchanged.

RQ2 cache/tree speed remains unestablished. Retained incomplete normalizer windows are not selected into accepted timing ratios.

### normalizer_qwen3b_20261013_v1

Study kind: `full_reference_normalizer_study`; failure cause: `runtime_budget_exhausted`; declared runtime allowance: 600s. Failed process interval: 602.445s. Frozen fixtures: simple_python_259, simple_python_279, simple_python_283, simple_python_393. Failed before any timed window: False. Accepted pair ratios: 0; no quality output is produced.
full-reference normalizer study exceeded its declared runtime budget; preserve every partial raw/strict-valid window and process cost; no incomplete-window ratios are accepted; a separately declared retry keeps fixtures, arms, seeds and numerical thresholds unchanged

Retained scorer windows: 78; strictly validated: 78; total windows in saved plan: 80.
Retained reducer windows: 0; strictly validated: 0; total windows in saved plan: None.

RQ2 cache/tree speed remains unestablished. Retained incomplete normalizer windows are not selected into accepted timing ratios.

| Recorded component | Seconds |
| --- | --- |
| Final generation, every executed stage once | 1018.564 |
| Final ranking, all three measured scorer paths | 255.845 |
| Supplied failed ranking attempts | 112.290 |
| Supplied failed benchmark process intervals | 613.428 |
| Final benchmark complete process intervals | 967.953 |
| Final whole-population FULL/Triton audit process intervals | 193.120 |
| **Cumulative measured component lower bound** | **3161.201** |

The measured component subtotal is **0.8781 host/GPU-stage hours, a lower bound**. These are actual host wall intervals on Spark, not pure GPU-busy time. The declared initial budget remains four Spark GPU-hours. An incomplete subtotal cannot establish complete budget compliance. Downloads, initial generation/ranking loads, standalone gates not supplied here, unrecorded work, interstage overhead and CPU evaluation/analysis are not fully accounted; supplied audit and failed/completed benchmark process intervals already include their own setup/gates/warmup/validation/I/O.

## Reproducibility and limits

Runtime: Python 3.12.3, architecture aarch64; package versions and input/report/raw-source hashes are in `summary.json`.

Both public checkpoints and the same frozen population are reported. The small held-out dataset, four timing blocks, restricted scalar evaluator, separate generation/scoring precision and finite serialization family limit generalization. All fixed-order predictions were frozen before labels; only the declared development selection uses development answers. Earlier BF16 failures are retained and must not be represented as passing. No publication, official leaderboard, universal quality, or serving speedup claim follows from this report.
