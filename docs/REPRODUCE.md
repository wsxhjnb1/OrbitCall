# Historical FP32 study reproduction

This document records the historical FP32 study and its original experiment
commands. Use [CUDA/vLLM reproduction](VLLM_REPRODUCE.md) for the released native
BF16 adapter and portable Agent demo.
The completed, reviewed results are in [RESULTS.md](../evidence/RESULTS.md). The frozen
[protocol](../PROTOCOL.md) controls precision, samples, splits, comparisons, and
the four-GPU-hour initial compute budget. Retain failed runs and amendments.

The current design uses full FP32 reference scores for all task-quality
predictions. Cache/tree tolerance failures remain diagnostics; no packed-path
speed result is accepted. The separate systems study compares normalization
backends on an unchanged full backbone. Both models completed the study and
separate post-freeze evaluation; every protocol amendment preceded label access.

All commands below that import a model, execute tests, form the input split,
compute scores, benchmark, or evaluate predictions run on the authorized
**DGX Spark GB10**. Public download and byte-hash verification commands may run
on a staging computer. Run commands from the repository root; every path shown
is repository-relative. Choose fresh run directories rather than overwriting
completed evidence. Command blocks use Bash syntax.

## 1. Provision and record the Spark environment

Use an existing Linux aarch64 environment with a GB10-compatible CUDA PyTorch
build. The study's recorded versions are:

| Component | Recorded version |
| --- | --- |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| Transformers | 5.15.0 |
| Triton | 3.7.1 |
| NumPy | 2.3.5 |
| pytest | 9.1.1 |
| SciPy | 1.18.0 |

Package extras declare the recorded public versions; the PyTorch CUDA local
version and platform must also match the actual Spark environment. Check the
installed stack before adding dependencies. Do not blindly replace it through
an unqualified GPU-package installation.

~~~bash
set -euo pipefail
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
nvidia-smi
~~~

Confirm whole-host availability and the GPU identity before starting work.
The inference/ranking guards check availability and ownership, but they do not
authorize interrupting unrelated processes. Save environment and source
provenance with the new run. `metadata/runtime.json` is the initial study
record; `scripts/runtime_inventory.py --analysis --output PATH` exclusively
creates a new inventory (default `metadata/runtime_current.json`). Supply a
fresh path when an inventory already exists. Runtime or source changes require
a new run identity and fresh validity checks.

This is an **original-host study replay**, not an unrestricted hardware runner.
The original GPU UUID is `GPU-2436ad33-655e-6ff1-a4ba-96a159b20bc4`.
The audit and analyzer require that live UUID and matching saved runtime
metadata. The benchmark exposes `--expected-gpu-uuid`, but changing that flag
alone does not adapt the audit or analyzer. A study on another authorized host
requires a prospective hardware/protocol amendment, new source/runtime hashes,
fresh run IDs and new validity checks. Preserve the original guards, source
snapshots and results; do not relabel a new-host run as this study's evidence.

The base editable install covers standard-library modules. The `runtime`
extra declares Torch, Transformers, and NumPy; `cuda` declares Triton;
`test` declares pytest; `analysis` pins SciPy 1.18.0 for evaluation and
Matplotlib 3.10.3 for report figures; `assets` supplies the Hugging Face
download client. Record actual plot-library installation before generating
figures. Record the Hugging Face Hub version as well because its declared
range is not an exact environment lock.
Extras describe dependencies, not proof that a generic wheel is suitable for
this hardware.

## 2. Download public assets at fixed revisions

Install/use the Hugging Face `hf` CLI for public checkpoint downloads. No
credential is required by these commands for publicly accessible checkpoints.
Keep any locally required authentication outside commands and artifacts.

~~~bash
hf download Qwen/Qwen2.5-3B-Instruct \
  --revision aa8e72537993ba99e69dfaafa59ed015b17504d1 \
  --local-dir models/Qwen2.5-3B-Instruct
hf download Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --local-dir models/Qwen2.5-1.5B-Instruct

python scripts/verify_public_model.py metadata/public_models.json \
  Qwen/Qwen2.5-3B-Instruct models/Qwen2.5-3B-Instruct \
  --output metadata/model_3b_reproduction_verified.json
python scripts/verify_public_model.py metadata/public_models.json \
  Qwen/Qwen2.5-1.5B-Instruct models/Qwen2.5-1.5B-Instruct \
  --output metadata/model_1p5b_reproduction_verified.json

python scripts/stage_bfcl.py --root . download
~~~

The model verifier checks LFS SHA256 or the pinned Git blob identity for every
manifest file and records actual SHA256 values. Keep the checkpoint directory
names above because generation/ranking manifests also compare those names.
Do not regenerate `metadata/public_models.json` from current Hub HEAD:
`record_public_models.py` is a metadata-freezing utility for a new study, not
a way to reproduce the existing revisions.

The release also retains the canonical `metadata/model_3b_verified.json` and
`metadata/model_1p5b_verified.json` receipts. The audit and analyzer require
these exact filenames, and existing audit manifests bind their byte hashes.
Do not replace them with the separately named reproduction receipts above.
The audit independently rehashes the local model files against the canonical
receipts and pinned public manifest. A new checkpoint revision is a new study,
not a repair of an existing receipt.

BFCL staging downloads only two pinned input files and their license. It does
not access answers or execute eligibility selection. The downloaded `.json`
input files contain JSONL. The immutable receipt is
`metadata/bfcl_inputs_download.json`. Existing recognized input files are reused,
and an existing receipt is accepted only when the newly derived bytes match
exactly. Changed assets or receipt contents fail closed rather than being
silently overwritten. Retain any partial transfer for diagnosis.

Model licenses differ. The pinned
[Qwen2.5-3B-Instruct license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/LICENSE)
is the Qwen Research License Agreement with noncommercial-use terms; the pinned
[Qwen2.5-1.5B-Instruct license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE)
is Apache-2.0. Read the actual upstream terms before use. OrbitCall's MIT license
covers its original code and documentation and does not relicense either
checkpoint or BFCL materials. Keep model weights under the local `models/`
directory, outside the public release.

## 3. Freeze inputs on Spark before labels

Create the freeze on Spark, or reproduce it against the byte-identical supplied
freeze. An existing different artifact is rejected:

~~~bash
python scripts/stage_bfcl.py --root . freeze --output data/frozen_inputs.json
~~~

The later commands use this unchanged freeze. If reproduction fails, stop and
investigate before generation. The recorded input-only freeze has 119 eligible
records, 20 development IDs and 99 test IDs; its SHA256 is
`1e8edf278caa6066c1b4c1fab07599cc8acb03fdf0037d3447603ff75916060c`.
These are frozen population counts, not accuracy results. Eligibility is
restricted to one function and two/three required flat scalar arguments.
Schema groups stay intact across the split. Keep every eligible question,
including later parse failures and empty candidate pools.

## 4. Run and retain correctness gates

Run tests and real-checkpoint gates on Spark, retaining console logs and JSON
outputs. The entire test suite, including CPU/table tests, belongs to the Spark
execution record.

~~~bash
mkdir -p runs/gates_fp32_v1
python -m pytest -q tests > runs/gates_fp32_v1/pytest.log 2>&1

python scripts/check_scoring.py \
  --model models/Qwen2.5-3B-Instruct --dtype float32 --attention eager \
  --output runs/gates_fp32_v1/check_3b.json
python scripts/check_scoring.py \
  --model models/Qwen2.5-1.5B-Instruct --dtype float32 --attention eager \
  --output runs/gates_fp32_v1/check_1p5b.json
~~~

Use a shell that stops on nonzero exit (for example, `set -euo pipefail` in
Bash), or check each exit explicitly. A failed gate cannot authorize timing
for that backend. Preserve the failure and follow only a prospective protocol
amendment; never enlarge a threshold or discard its workload to obtain a pass.
Retain the randomized tiny-model and actual-checkpoint errors, sequence scores,
winner margins and winner agreement. The real-fixture token tolerance is
1e-3 nats; the tiny FP32 tolerance is 3e-5. These tolerances are acceptance
criteria, not measurements.

Keep BF16 scoring failures separate from the FP32 repair. Candidate generation
continues to use BF16/SDPA; every full/cached/tree likelihood comparison uses
FP32/eager with TF32 disabled. Kernel validity includes odd/extreme shapes and
the complete vocabulary denominator. A passing fixture does not certify every
architecture or every workload.

The original all-item FP32 ranking and packed development benchmark subsequently
failed the unchanged 1e-3-nat cache/tree gate. The benchmark failed before any
timing. Their failed directories remain immutable. The reference-quality
amendment below uses full FP32 predictions for every item and keeps cache/tree
errors as approximate-backend diagnostics. It does not repair or certify those
paths. The separate normalizer study retains the original development fixtures.

## 5. Generate both candidate pools without answers

Run the two models sequentially on the available GPU:

~~~bash
python scripts/generate_candidates.py \
  --model models/Qwen2.5-3B-Instruct --inputs data/frozen_inputs.json \
  --run runs/orbitcall_qwen3b_20261013_v1 --batch-size 8
python scripts/generate_candidates.py \
  --model models/Qwen2.5-1.5B-Instruct --inputs data/frozen_inputs.json \
  --run runs/orbitcall_qwen1p5b_20261013_v1 --batch-size 8
~~~

Each run records `generation_manifest.json`, immutable batch files under
`generation/`, and `candidates.json`. Primary seeds are 17, 29, 43, and 71;
additional-sampling seeds are 101, 113, 127, and 139. Batch seeds are
`base_seed + 1000003 * batch_index`. Direct samples use temperature 0.7,
top-p 0.9, and at most 128 generated tokens. Greedy is a separate control.
Freeform and reformat stages each allow 64 tokens for each primary seed.
Record actual token/time cost; matching maximum-token caps does not establish
equal runtime.

Use `--resume` only for the same interrupted run with unchanged source,
inputs, model and settings. Do not reuse a completed directory for another
trial. Retain raw text, token IDs and invalid outputs.

## 6. Score and freeze predictions for both models

~~~bash
python scripts/rank_candidates.py \
  --model models/Qwen2.5-3B-Instruct --generation runs/orbitcall_qwen3b_20261013_v1 \
  --run runs/rank_qwen3b_20261013_reference_v2 --inputs data/frozen_inputs.json \
  --reducer torch_lse --batch-size 8 --head-chunk 32
python scripts/rank_candidates.py \
  --model models/Qwen2.5-1.5B-Instruct --generation runs/orbitcall_qwen1p5b_20261013_v1 \
  --run runs/rank_qwen1p5b_20261013_reference_v2 --inputs data/frozen_inputs.json \
  --reducer torch_lse --batch-size 8 --head-chunk 32
~~~

Each rank run writes `ranking_manifest.json`, per-item `items/*.json`,
`predictions.json`, and `pre_gold_freeze.json`. It evaluates full, cached,
and tree paths against the same complete finite families and freezes every
fixed-order prediction from **full FP32/Torch-LSE scores only**. The manifest's
`selected_method` and predictions' `selected_scoring_method` must both be
`full`. Cache/tree edge errors and aggregator-winner agreement remain recorded
for every nonempty family; failed comparisons retain `valid: false` and the
item's `fast_path_valid: false`. They do not remove the item or change its full
reference prediction. An execution or integrity failure is retained and stops
the run. Do not reuse the original failed ranking's directory or relabel it
as reference-v2 evidence.

Retain independent copies of **both** pre-gold receipts before obtaining
answers. Receipts created after seeing labels do not establish isolation.
Ranking-stage elapsed times are descriptive execution costs; they are not a
replacement for the separately randomized timing experiment.

## 7. Benchmark normalizers on the same full reference backbone

Run the answer-free benchmark entry point with each model's frozen candidates:

~~~bash
python scripts/benchmark_scoring.py \
  --model models/Qwen2.5-3B-Instruct \
  --candidates runs/orbitcall_qwen3b_20261013_v1/candidates.json \
  --run runs/normalizer_qwen3b_20261013_budget_v2 \
  --normalizer-only --max-seconds 1200
python scripts/benchmark_scoring.py \
  --model models/Qwen2.5-1.5B-Instruct \
  --candidates runs/orbitcall_qwen1p5b_20261013_v1/candidates.json \
  --run runs/normalizer_qwen1p5b_20261013_budget_v2 \
  --normalizer-only --max-seconds 1200
~~~

The benchmark writes `fixtures.json`, `manifest.json`, raw and validated scorer
windows, timing plans, `scorer_summary.json`, the captured
`real_reducer_fixture.pt` and its metadata, raw and validated reducer windows,
and `reducer_summary.json`. Only a successful `completed.json` with
`kind: full_reference_normalizer_study` marks an accepted completed study.
`failure.json` and partial artifacts must be retained if a validity or
time-budget check fails. No timing result is implied by these commands.

Freeze benchmark inputs using development inputs/candidates without answers.
Use all selected development fixtures, warmed methods, the fixed ABBA seed and
four paired blocks. Every scorer arm uses the same independent full FP32
backbone; only Torch log-softmax, centered LSE, direct native CE or Triton
requested-token normalization changes. Report all five declared direct pairs:
Torch/LSE, Torch/CE, Torch/Triton, LSE/Triton and CE/Triton. Five pairs times
four blocks times four ABBA windows produce 80 scorer windows; the resident
reducer stage has its own complete 80-window schedule. Its actual full-logit
fixture has one target per unique row, and native CE runs directly.

The first 3B normalizer-v1 attempt reached the 600-second ceiling after 78 of
80 validated scorer windows and before the reducer stage. Retain the incomplete
attempt and accept no final ratio from it. The budget-v2 command is an
infrastructure retry under a prospective 1200-second ceiling: all fixtures,
arms, pairs, blocks, repeats, warmups, random seed and numerical tolerances stay
unchanged. Run the whole schedule in a new directory; do not finish two missing
windows in the old run, combine attempts, or select favorable partial ratios.
The larger per-run ceiling is charged cumulatively within the initial project
allocation; it does not reset the project budget.

Keep resident-reducer CUDA intervals separate from synchronized wall time for
the complete full scorer, including parsing/packing, transfers and extraction.
Its CUDA bracket also includes host-induced gaps. Neither measure establishes
cache/tree speed, serving throughput or full generation-pipeline speedup.
Generation/ranking component costs and all failed-attempt costs are reported
separately. Retain raw samples, numerical errors, winner agreement and memory.

## 8. Audit full Triton against frozen full-LSE scores

Run the answer-free population audit for each model after its reference freeze:

~~~bash
python scripts/audit_full_triton.py \
  --model models/Qwen2.5-3B-Instruct \
  --ranking runs/rank_qwen3b_20261013_reference_v2 \
  --run runs/audit_full_triton_qwen3b_20261013_v1
python scripts/audit_full_triton.py \
  --model models/Qwen2.5-1.5B-Instruct \
  --ranking runs/rank_qwen1p5b_20261013_reference_v2 \
  --run runs/audit_full_triton_qwen1p5b_20261013_v1
~~~

The audit uses the stored prompt/suffix token arrays, FP32/eager attention,
TF32 disabled, batch size 8 and head chunk 32. It compares a fresh full-Triton
pass with retained full-LSE token vectors at the unchanged 3e-5-nat per-token
threshold, checks sequence bounds and every aggregator's winner, and retains
all 119 inputs. Zero-family items remain visible but do not count as numerical
comparisons. Numerical failures are retained while the remaining population is
audited; any failed item makes the final audit invalid. Execution or integrity
failures stop the audit and preserve partial artifacts.

Retain `manifest.json`, raw item shards, `summary.json`, and `completed.json` or
`failure.json`, plus source/reference/checkpoint hashes and audit cost. The
script has a 1200-second audit ceiling. This is empirical offline agreement
evidence for the declared population, not kernel-only error attribution, a
speed measurement, an online guard or a universal correctness certificate.

## 9. Fetch labels only after both freezes, then evaluate on Spark

The required order is:

1. Complete generation and ranking for both models.
2. Independently retain both candidate/prediction freeze receipts and their hashes.
3. Complete and retain both full-normalizer studies and population audits;
   freeze their dispositions without changing predictions or workloads.
4. Download only the pinned answer assets.
5. Evaluate on Spark, selecting fixed-order policy with development labels only.

~~~bash
python scripts/fetch_answers.py --root . \
  --freeze runs/rank_qwen3b_20261013_reference_v2 runs/rank_qwen1p5b_20261013_reference_v2

python scripts/evaluate_predictions.py \
  --candidates runs/orbitcall_qwen3b_20261013_v1/candidates.json \
  --predictions runs/rank_qwen3b_20261013_reference_v2/predictions.json \
  --receipt runs/rank_qwen3b_20261013_reference_v2/pre_gold_freeze.json \
  --inputs data/frozen_inputs.json \
  --gold data/answers/BFCL_v4_simple_python.json data/answers/BFCL_v4_live_simple.json \
  --output runs/rank_qwen3b_20261013_reference_v2/evaluation.json
python scripts/evaluate_predictions.py \
  --candidates runs/orbitcall_qwen1p5b_20261013_v1/candidates.json \
  --predictions runs/rank_qwen1p5b_20261013_reference_v2/predictions.json \
  --receipt runs/rank_qwen1p5b_20261013_reference_v2/pre_gold_freeze.json \
  --inputs data/frozen_inputs.json \
  --gold data/answers/BFCL_v4_simple_python.json data/answers/BFCL_v4_live_simple.json \
  --output runs/rank_qwen1p5b_20261013_reference_v2/evaluation.json
~~~

The fetch command requires two distinct freezes, the two declared model names,
and an identical input-population hash before downloading. It verifies the
candidate/prediction receipts and writes `metadata/bfcl_gold_download.json`.
The evaluator verifies candidate,
prediction, input, manifest and item-shard hashes before opening gold.

Evaluation output is a single exclusive `evaluation.json` per model
with per-item correctness, unsupported-gold accounting, candidate coverage,
control accuracy, development-selected fixed policy, and paired question/schema
uncertainty. This is a restricted scalar AST/value metric, not full BFCL.
No-candidate items remain failures; unsupported gold is reported explicitly.
Gold may not change candidates, samples, orders or the input split.

## 10. Analyze completed runs on Spark

The command below identifies the original study's exact run directories. Run
it only after both `evaluation.json` files and both completed, valid audits and
normalizer benchmarks exist. Each two-model argument uses the same order:
Qwen2.5-3B-Instruct, then Qwen2.5-1.5B-Instruct. The analyzer checks those model,
input, prediction, raw-shard and source-hash relationships before writing new
outputs. It reads frozen reports; it does not access answer files or run a model.

~~~bash
python scripts/analyze_study.py \
  --generations \
    runs/orbitcall_qwen3b_20261013_v1 \
    runs/orbitcall_qwen1p5b_20261013_v1 \
  --rankings \
    runs/rank_qwen3b_20261013_reference_v2 \
    runs/rank_qwen1p5b_20261013_reference_v2 \
  --benchmarks \
    runs/normalizer_qwen3b_20261013_budget_v2 \
    runs/normalizer_qwen1p5b_20261013_budget_v2 \
  --audits \
    runs/audit_full_triton_qwen3b_20261013_v1 \
    runs/audit_full_triton_qwen1p5b_20261013_v1 \
  --failed-rankings runs/rank_qwen3b_20261013_v1 \
  --failed-benchmarks \
    runs/benchmark_qwen3b_20261013_v1 \
    runs/normalizer_qwen3b_20261013_v1 \
  --output evidence
~~~

The output is `evidence/summary.json`, `evidence/RESULTS.md` and PNG/PDF pairs
named `test_accuracy`, `scorer_block_ratios` and `reducer_block_ratios`.
`--audits` is required. Failed-attempt flags retain actual failures and costs;
they do not supply accepted quality or speed ratios. Original archives must
include these historical directories. In a new study, use its own fresh run
IDs consistently, list every actual retained attempt, and do not manufacture
historical failures merely to match this command. Missing history is an
explicit cost-accounting limitation, not zero cost.

Output files are exclusive. A failed analysis leaves partial artifacts; use a
fresh output directory after correcting the documented cause. Do not overwrite
an earlier report or select a more favorable result. Quality and performance
claims must use the reviewed report. The released report establishes a resident
normalizer benefit, no mass-over-strong-control quality gain, and essentially
unchanged complete-scorer latency in this scope.

## 11. Preserve and export the release evidence

Retain exact source revisions, runtime/checkpoint hashes, frozen IDs, all
generation outputs, all sequence/edge scores, prediction receipts, raw timing
windows, evaluator outputs, amendments, failures and the measured compute cost.
Negative outcomes belong in the result table.

Keep original `predictions.json`, pre-gold receipts, ranking/audit manifests,
raw items and retained `source/` snapshots byte-identical. Audit validation
matches the seven shared scoring modules against the reference ranking; current
checkout files cannot substitute for an older snapshot. Preserve all other
source files listed by each manifest as well. The repository's `.gitattributes`
disables automatic newline conversion so cross-platform checkout does not
silently invalidate source or receipt hashes. Do not regenerate original
receipts after labels or change them to match edited files.

Keep the original failed ranking, failed packed benchmark, incomplete normalizer
attempt and fresh budget-v2 runs separate. Charge completed scorer windows for
failed ranking prefixes and complete process spans for supplied failed or
completed benchmark attempts once each; do not add their nested windows again.
Include audit costs separately. Any incomplete measured-component subtotal is
a lower bound on recorded host-stage wall time, not pure GPU-busy time or proof
of complete compliance with the initial four-hour allocation. Missing setup,
unrecorded work and interstage overhead must remain explicit.

Export from the original Spark workspace, after checking the exact public file
set and retained `.partial` files for private paths or credentials:

~~~bash
python scripts/render_report_figures.py --summary evidence/summary.json --output evidence/figures
python scripts/export_evidence.py --root . --output evidence/raw_results_v2.tar.gz
~~~

The first command only improves figure layout from the unchanged numerical
summary. Original analysis figures remain retained. Its separate render receipt
records the summary, renderer and output hashes; it changes no estimate, block,
control or population selection. Rendering runs on Spark as well.

This creates `evidence/raw_results_v2.tar.gz` and `evidence/MANIFEST.json` without
overwriting an existing archive or manifest. Publish them alongside the
analysis JSON, Markdown and figures; the exporter does not put those separate
analysis outputs inside the raw archive. The archive contains run JSON/JSONL,
retained partials, original source snapshots, actual reducer-logit fixtures,
public metadata, the input freeze and scrubbed logs. Reducer logits are measured
public-model outputs, not checkpoint weights. Model weights and downloaded
answer files are excluded.

An interrupted or rejected export can leave a partial archive. Publish only a
completed archive together with its matching `MANIFEST.json`; preserve a failed
export separately and use a fresh destination for any corrected retry.

Scientific score/prediction/audit JSON stays byte-identical. There is an
explicit exception for **console traceback paths in `failure.json`**, and
console logs are also scrubbed: the exported copy can differ from the original.
For every member, `MANIFEST.json` records `source_sha256` for the retained
original, `sha256` for the published bytes, and `console_path_redaction`.
Therefore a redacted failure file's published hash need not equal the original
hash in `summary.json`; verify that original hash against `source_sha256`, then
verify the published file against `sha256`. This mapping documents a deliberate
publication transform, not changed scientific measurements. Original private
files remain untouched and are required for byte-identical archival replay;
an analysis rerun on redacted copies cannot reproduce the original failure-file
hash entries. Retained `.partial` files need separate inspection because the
exporter's JSON/path rules do not automatically cover that suffix. The first
export was rejected because scanning raw JSON escapes falsely classified prose
`Innocent III:\n` as a Windows path. The corrected exporter inspects decoded
JSON strings and preserves every scientific byte; the fresh `v2` archive is
the completed release. The rejected partial remains privately retained, with
[retry provenance](../metadata/export_retry_receipt.json).

Before publication, populate README tables only with audited results and
link the actual evidence. Check the public file set for model weights, answer
files, private paths, credentials and unrelated research artifacts. Keep
third-party license notices. Reproducing this implementation does not establish
upstream acceptance, an accuracy improvement or full-pipeline acceleration.

