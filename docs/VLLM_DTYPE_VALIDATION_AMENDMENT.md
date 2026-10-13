# Validation infrastructure amendment, 2026-10-14

The first attempt, `vllm_dtype_compat_20261014_v1`, stopped during pytest
collection: recursive collection included the retained pre-edit backup under
`tmp/`, producing an import-name collision. No model engine or inference ran.
All original source, logs and failure receipts are preserved.

Retry `vllm_dtype_compat_20261014_v2` collects the official `tests/` directory
explicitly (`python -m pytest -q tests`). Model, two dtype configurations,
fixtures, score thresholds, Agent comparisons and plugin source are unchanged.
This changes infrastructure discovery only; it does not adapt a numerical gate
or performance claim to an observed result.

Before v2 completion, an independent source review identified missing
Agent-round score-array comparisons and persist-before-assert handling for
mixed requests. The owned queue and active engine group were stopped; its
partial files and successful test-stage logs remain. No numerical output was
read to make this change.

Retry `vllm_dtype_compat_20261014_v3` adds an explicit three-argument locked family,
EOS/array-length checks and per-round score/decision comparisons. Mixed raw
output is written before gate assertions. The model, plugin/demo implementation,
runtime, dtype options, development fixtures and numerical thresholds are
unchanged. These prospective coverage corrections are recorded before v3
inference. They do not alter any historical performance evidence.
