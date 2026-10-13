# Native BF16 vLLM Agent reproduction

The released mode is **native_shape_suffix**, not the retained failed APC/FP32
development modes. It keeps the full native BF16 projection and crops only
downstream scoring work. Read the [frozen protocol](VLLM_AGENT_SHAPE_PROTOCOL.md)
and [results](../evidence/vllm_agent/RESULTS.md) before interpreting timings.

## Use the adapter

OrbitCall targets compatible CUDA/vLLM environments. The released adapter and
Agent demo do not bind to a Spark GPU UUID. DGX Spark (GB10) was the experimental
platform for the published results; other GPU configurations need their own
compatibility and performance validation.

Provision public vLLM0.29.0 with a CUDA PyTorch build compatible with your GPU.
The measured GB10 environment used Python3.12.3, Torch2.13.0+cu130,
Transformers5.15.0 and Triton3.7.1. Source and native-library identities are
retained with each run. Model downloads and their licenses are described in
[historical reproduction](REPRODUCE.md#2-download-public-assets-at-fixed-revisions).
No weights are distributed here. The3B checkpoint has separate noncommercial
terms; the1.5B checkpoint is Apache-2.0.

```bash
python -m pip install --no-deps --no-build-isolation -e .
PYTHONPATH=src python examples/vllm_agent.py \
  --model models/Qwen2.5-1.5B-Instruct \
  --policy native_shape_suffix --output demo_agent.json
```

The demo runs two development fixtures through actual candidate generation,
strict parsing, both tools, observation feedback and final generation. It
retains all text, token IDs, scores, calls and errors. It does not evaluate
gold answers or represent a speed benchmark. Use `native_default` for stock
BF16 scoring. Outputs must use a fresh file name.

The general-plugin registration is opt-in. The example/study initialization
sets ORBITCALL_VLLM_SCORING=1, VLLM_USE_V2_MODEL_RUNNER=0 and
VLLM_PLUGINS=orbitcall_scoring before constructing the engine. The validated
path uses synchronous legacy-runner scheduling, nonquantized BF16 Qwen2,
ordinary single-group full-attention KV and no LoRA/speculative/KV connector.
Normal vLLM generation, torch.compile and CUDA Graphs remain enabled.

Library callers can use `VLLMScorer.score(prompt_token_ids, validated_calls,
"native_shape_suffix", episode_cache_salt)`. Only complete declared2!/3!
JSON argument-order families plus EOS are accepted by the study. Probabilities
normalize over the full vocabulary. Returned prefix positions are unscored,
not fake probability zeros. A finite-family likelihood is not semantic truth.

## Replay the frozen study on its original host

The frozen benchmark driver/analyzer retain the original Spark GPU UUID as an
experimental evidence guard.
Portable demos do not assert that identity. A new-host experiment needs a
prospective hardware/configuration amendment and fresh source/run records;
do not replace the original receipts to relabel new results.

The recorded input authority is data/agent_inputs_v2.json. Its first preflight
version is retained. Reconstructing tasks uses `scripts/freeze_agent_tasks.py`
on Spark and a fresh output path. A successful new development run must precede
any test inference; the queue freezes its source/configuration selection.

```bash
PYTHONPATH=src python scripts/run_vllm_agent_shape.py \
  --model models/Qwen2.5-3B-Instruct --inputs data/agent_inputs_v2.json \
  --run runs/NEW_DEVELOPMENT_ID --development
PYTHONPATH=src python scripts/queue_vllm_agent_shape.py \
  --development-run NEW_DEVELOPMENT_ID --queue-id NEW_QUEUE_ID
```

The retained release queue uses fixed confirmation IDs; adapt those IDs
prospectively when replaying. Existing directories stop execution and must be
reconciled, never deleted or rerun. The process watchdog is1800s/development
and4200s/model confirmation. Queue children own their fresh engine processes;
no external workloads are terminated.

After both confirmations complete, independent analysis validates all raw
arrays, tokenizer/model/runtime/source receipts, freezes and timing windows
before reference task evaluation. Then render and archive on Spark:

```bash
PYTHONPATH=src python scripts/analyze_vllm_agent_shape.py \
  --runs runs/vllm_agent_shape_confirmation_3b_20261013_v1 \
         runs/vllm_agent_shape_confirmation_1p5b_20261013_v1 \
  --inputs data/agent_inputs_v2.json \
  --selection metadata/NEW_QUEUE_ID_selection.json --output metadata/NEW_ANALYSIS.json
python scripts/render_vllm_agent_report.py \
  --analysis metadata/NEW_ANALYSIS.json --output evidence/NEW_REPORT
```

Scientific JSON bytes are preserved in the raw archive. Only console host
paths are redacted, with original/exported hashes retained. Failed mechanisms,
empty candidate pools, adaptive branches and all four quality controls stay
visible. Whole Agent wall time is primary; scoring-only ratios are separate.
