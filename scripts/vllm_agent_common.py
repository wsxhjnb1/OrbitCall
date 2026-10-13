"""Pinned BF16 vLLM construction and owned-process lifecycle for Spark studies."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
os.environ["ORBITCALL_VLLM_SCORING"] = "1"
os.environ["VLLM_PLUGINS"] = "orbitcall_scoring"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from audit_full_triton import GPU_UUID, verify_model, sha256, write_new
from orbitcall.gpu_guard import gpu_pids, require_idle_gpu

ROOT = Path(__file__).resolve().parents[1]
ENGINE = dict(dtype="bfloat16", kv_cache_dtype="auto", tensor_parallel_size=1,
              gpu_memory_utilization=0.65, max_model_len=4096, max_num_seqs=64,
              max_num_batched_tokens=8192, enable_prefix_caching=True,
              enable_chunked_prefill=True, async_scheduling=False,
              enforce_eager=False, trust_remote_code=False, seed=20261013,
              logprobs_mode="raw_logprobs")
LIMITS = dict(max_token_absolute_nats=0.05, mean_token_absolute_nats=0.005,
              max_sequence_absolute_nats=0.5, minimum_winner_agreement=0.99)


def read_json(path):
    return json.loads(Path(path).read_text())


def initialize(model, run):
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise RuntimeError("All runtime validation runs on authorized Spark")
    identity = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],text=True,timeout=15).strip()
    if GPU_UUID not in identity or "GB10" not in identity:
        raise RuntimeError("Unexpected GPU")
    require_idle_gpu()
    receipt, stats = verify_model(model,read_json)
    versions = {name:importlib.metadata.version(name) for name in ("vllm","torch","transformers","triton")}
    if versions["vllm"] != "0.29.0":
        raise RuntimeError("Pinned vLLM version mismatch")
    from vllm import LLM
    import vllm
    package=Path(vllm.__file__).resolve().parent
    runtime_sources={name:sha256(package/name) for name in
        ("sampling_params.py","v1/worker/gpu_model_runner.py","v1/worker/gpu_input_batch.py",
         "v1/core/kv_cache_manager.py","v1/core/kv_cache_coordinator.py","v1/engine/logprobs.py",
         "v1/sample/sampler.py","model_executor/layers/logits_processor.py")}
    native_binaries={str(path.relative_to(package)):sha256(path)
                     for path in sorted(package.glob("*.so"))}
    begin=time.perf_counter()
    llm=LLM(model=str(model),**ENGINE)
    owned_gpu_check()
    config=llm.llm_engine.vllm_config
    write_new(run/"engine.json",dict(model=receipt, runtime=versions,gpu=identity,
                                    options=ENGINE,initialization_seconds=time.perf_counter()-begin,
                                    runtime_source_sha256=runtime_sources,
                                    native_binary_sha256=native_binaries,
                                    actual_compilation_mode=str(config.compilation_config.mode),
                                    actual_cudagraph_mode=str(config.compilation_config.cudagraph_mode),
                                    plugin_source_sha256=sha256(ROOT/"src/orbitcall/vllm_plugin.py")))
    return llm


def owned_gpu_check():
    import psutil
    allowed={os.getpid(),*(p.pid for p in psutil.Process().children(recursive=True))}
    if gpu_pids()-allowed:
        raise RuntimeError("Unrelated GPU process invalidates this run")


def shutdown(llm):
    # Native EngineCore lifecycle; never terminate a process selected by name.
    llm.llm_engine.engine_core.shutdown()


def parse_candidates(outputs,schemas):
    from orbitcall.serialization import parse_call, semantic_key
    calls,errors={},[]
    for sample in outputs:
        parsed=None
        for schema in schemas:
            try:
                parsed=parse_call(sample["text"],schema)
                break
            except ValueError:
                pass
        if parsed is None:
            errors.append({"seed":sample["seed"],"reason":"invalid strict tool call"})
        else:
            calls[semantic_key(parsed)]=parsed
    return [calls[key] for key in sorted(calls)],errors
