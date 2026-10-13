"""CPU-only dtype routing contracts; execute on Spark, not Windows.

The fake engine checks argument resolution, immutable study defaults and demo
receipts. It loads no checkpoint and creates no model or CUDA context.
"""

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


def load_demo(monkeypatch, resolved):
    defaults = {"dtype": "bfloat16", "seed": 20261013}
    state = SimpleNamespace(defaults=defaults, created=[], pools=[], shutdown=[])

    class FakeLLM:
        def __init__(self, **options):
            state.created.append(options)
            self.llm_engine = SimpleNamespace(vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(dtype=resolved)))

    class FakeScorer:
        def __init__(self, llm):
            self.llm = llm

    def run_pool(scorer, inputs, policy, key):
        state.pools.append((scorer.llm, inputs, policy, key))
        return {"records": [{"id": row["id"]} for row in inputs],
                "policy": policy, "elapsed_seconds": 0.01}

    def install(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    install("vllm", LLM=FakeLLM)
    install("run_vllm_agent_shape", run_pool=run_pool)
    install("vllm_agent_common", ENGINE=defaults, shutdown=state.shutdown.append)
    install("orbitcall.vllm_scoring", SHAPE_ARMS=("native_default", "native_shape_suffix"),
            VLLMScorer=FakeScorer)
    install("orbitcall.agent_tasks", tasks=lambda: [
        {"id": "catalog_dev", "split": "development", "family": "catalog"},
        {"id": "calendar_dev", "split": "development", "family": "calendar"}])
    monkeypatch.setattr(sys, "path", list(sys.path))
    path = Path(__file__).resolve().parents[1] / "examples/vllm_agent.py"
    spec = importlib.util.spec_from_file_location("orbitcall_dtype_demo", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return demo, state


@pytest.mark.parametrize("requested,resolved", [
    (None, torch.bfloat16), ("auto", torch.float16),
    ("float16", torch.float16), ("bfloat16", torch.bfloat16),
])
def test_demo_routes_requested_dtype_records_actual_resolution_without_mutating_engine(
        monkeypatch, tmp_path, requested, resolved):
    demo, state = load_demo(monkeypatch, resolved)
    output = tmp_path / "demo.json"
    arguments = ["demo", "--model", "public-checkpoint", "--output", str(output)]
    if requested is not None:
        arguments.extend(("--dtype", requested))
    monkeypatch.setattr(sys, "argv", arguments)
    demo.main()
    saved = json.loads(output.read_text(encoding="utf-8"))
    selected = "auto" if requested is None else requested
    assert state.created == [{"model": "public-checkpoint", "dtype": selected, "seed": 20261013}]
    assert saved["requested_dtype"] == selected
    assert saved["resolved_dtype"] == str(resolved).removeprefix("torch.")
    assert saved["engine"]["dtype"] == selected
    assert saved["scope"] == "development demonstration, not benchmark evidence"
    assert saved["answers_accessed"] is False
    assert state.defaults == {"dtype": "bfloat16", "seed": 20261013}
    assert len(state.pools) == len(state.shutdown) == 1
    assert state.pools[0][2] == "native_shape_suffix"


@pytest.mark.parametrize("requested,resolved,error", [
    ("auto", torch.float32, "native FP16/BF16"),
    ("float16", torch.bfloat16, "explicit override"),
    ("bfloat16", torch.float16, "explicit override"),
])
def test_demo_rejects_bad_resolution_before_agent_work_and_still_closes_engine(
        monkeypatch, tmp_path, requested, resolved, error):
    demo, state = load_demo(monkeypatch, resolved)
    output = tmp_path / "demo.json"
    monkeypatch.setattr(sys, "argv", ["demo", "--model", "public-checkpoint",
                                      "--dtype", requested, "--output", str(output)])
    with pytest.raises(RuntimeError, match=error):
        demo.main()
    assert not state.pools and not output.exists()
    assert len(state.shutdown) == 1
    assert state.defaults == {"dtype": "bfloat16", "seed": 20261013}


def test_demo_parser_rejects_unsupported_dtype_without_creating_engine(monkeypatch, tmp_path):
    demo, state = load_demo(monkeypatch, torch.bfloat16)
    monkeypatch.setattr(sys, "argv", ["demo", "--model", "public-checkpoint",
                                      "--dtype", "float32", "--output", str(tmp_path / "demo.json")])
    with pytest.raises(SystemExit) as error:
        demo.main()
    assert error.value.code == 2
    assert not state.created and not state.pools and not state.shutdown
