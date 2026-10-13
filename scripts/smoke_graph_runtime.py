"""Spark-only tiny-Qwen graph API smoke, outside scientific fixture selection."""
import json
import os
from pathlib import Path
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from orbitcall.scoring import SequenceScorer
from orbitcall.graph_scoring import GraphFullScorer
from orbitcall.gpu_guard import require_idle_gpu
from benchmark_scoring import write_new, sha256

require_idle_gpu()
out = Path("runs/graph_runtime_smoke_v2")
out.mkdir(exist_ok=False)
import shutil
for source in (Path(__file__), Path("src/orbitcall/graph_scoring.py")):
    relative = Path("scripts") / source.name if source == Path(__file__) else source
    destination = out / "source" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
try:
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(771)
    c = Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=128,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    attention_dropout=0.0, max_position_embeddings=512)
    c._attn_implementation = "eager"
    native = SequenceScorer.__new__(SequenceScorer)
    native.model = Qwen2ForCausalLM(c).cuda().float().eval()
    native.tokenizer = None
    native.device = torch.device("cuda")
    native.pad_token, native.end_token, native.head_chunk = 96, 95, 32
    graph = GraphFullScorer.from_scorer(native)
    prompt, paths = [1, 2, 3, 4], [[6, 7, 95], [8, 95]]
    reference = native.score(prompt, paths, "full", "torch_ce")
    first = graph.score(prompt, paths, "full", "torch_ce")
    for left, right in zip(reference.token_logprobs, first.token_logprobs, strict=True):
        torch.testing.assert_close(torch.tensor(left), torch.tensor(right), rtol=0, atol=3e-5)
    prompt2 = [9, 11, 10, 12]
    second = graph.score(prompt2, paths, "full", "torch_ce")
    reference2 = native.score(prompt2, paths, "full", "torch_ce")
    for left, right in zip(reference2.token_logprobs, second.token_logprobs, strict=True):
        torch.testing.assert_close(torch.tensor(left), torch.tensor(right), rtol=0, atol=3e-5)
    write_new(out / "complete.json", {"valid": True, "graph_stats": graph.graph_stats,
              "source_sha256": sha256(Path("src/orbitcall/graph_scoring.py")),
              "scope": "tiny random network API validation, not checkpoint quality/performance"})
    print(json.dumps(graph.graph_stats))
except Exception as exc:
    write_new(out / "failure.json", {"exception": str(exc), "traceback": traceback.format_exc()})
    raise
