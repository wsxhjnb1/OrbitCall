"""Checkpointed Spark-only candidate generation; never reads answer files."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from orbitcall.bfcl import load_frozen_inputs, render_messages
from orbitcall.gpu_guard import require_idle_gpu, require_owned_gpu
from orbitcall.serialization import parse_call, semantic_key


def dump(path, value):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("data/frozen_inputs.json"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or "GB10" not in torch.cuda.get_device_name():
        raise RuntimeError("All inference must run on Spark GB10")
    records = load_frozen_inputs(args.inputs)
    require_idle_gpu()
    if args.run.exists() and not args.resume:
        raise FileExistsError(args.run)
    args.run.mkdir(parents=True, exist_ok=True)
    manifest = {"model": Path(args.model).name, "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                "generation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "batch_size": args.batch_size, "dtype": "bfloat16", "attention": "sdpa",
                "direct_seeds": [17, 29, 43, 71], "additional_seeds": [101, 113, 127, 139],
                "batch_seed_rule": "base_seed + 1000003 * batch_index",
                "greedy_max_tokens": 128, "direct_max_tokens": 128,
                "freeform_max_tokens": 64, "reformat_max_tokens": 64,
                "temperature": 0.7, "top_p": 0.9, "answers_accessed": False,
                "record_ids": [r["id"] for r in records]}
    manifest_path = args.run / "generation_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("Refuse to resume changed generation inputs/code/config")
    else:
        dump(manifest_path, manifest)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
                                                dtype=torch.bfloat16,
                                                attn_implementation="sdpa").to("cuda").eval()
    plans = [("greedy", 0)] + [("direct", x) for x in [17, 29, 43, 71, 101, 113, 127, 139]]
    plans += [("freeform", x) for x in [17, 29, 43, 71]]
    plans += [("reformat", x) for x in [17, 29, 43, 71]]
    for mode, seed in plans:
        for index, begin in enumerate(range(0, len(records), args.batch_size)):
            path = args.run / "generation" / f"{mode}_{seed}_{index:03d}.json"
            if path.exists():
                continue
            group = records[begin:begin + args.batch_size]
            conversations = []
            if mode == "reformat":
                previous = json.loads((args.run / "generation" / f"freeform_{seed}_{index:03d}.json").read_text())
            for offset, row in enumerate(group):
                messages = render_messages(row)
                if mode == "freeform":
                    schema = json.dumps(row["function"][0], ensure_ascii=False)
                    messages[0] = {"role": "system", "content":
                                   "Identify the required argument values for this function. Explain briefly in plain text; do not format JSON.\n" + schema}
                if mode == "reformat":
                    messages += [{"role": "assistant", "content": previous["samples"][offset]["text"]},
                                 {"role": "user", "content": "Now return only the compact JSON call in the required name/arguments envelope."}]
                conversations.append(tokenizer.apply_chat_template(messages, tokenize=False,
                                                                      add_generation_prompt=True))
            inputs = tokenizer(conversations, return_tensors="pt", padding=True,
                               add_special_tokens=False).to("cuda")
            require_owned_gpu()
            torch.manual_seed(seed + 1000003 * index)
            torch.cuda.synchronize()
            started = time.time()
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=64 if mode in {"freeform", "reformat"} else 128,
                                         do_sample=mode != "greedy", temperature=0.7 if mode != "greedy" else None,
                                         top_p=0.9 if mode != "greedy" else None,
                                         eos_token_id=tokenizer.eos_token_id,
                                         pad_token_id=tokenizer.pad_token_id)
            torch.cuda.synchronize()
            ended = time.time()
            require_owned_gpu()
            generated = outputs[:, inputs.input_ids.shape[1]:].cpu().tolist()
            samples = []
            for row, tokens in zip(group, generated):
                if tokenizer.eos_token_id in tokens:
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                text = tokenizer.decode(tokens, skip_special_tokens=True)
                parsed, error = None, None
                if mode != "freeform":
                    try:
                        parsed = parse_call(text, row["function"][0])
                    except ValueError as exc:
                        error = str(exc)
                samples.append({"id": row["id"], "text": text, "token_ids": tokens,
                                "call": parsed, "parse_error": error,
                                "semantic_key": semantic_key(parsed) if parsed else None})
            dump(path, {"mode": mode, "base_seed": seed, "batch_index": index,
                        "started_epoch": started, "ended_epoch": ended,
                        "wall_seconds": ended - started,
                        "input_lengths": inputs.attention_mask.sum(-1).cpu().tolist(),
                        "samples": samples})
            print(json.dumps({"stage": mode, "seed": seed, "batch": index,
                              "seconds": ended - started}), flush=True)
    rows = {r["id"]: {"id": r["id"], "split": r["split"], "schema_group": r["schema_group"],
                        "function": r["function"][0], "samples": []} for r in records}
    for path in sorted((args.run / "generation").glob("*.json")):
        batch = json.loads(path.read_text())
        for sample in batch["samples"]:
            rows[sample["id"]]["samples"].append({**sample, "mode": batch["mode"],
                                                  "seed": batch["base_seed"]})
    dump(args.run / "candidates.json", {"records": list(rows.values()), "answers_accessed": False})
    print(json.dumps({"status": "candidates_complete", "records": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
