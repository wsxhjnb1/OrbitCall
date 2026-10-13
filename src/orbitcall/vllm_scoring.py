"""Model-native vLLM finite-family scoring with request-scoped suffix reuse.

No Torch model sidecar is used. Native model/head/attention/logprob arithmetic
is retained. Only the integration plugin changes which causal rows are scored
and the permitted prefix-cache read boundary.
The released native_shape_suffix mode follows the engine's resolved model
dtype. The retained experimental FP32-output arms keep their original limits.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import time
from typing import Any

from .ranking import logsumexp
from .serialization import semantic_key, serialization_orbit

ARMS = ("native_full", "suffix_no_cache", "bounded_apc")
SYSTEM_ARMS = ("native_default",*ARMS)
SHAPE_ARMS = ("native_default", "native_shape_suffix")
SEEDS = (17, 29, 43, 71)


@dataclass
class ScoredFamily:
    calls: list[dict]
    suffix_ids: list[list[int]]
    token_logprobs: list[list[float]]
    sequence_logprobs: list[float]
    log_masses: list[dict]
    winner: dict | None
    cached_tokens: list[int]
    elapsed_seconds: float


class VLLMScorer:
    def __init__(self, llm: Any):
        self.llm = llm
        self.tokenizer = llm.get_tokenizer()

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        if not ids or any(type(token) is not int for token in ids):
            raise RuntimeError("Expected a nonempty integer chat-template token sequence")
        return ids

    @staticmethod
    def token_prompt(ids: list[int], salt: str) -> dict:
        return {"prompt_token_ids": ids, "cache_salt": salt}

    def generate(self, messages: list[dict], salt: str, *, final=False) -> dict:
        return self.generate_many([messages],[salt],final=final)[0]

    def generate_many(self, messages: list[list[dict]], salts: list[str], *, final=False, single=False) -> list[dict]:
        from vllm import SamplingParams
        begin = time.perf_counter()
        if len(messages)!=len(salts):
            raise ValueError("Messages/salts mismatch")
        ids = [self.prompt_ids(row) for row in messages]
        seeds=(17,) if final or single else SEEDS
        params = [SamplingParams(temperature=0.0 if final or single else 0.7,
                                 top_p=1.0 if final or single else 0.9, seed=seed,
                                 max_tokens=64 if final else 128,
                                 skip_special_tokens=True)
                  for _ in messages for seed in seeds]
        outputs = self.llm.generate([self.token_prompt(prompt,salt) for prompt,salt in zip(ids,salts,strict=True) for _ in seeds],
                                    params,use_tqdm=False)
        elapsed=time.perf_counter()-begin
        records=[]
        for index,prompt in enumerate(ids):
            selected=outputs[index*len(seeds):(index+1)*len(seeds)]
            raw=[{"seed":seed,"text":out.outputs[0].text,"token_ids":list(out.outputs[0].token_ids),
                  "finish_reason":out.outputs[0].finish_reason,"cached_tokens":int(out.num_cached_tokens)}
                 for seed,out in zip(seeds,selected,strict=True)]
            records.append({"prompt_ids":prompt,"outputs":raw,"batch_elapsed_seconds":elapsed})
        return records

    def score(self, prompt_ids: list[int], calls: list[dict], arm: str, salt: str) -> ScoredFamily:
        from vllm import SamplingParams
        if arm not in (*SYSTEM_ARMS,"native_shape_suffix") or not prompt_ids:
            raise ValueError("Unsupported arm or empty prompt")
        begin = time.perf_counter()
        unique = {semantic_key(call): call for call in calls}
        ordered = [unique[key] for key in sorted(unique)]
        ids, groups = [], []
        for call in ordered:
            sequences = []
            for text in serialization_orbit(call):
                suffix = self.tokenizer.encode(text, add_special_tokens=False)
                if self.tokenizer.eos_token_id is None or self.tokenizer.eos_token_id in suffix:
                    raise ValueError("Unsafe termination")
                sequences.append(suffix + [self.tokenizer.eos_token_id])
            distinct = list(dict.fromkeys(tuple(seq) for seq in sequences))
            groups.append((call, len(distinct)))
            ids.extend([list(seq) for seq in distinct])
        if not ids:
            return ScoredFamily([], [], [], [], [], None, [], time.perf_counter() - begin)
        start = len(prompt_ids) - 1
        tagged = arm in ("suffix_no_cache","bounded_apc","native_shape_suffix")
        extras={} if arm in SHAPE_ARMS else {"orbitcall_score_logit_dtype":"float32"}
        if tagged: extras["orbitcall_score_start"]=start
        if arm=="native_shape_suffix": extras["orbitcall_score_mode"]="native_shape_suffix"
        params = SamplingParams(prompt_logprobs=0, max_tokens=1, temperature=0.0,
                                detokenize=False, flat_logprobs=False,
                                skip_reading_prefix_cache=arm != "bounded_apc",
                                extra_args=extras or None)
        outputs = self.llm.generate([self.token_prompt(prompt_ids + suffix, salt) for suffix in ids],
                                    params, use_tqdm=False)
        arrays, cached = [], []
        for suffix, output in zip(ids, outputs, strict=True):
            rows = output.prompt_logprobs
            if rows is None or len(rows) != len(prompt_ids) + len(suffix):
                raise RuntimeError("Incomplete or misaligned prompt probabilities")
            if tagged and any(row is not None for row in rows[:len(prompt_ids)]):
                raise RuntimeError("Unscored prefix must be explicit None")
            values = []
            for token_id, row in zip(suffix, rows[len(prompt_ids):], strict=True):
                if row is None or token_id not in row:
                    raise RuntimeError("Missing requested suffix token probability")
                value = float(row[token_id].logprob)
                if not math.isfinite(value):
                    raise RuntimeError("Nonfinite score")
                values.append(value)
            arrays.append(values)
            cached.append(int(output.num_cached_tokens))
            if arm=="native_shape_suffix" and cached[-1]!=0:
                raise RuntimeError("Shape-preserving scoring must bypass prefix reads")
            if arm == "bounded_apc" and cached[-1] > start:
                raise RuntimeError("Cache skipped a required causal score row")
        totals = [math.fsum(values) for values in arrays]
        masses, offset = [], 0
        for call, count in groups:
            masses.append({"call": call, "semantic_key": semantic_key(call),
                           "log_mass": logsumexp(totals[offset:offset + count]), "family_size": count})
            offset += count
        winner = sorted(masses, key=lambda row: (-row["log_mass"], row["semantic_key"]))[0]["call"]
        return ScoredFamily(ordered, ids, arrays, totals, masses, winner, cached,
                            time.perf_counter() - begin)


def decision_predictions(result: ScoredFamily) -> dict:
    """Eleven fixed diagnostics; canonical order is alphabetical in this study."""
    names = ["mass", "max_order", "mean_logprob", "length_normalized", "canonical_alphabetic"]
    permutations = list(itertools.permutations(range(3)))
    names += ["fixed_" + "".join(map(str, order)) for order in permutations]
    if not result.calls:
        return {name: None for name in names}
    scores = {name: [] for name in names}
    offset = 0
    for row in result.log_masses:
        count = row["family_size"]
        values = result.sequence_logprobs[offset:offset+count]
        lengths = [len(ids) for ids in result.suffix_ids[offset:offset+count]]
        key = row["semantic_key"]
        basics = (row["log_mass"], max(values), math.fsum(values)/count,
                  math.fsum(v/n for v,n in zip(values,lengths,strict=True))/count, values[0])
        for name, value in zip(names[:5], basics, strict=True):
            scores[name].append((value,key))
        nargs = len(row["call"]["arguments"])
        valid_orders = list(itertools.permutations(range(nargs)))
        for name, order in zip(names[5:],permutations,strict=True):
            index = valid_orders.index(tuple(i for i in order if i<nargs))
            scores[name].append((values[index],key))
        offset += count
    return {name: sorted(values,key=lambda v:(-v[0],v[1]))[0][1] for name,values in scores.items()}
