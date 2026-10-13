"""Client contract checks; execution belongs to the Spark test runner."""
import sys
from types import SimpleNamespace
import pytest
from orbitcall.vllm_scoring import VLLMScorer,decision_predictions


class Tokenizer:
    eos_token_id=0
    def encode(self,text,**kwargs): return list(text.encode())
    def apply_chat_template(self,*args,**kwargs): return "chat"


class FakeEngine:
    excessive_cache=False
    missing=False
    def get_tokenizer(self): return Tokenizer()
    def generate(self,prompts,params,**kwargs):
        rows=[]
        for prompt in prompts:
            ids=prompt["prompt_token_ids"]
            start=(params.extra_args or {}).get("orbitcall_score_start",0)
            values=[None]+[{t:SimpleNamespace(logprob=-float(t)/100)} for t in ids[1:]]
            if params.extra_args and "orbitcall_score_start" in params.extra_args: values[:start+1]=[None]*(start+1)
            if self.missing: values[-1]=None
            rows.append(SimpleNamespace(prompt_logprobs=values,
                        num_cached_tokens=999 if self.excessive_cache else 0))
        return rows


@pytest.fixture
def scorer(monkeypatch):
    def params(**kwargs): return SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules,"vllm",SimpleNamespace(SamplingParams=params))
    return VLLMScorer(FakeEngine())


def calls(): return [{"name":"add","arguments":{"a":3,"b":5}},
                     {"name":"add","arguments":{"a":3,"b":4}}]


def test_same_complete_families_and_eos_all_arms(scorer):
    results=[scorer.score([99,88],calls(),arm,"episode") for arm in
             ("native_full","suffix_no_cache","bounded_apc")]
    assert all(row.suffix_ids==results[0].suffix_ids for row in results)
    assert all(row.token_logprobs==results[0].token_logprobs for row in results)
    assert all(ids[-1]==0 for ids in results[0].suffix_ids)
    assert len(decision_predictions(results[0]))==11


def test_missing_eos_probability_fails(scorer):
    scorer.llm.missing=True
    with pytest.raises(RuntimeError,match="Missing requested"):
        scorer.score([99,88],calls(),"bounded_apc","episode")


def test_cache_cannot_skip_first_suffix_predictor(scorer):
    scorer.llm.excessive_cache=True
    with pytest.raises(RuntimeError,match="Cache skipped"):
        scorer.score([99,88],calls(),"bounded_apc","episode")


def test_null_candidate_family_is_retained(scorer):
    result=scorer.score([99,88],[],"bounded_apc","episode")
    assert result.winner is None and result.token_logprobs==[]
    assert set(decision_predictions(result).values())=={None}


def test_three_key_orbit_is_complete(scorer):
    result=scorer.score([99,88],[{"name":"sum3","arguments":{"c":3,"a":1,"b":2}}],
                        "bounded_apc","episode")
    assert len(result.suffix_ids)==6
    assert all(value==result.log_masses[0]["semantic_key"] for value in decision_predictions(result).values())


def test_template_uses_text_then_explicit_tokenization(scorer):
    assert scorer.prompt_ids([{"role":"user","content":"test"}])==list(b"chat")


def test_shape_arm_is_stock_bf16_no_cache_and_explicit_mode(scorer):
    original=scorer.llm.generate
    observed=[]
    def generate(prompts,params,**kwargs):
        observed.append(params)
        return original(prompts,params,**kwargs)
    scorer.llm.generate=generate
    stock=scorer.score([99,88],calls(),"native_default","episode")
    shape=scorer.score([99,88],calls(),"native_shape_suffix","episode")
    assert shape.token_logprobs==stock.token_logprobs
    assert observed[0].extra_args is None
    assert observed[1].extra_args=={"orbitcall_score_start":1,"orbitcall_score_mode":"native_shape_suffix"}
    assert observed[1].skip_reading_prefix_cache is True


def test_shape_arm_rejects_any_prefix_cache_read(scorer):
    scorer.llm.excessive_cache=True
    with pytest.raises(RuntimeError,match="bypass prefix"):
        scorer.score([99,88],calls(),"native_shape_suffix","episode")
