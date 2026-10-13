"""Right-padding omission must preserve actual causal Qwen2 likelihoods."""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from orbitcall.full_acceleration import PackedFullScorer
from orbitcall.scoring import SequenceScorer


@pytest.mark.parametrize("lengths", [(1, 2, 7), (9, 3), (2, 2, 2)])
@pytest.mark.parametrize("omit_mask", [False, True])
def test_full_padding_and_explicit_eos(lengths, omit_mask):
    if not torch.cuda.is_available():
        pytest.skip("Numerical tests require Spark")
    torch.manual_seed(801)
    config = Qwen2Config(vocab_size=79, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=128,
                         attention_dropout=0.0)
    config._attn_implementation = "eager"
    native = SequenceScorer.__new__(SequenceScorer)
    native.model = Qwen2ForCausalLM(config).cuda().float().eval()
    native.device = torch.device("cuda")
    native.head_chunk, native.pad_token, native.end_token = 32, 78, 77
    packed = PackedFullScorer.__new__(PackedFullScorer)
    packed.__dict__.update(native.__dict__)
    packed.omit_padding_mask = omit_mask
    prompt = [1, 3, 4, 5]
    sequences = [[6 + row] + [10 + row] * (length - 2) + [77]
                 if length > 1 else [77] for row, length in enumerate(lengths)]
    expected = native.score(prompt, sequences, "full", reducer="torch_ce", batch_size=2)
    actual = packed.score(prompt, sequences, "full", reducer="torch_ce", batch_size=2)
    assert expected.processed_prompt_positions == actual.processed_prompt_positions
    assert expected.processed_query_positions == actual.processed_query_positions
    assert [len(x) for x in actual.token_logprobs] == list(lengths)
    for left, right in zip(expected.token_logprobs, actual.token_logprobs, strict=True):
        torch.testing.assert_close(torch.tensor(left), torch.tensor(right), rtol=0, atol=3e-5)


def test_packed_rejects_invalid_paths_before_forward():
    scorer = PackedFullScorer.__new__(PackedFullScorer)
    for prompt, paths in (([], [[1]]), ([1], []), ([1], [[]])):
        with pytest.raises(ValueError, match="Nonempty"):
            scorer.score(prompt, paths)
    with pytest.raises(ValueError, match="does not implement"):
        scorer.score([1], [[2]], method="cached")
    with pytest.raises(ValueError, match="positive"):
        scorer.score([1], [[2]], batch_size=0)
