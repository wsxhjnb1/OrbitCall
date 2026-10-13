"""Opt-in full-prompt scoring with bulk packing and wider vocabulary projection.

No cache sharing or tree mask is used. Right padding can be left unmasked only
because a causal full-prompt forward cannot see future padding, and only valid
predictive positions are projected. Every sequence retains its explicit EOS.
"""

import torch

from .scoring import Scores, SequenceScorer


class PackedFullScorer(SequenceScorer):
    """Keep all independent paths while removing redundant host operations."""

    def __init__(self, *args, omit_padding_mask=True, **kwargs):
        super().__init__(*args, **kwargs)
        if (self.model.config.model_type != "qwen2"
                or getattr(self.model.config, "use_sliding_window", False)):
            raise ValueError("Packed full scorer supports only non-sliding Qwen2")
        self.omit_padding_mask = omit_padding_mask

    @torch.inference_mode()
    def score(self, prompt, sequences, method="full", reducer="triton", batch_size=8):
        if method != "full":
            raise ValueError("Packed full scorer does not implement cache/tree paths")
        if not prompt or not sequences or any(not sequence for sequence in sequences):
            raise ValueError("Nonempty prompt and suffix paths are required")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        p = len(prompt)
        all_values, processed_queries, processed_prompts = [], 0, 0
        for begin in range(0, len(sequences), batch_size):
            group = sequences[begin:begin + batch_size]
            counts = list(map(len, group))
            suffix_width = max(counts) - 1
            width = p + suffix_width
            packed = [list(prompt) + list(seq[:-1])
                      + [self.pad_token] * (suffix_width - len(seq) + 1)
                      for seq in group]
            ids = torch.tensor(packed, dtype=torch.long, device=self.device)
            positions = torch.arange(width, device=self.device).expand(len(group), -1)
            if self.omit_padding_mask:
                mask = None
            else:
                mask = torch.tensor([[1] * (p + count - 1)
                                     + [0] * (suffix_width - count + 1)
                                     for count in counts], dtype=torch.long, device=self.device)
            output = self.model.model(input_ids=ids, attention_mask=mask,
                                      position_ids=positions, use_cache=False).last_hidden_state
            # Positions have exactly the same semantic and batch order as FULL.
            predictive = torch.cat([output[row, p - 1:p + count - 1]
                                    for row, count in enumerate(counts)])
            targets = torch.tensor([token for seq in group for token in seq],
                                   dtype=torch.long, device=self.device)
            values = self._project(predictive, targets, reducer=reducer).cpu().tolist()
            offset = 0
            for count in counts:
                all_values.append(values[offset:offset + count])
                offset += count
            processed_queries += len(group) * suffix_width
            processed_prompts += len(group) * p
        self._check(all_values)
        return Scores([float(sum(values)) for values in all_values], all_values, p,
                      sum(counts for counts in map(len, sequences)),
                      sum(map(len, sequences)), "full", processed_prompts, processed_queries)
