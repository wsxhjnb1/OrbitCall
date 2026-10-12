"""Matched full, prompt-cached, and packed-tree teacher-forced scorers."""

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.cache_utils import DynamicLayer

from .kernels import selected_logprobs
from .trie import TokenTrie


@dataclass
class Scores:
    sequence_logprobs: list[float]
    token_logprobs: list[list[float]]
    prompt_tokens: int
    independent_suffix_tokens: int
    scored_edges: int
    method: str
    processed_prompt_positions: int = 0
    processed_query_positions: int = 0


class SequenceScorer:
    def __init__(self, model_path, dtype="float32", attention="eager", head_chunk=32):
        if not torch.cuda.is_available():
            raise RuntimeError("Model experiments require the Spark GPU")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=getattr(torch, dtype),
            attn_implementation=attention,
        ).to("cuda").eval()
        self.head_chunk = head_chunk
        self.device = torch.device("cuda")
        self.end_token = self.tokenizer.eos_token_id
        self.pad_token = self.tokenizer.pad_token_id or self.end_token

    def prompt_ids(self, messages):
        value = self.tokenizer.apply_chat_template(messages, tokenize=True,
                                                    add_generation_prompt=True)
        if hasattr(value, "keys"):
            value = value["input_ids"]
        if isinstance(value, torch.Tensor):
            value = value.tolist()
        if value and isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError("Scorer expects one conversation")
            value = value[0]
        return [int(token) for token in value]

    def sequence_ids(self, texts):
        result = []
        stop_ids = {self.end_token}
        configured = self.model.generation_config.eos_token_id
        stop_ids.update(configured if isinstance(configured, list) else [configured])
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            if any(token in stop_ids for token in tokens):
                raise ValueError("Serialization contains an interior stop token")
            result.append(tokens + [self.end_token])
        if len(set(map(tuple, result))) != len(result):
            raise ValueError("Serialization family contains duplicate token paths")
        return result

    def _new_cache(self, pairs, batch=1):
        # DynamicCache(ddp_cache_data=...) concatenates each supplied prefix
        # with an empty tensor and copies the expanded batch. Initialize fresh
        # DynamicLayer wrappers from immutable views instead. Its later update
        # concatenates suffixes into new storage and does not mutate the prefix.
        cache = DynamicCache(config=self.model.config)
        if len(cache.layers) != len(pairs):
            raise RuntimeError("Cache layer count differs from immutable prefix")
        for layer, (key, value) in zip(cache.layers, pairs, strict=True):
            if type(layer) is not DynamicLayer:
                raise RuntimeError("Only non-sliding DynamicLayer prefix views are supported")
            layer.keys = key.expand(batch, -1, -1, -1)
            layer.values = value.expand(batch, -1, -1, -1)
            layer.dtype, layer.device = key.dtype, key.device
            layer.is_initialized = True
        return cache

    @torch.inference_mode()
    def _prefill(self, prompt):
        ids = torch.tensor([prompt], device=self.device)
        out = self.model.model(input_ids=ids, use_cache=True)
        pairs = [(layer.keys, layer.values) for layer in out.past_key_values.layers]
        return pairs, out.last_hidden_state[0, -1]

    def _project(self, hidden, targets, row_ids=None, reducer="torch"):
        # Chunking limits LM-head workspace. Normalize against the full vocabulary.
        values = torch.empty(len(targets), device=self.device, dtype=torch.float32)
        for begin in range(0, hidden.shape[0], self.head_chunk):
            end = min(begin + self.head_chunk, hidden.shape[0])
            logits = self.model.lm_head(hidden[begin:end]).float()
            if row_ids is None:
                values[begin:end] = selected_logprobs(logits, targets[begin:end],
                                                     backend=reducer)
            else:
                # Plan membership on CPU from the immutable tree, avoiding
                # nonzero/boolean indexing of GPU tensors and its synchronization.
                indices = [i for i, row in enumerate(row_ids) if begin <= row < end]
                positions = torch.tensor(indices, device=self.device)
                rows = torch.tensor([row_ids[i] - begin for i in indices], device=self.device)
                values[positions] = selected_logprobs(
                    logits, targets[positions], row_ids=rows, backend=reducer)
        return values

    @torch.inference_mode()
    def score(self, prompt, sequences, method="full", reducer="torch", batch_size=8):
        if method not in {"full", "cached", "tree"}:
            raise ValueError(method)
        if not sequences:
            raise ValueError("Empty sequence family")
        if method == "tree":
            return self._tree(prompt, sequences, reducer)
        p = len(prompt)
        pairs, root_hidden = self._prefill(prompt) if method == "cached" else (None, None)
        all_values = []
        processed_queries = 0
        processed_prompts = p if method == "cached" else 0
        for begin in range(0, len(sequences), batch_size):
            group = sequences[begin:begin + batch_size]
            width = max(map(len, group)) - 1
            total_width = width + (p if method == "full" else 0)
            ids = torch.full((len(group), total_width), self.pad_token,
                             device=self.device, dtype=torch.long)
            mask = torch.zeros((len(group), p + width), device=self.device,
                               dtype=torch.long)
            for row, sequence in enumerate(group):
                values = prompt + list(sequence[:-1]) if method == "full" else list(sequence[:-1])
                ids[row, :len(values)] = torch.tensor(values, device=self.device)
                mask[row, :p + len(sequence) - 1] = 1
            positions = torch.arange(total_width, device=self.device).expand(len(group), -1)
            if method == "cached":
                positions = positions + p
            if total_width:
                output = self.model.model(
                    input_ids=ids, attention_mask=mask, position_ids=positions,
                    past_key_values=self._new_cache(pairs, len(group)) if pairs else None,
                    use_cache=False,
                ).last_hidden_state
            else:
                output = root_hidden.new_empty((len(group), 0, root_hidden.numel()))
            processed_queries += len(group) * width
            if method == "full":
                processed_prompts += len(group) * p
            hidden, targets, counts = [], [], []
            for row, sequence in enumerate(group):
                length = len(sequence)
                if method == "full":
                    hidden.append(output[row, p - 1:p + length - 1])
                else:
                    hidden.append(torch.cat((root_hidden[None], output[row, :length - 1]), 0))
                targets.extend(sequence)
                counts.append(length)
            values = self._project(torch.cat(hidden), torch.tensor(targets, device=self.device),
                                   reducer=reducer).cpu().tolist()
            offset = 0
            for count in counts:
                all_values.append(values[offset:offset + count])
                offset += count
        self._check(all_values)
        return Scores([float(sum(x)) for x in all_values], all_values, p,
                      sum(map(len, sequences)), sum(map(len, sequences)), method,
                      processed_prompts, processed_queries)

    @torch.inference_mode()
    def _tree(self, prompt, sequences, reducer):
        tree = TokenTrie.build(sequences)
        p = len(prompt)
        pairs, root_hidden = self._prefill(prompt)
        parent_nodes = sorted(set(tree.parents))
        active = [node for node in parent_nodes if node >= 0]
        ids = torch.tensor([[tree.tokens[node] for node in active]], device=self.device)
        positions = torch.tensor([[p + tree.depths[node] - 1 for node in active]], device=self.device)
        columns = list(range(p)) + [p + node for node in active]
        allowed = torch.from_numpy(tree.ancestor_mask(p)[np.ix_(active, columns)]).to(self.device)
        attention_mask = torch.zeros(allowed.shape, dtype=self.model.dtype, device=self.device)
        attention_mask.masked_fill_(~allowed, float("-inf"))
        if active:
            output = self.model.model(
                input_ids=ids, attention_mask=attention_mask[None, None],
                position_ids=positions, past_key_values=self._new_cache(pairs),
                use_cache=False,
            ).last_hidden_state[0]
        else:
            output = root_hidden.new_empty((0, root_hidden.numel()))
        active_rows = {node: row for row, node in enumerate(active)}
        parent_rows = {node: row for row, node in enumerate(parent_nodes)}
        hidden = torch.stack([root_hidden if node == -1 else output[active_rows[node]] for node in parent_nodes])
        rows = [parent_rows[node] for node in tree.parents]
        targets = torch.tensor(tree.tokens, device=self.device)
        edge_values = self._project(hidden, targets, rows, reducer).cpu().tolist()
        values = [[edge_values[node] for node in path] for path in tree.paths]
        self._check(values)
        return Scores([float(sum(x)) for x in values], values, p,
                      tree.independent_token_count, tree.edge_count, "tree", p, len(active))

    @staticmethod
    def _check(values):
        if any(not np.isfinite(x) for sequence in values for x in sequence):
            raise RuntimeError("Non-finite likelihood; result invalid")
