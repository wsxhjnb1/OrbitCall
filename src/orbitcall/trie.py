"""Token-prefix trees for a fixed, deduplicated sequence family."""

from dataclasses import dataclass

import numpy as np


@dataclass
class TokenTrie:
    tokens: list[int]
    parents: list[int]
    depths: list[int]
    paths: list[list[int]]

    @classmethod
    def build(cls, sequences):
        tokens, parents, depths, paths = [], [], [], []
        children = {-1: {}}
        seen = set()
        for sequence in sequences:
            sequence = tuple(int(x) for x in sequence)
            if not sequence or sequence in seen:
                raise ValueError("Sequences must be nonempty and distinct")
            seen.add(sequence)
            parent, path = -1, []
            for depth, token in enumerate(sequence, 1):
                branch = children.setdefault(parent, {})
                if token not in branch:
                    branch[token] = len(tokens)
                    tokens.append(token)
                    parents.append(parent)
                    depths.append(depth)
                parent = branch[token]
                path.append(parent)
            paths.append(path)
        return cls(tokens, parents, depths, paths)

    def ancestor_mask(self, prompt_length):
        count = len(self.tokens)
        mask = np.zeros((count, prompt_length + count), dtype=np.bool_)
        mask[:, :prompt_length] = True
        for node, parent in enumerate(self.parents):
            if parent >= 0:
                mask[node, prompt_length:] = mask[parent, prompt_length:]
            mask[node, prompt_length + node] = True
        return mask

    @property
    def edge_count(self):
        return len(self.tokens)

    @property
    def independent_token_count(self):
        return sum(map(len, self.paths))
