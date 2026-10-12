import numpy as np
import pytest

from orbitcall.trie import TokenTrie


def test_masks_include_only_shared_prompt_and_ancestors():
    tree = TokenTrie.build([[2, 3, 4], [2, 5, 4], [9, 4]])
    mask = tree.ancestor_mask(3)
    for path in tree.paths:
        for end, node in enumerate(path):
            assert np.flatnonzero(mask[node, 3:]).tolist() == sorted(path[:end + 1])
            assert mask[node, :3].all()
    assert tree.edge_count == 7
    assert tree.independent_token_count == 8


def test_duplicate_paths_rejected():
    with pytest.raises(ValueError):
        TokenTrie.build([[1, 2], [1, 2]])
