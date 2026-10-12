"""Finite-table controls; execute these tests on DGX Spark, not Windows."""

from fractions import Fraction
import random

import pytest

from orbitcall.certificates import FiniteProbabilityTree, certify_semantic_map


EOS = -1


def _independent_enumeration(candidates, probabilities):
    result = {}
    for label, sequences in candidates.items():
        total = Fraction(0)
        for sequence in sequences:
            prefix, probability = (), Fraction(1)
            for token in sequence:
                probability *= Fraction(probabilities[prefix][token])
                prefix = (*prefix, token)
            total += probability
        result[label] = total
    return result


def _assert_every_bound_is_sound(audit, expected):
    previous_lower, previous_upper = None, None
    for snapshot in audit.history:
        lower, upper = dict(snapshot.lower), dict(snapshot.upper)
        for label, mass in expected.items():
            assert 0 <= lower[label] <= mass <= upper[label]
            if previous_lower is not None:
                assert lower[label] >= previous_lower[label]
                assert upper[label] <= previous_upper[label]
        for left in snapshot.frontier:
            for right in snapshot.frontier:
                if left != right:
                    assert not (len(left) <= len(right) and right[: len(left)] == left)
        previous_lower, previous_upper = lower, upper


def test_semantic_mass_can_differ_from_maximum_single_serialization():
    candidates = {
        "A": [(1, EOS), (2, EOS)],
        "B": [(3, EOS), (4, EOS)],
    }
    probabilities = {
        (): {1: "0.30", 2: "0.30", 3: "0.35", 4: "0.05"},
        (1,): {EOS: 1},
        (2,): {EOS: 1},
        (3,): {EOS: 1},
        (4,): {EOS: 1},
    }
    tree = FiniteProbabilityTree(candidates, probabilities)
    masses = tree.enumerate_masses()
    assert masses == {"A": Fraction(3, 5), "B": Fraction(2, 5)}
    assert tree.sequence_probability((3, EOS)) > tree.sequence_probability((1, EOS))
    # Equal family sizes make arithmetic-mean probability a redundant ranking.
    assert max(masses, key=masses.__getitem__) == max(
        masses, key=lambda label: masses[label] / len(candidates[label])
    )
    audit = tree.audit()
    assert audit.certified and audit.winner == "A"
    _assert_every_bound_is_sound(audit, masses)


def test_strict_bound_stops_before_low_mass_branch_is_expanded():
    candidates = {"A": [(1, EOS)], "B": [(2, EOS)]}
    probabilities = {
        (): {1: "0.9", 2: "0.1"},
        (1,): {EOS: 1},
        (2,): {EOS: 1},
    }
    tree = FiniteProbabilityTree(candidates, probabilities)
    audit = tree.audit()
    assert audit.winner == "A" and audit.reason == "strict_bounds"
    assert audit.expansions == 2
    assert audit.history[-1].frontier == ((2,),)
    _assert_every_bound_is_sound(audit, _independent_enumeration(candidates, probabilities))
    exported = audit.as_dict()
    assert exported["scope"] == "supplied_finite_probability_table_only"
    assert exported["history"][-1]["lower"]["A"] == "9/10"


def test_expansion_limit_fails_closed_and_ties_use_declared_order():
    candidates = {"first": [(1, EOS)], "second": [(2, EOS)]}
    probabilities = {(): {1: 0, 2: 0}, (1,): {EOS: 1}, (2,): {EOS: 1}}
    tree = FiniteProbabilityTree(candidates, probabilities)
    limited = tree.audit(max_expansions=1)
    assert not limited.certified and limited.winner is None
    assert limited.reason == "expansion_limit"
    complete = tree.audit()
    assert complete.winner == "first" and complete.reason == "exhaustive"
    _assert_every_bound_is_sound(complete, {"first": Fraction(0), "second": Fraction(0)})
    assert certify_semantic_map(candidates, probabilities).winner == complete.winner


def test_unrepresented_vocabulary_mass_is_not_renormalized():
    candidates = {"A": [(1, EOS)], "B": [(2, EOS)]}
    probabilities = {
        (): {1: "0.1", 2: "0.2", 99: "0.7"},
        (1,): {EOS: "0.5"},
        (2,): {EOS: "0.5"},
    }
    tree = FiniteProbabilityTree(candidates, probabilities)
    assert tree.enumerate_masses() == {"A": Fraction(1, 20), "B": Fraction(1, 10)}
    _assert_every_bound_is_sound(tree.audit(), tree.enumerate_masses())


def _random_probability_tree(seed):
    generator = random.Random(seed)
    candidates, probabilities = {}, {}
    max_depth = generator.randint(1, 4)

    def visit(prefix):
        if len(prefix) >= max_depth or (prefix and generator.random() < 0.3):
            probabilities[prefix] = {EOS: Fraction(generator.randint(0, 9), 10)}
            label = f"class_{generator.randrange(4)}"
            candidates.setdefault(label, []).append((*prefix, EOS))
            return
        tokens = list(range(1, generator.randint(1, 4) + 1))
        weights = [generator.randint(0, 9) for _ in tokens]
        # Strictly positive denominator; a random slack represents other tokens.
        denominator = max(1, sum(weights) + generator.randint(0, 9))
        probabilities[prefix] = {
            token: Fraction(weight, denominator)
            for token, weight in zip(tokens, weights, strict=True)
        }
        for token in tokens:
            visit((*prefix, token))

    visit(())
    return candidates, probabilities


@pytest.mark.parametrize("seed", range(100))
def test_random_tree_bounds_and_winner_match_independent_full_enumeration(seed):
    candidates, probabilities = _random_probability_tree(seed)
    expected = _independent_enumeration(candidates, probabilities)
    tree = FiniteProbabilityTree(candidates, probabilities)
    assert tree.enumerate_masses() == expected
    audit = tree.audit()
    assert audit.certified
    assert audit.winner == max(expected, key=expected.__getitem__)
    _assert_every_bound_is_sound(audit, expected)
    limited = tree.audit(max_expansions=seed % 5)
    _assert_every_bound_is_sound(limited, expected)
    if limited.certified:
        assert limited.winner == audit.winner


def test_invalid_tables_and_candidate_sets_are_rejected():
    candidates = {"A": [(1, EOS)], "B": [(2, EOS)]}
    valid = {(): {1: "0.5", 2: "0.5"}, (1,): {EOS: 1}, (2,): {EOS: 1}}
    with pytest.raises(ValueError, match="duplicate"):
        FiniteProbabilityTree({"A": [(1, EOS), (1, EOS)]}, valid)
    with pytest.raises(ValueError, match="duplicate"):
        FiniteProbabilityTree({"A": [(1, EOS)], "B": [(1, EOS)]}, valid)
    with pytest.raises(ValueError, match="prefix-free"):
        FiniteProbabilityTree({"A": [(1,)], "B": [(1, EOS)]}, valid)
    with pytest.raises(ValueError, match="prefix-free"):
        FiniteProbabilityTree({"A": [(1, EOS)], "B": [(1,)]}, valid)
    with pytest.raises(ValueError, match="termination"):
        FiniteProbabilityTree({"A": [()]}, {})
    with pytest.raises(ValueError, match="at least one sequence"):
        FiniteProbabilityTree({"A": []}, {})
    with pytest.raises(ValueError, match="missing probability row"):
        FiniteProbabilityTree(candidates, {})
    with pytest.raises(ValueError, match="missing trie-child"):
        FiniteProbabilityTree(candidates, {**valid, (): {1: "0.5"}})
    with pytest.raises(ValueError, match="exceeds unit mass"):
        FiniteProbabilityTree(candidates, {**valid, (): {1: "0.6", 2: "0.6"}})
    for invalid in [-0.1, 1.1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            FiniteProbabilityTree(candidates, {**valid, (): {1: invalid, 2: 0}})
    tree = FiniteProbabilityTree(candidates, valid)
    with pytest.raises(ValueError, match="declared complete"):
        tree.sequence_probability((1,))
    for limit in [-1, 1.5, True]:
        with pytest.raises(ValueError, match="nonnegative integer"):
            tree.audit(max_expansions=limit)


def test_float_entries_keep_binary_value_and_no_normalization_is_silent():
    candidates = {"A": [(1, EOS)], "B": [(2, EOS)]}
    # In binary64, exact stored 0.1 + exact stored 0.9 is slightly above one.
    with pytest.raises(ValueError, match="exceeds unit mass"):
        FiniteProbabilityTree(
            candidates, {(): {1: 0.1, 2: 0.9}, (1,): {EOS: 1}, (2,): {EOS: 1}}
        )
    tree = FiniteProbabilityTree(
        candidates, {(): {1: "0.1", 2: "0.9"}, (1,): {EOS: 1}, (2,): {EOS: 1}}
    )
    assert sum(tree.enumerate_masses().values()) == 1
