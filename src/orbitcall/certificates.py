"""Optional exact audits of a supplied finite, substochastic probability tree.

This module does not query a model, estimate unseen probabilities, or accelerate
inference. Fractions make the supplied table's arithmetic exact. The resulting
certificate is about that table and candidate set, not a GPU model's numerical
error or the probability of all possible semantically equivalent outputs.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import TypeAlias


Token: TypeAlias = Hashable
Prefix: TypeAlias = tuple[Token, ...]
Probability: TypeAlias = Fraction | Decimal | float | int | str


def _probability(value: Probability) -> Fraction:
    """Interpret floats as their exact stored binary value, without rounding."""
    if isinstance(value, bool):
        raise TypeError("Boolean values are not probability-table entries")
    try:
        result = Fraction(value)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as error:
        raise ValueError("probabilities must be finite rational-convertible values") from error
    if result < 0 or result > 1:
        raise ValueError("every conditional probability must lie in [0, 1]")
    return result


@dataclass(frozen=True)
class BoundSnapshot:
    """Exact rational lower/upper bounds in declared candidate-class order."""

    lower: tuple[tuple[str, Fraction], ...]
    upper: tuple[tuple[str, Fraction], ...]
    frontier: tuple[Prefix, ...]
    expanded_prefix: Prefix | None

    def as_dict(self) -> dict:
        return {
            "lower": {label: str(value) for label, value in self.lower},
            "upper": {label: str(value) for label, value in self.upper},
            "frontier": [list(prefix) for prefix in self.frontier],
            "expanded_prefix": (
                None if self.expanded_prefix is None else list(self.expanded_prefix)
            ),
        }


@dataclass(frozen=True)
class CertificateAudit:
    """Retained table-level trace; an expansion limit returns no winner."""

    winner: str | None
    reason: str
    expansions: int
    history: tuple[BoundSnapshot, ...]

    @property
    def certified(self) -> bool:
        return self.winner is not None

    def as_dict(self) -> dict:
        return {
            "scope": "supplied_finite_probability_table_only",
            "winner": self.winner,
            "reason": self.reason,
            "expansions": self.expansions,
            "certified": self.certified,
            "history": [snapshot.as_dict() for snapshot in self.history],
        }


class FiniteProbabilityTree:
    """Prefix-free finite sequences, grouped by semantic class.

    Args:
        candidates: Class ID -> nonempty iterable of complete token sequences.
            Include the declared explicit termination token in every sequence.
            Duplicate sequences, including within a class, are rejected.
        conditionals: Prefix -> token -> conditional probability. Every used
            prefix must supply all trie-child probabilities. A supplied row may
            include vocabulary tokens outside the finite candidate trie; its
            total must be at most one. Unspecified mass is allowed and ignored.

    The constructor checks substochasticity with exact Fraction arithmetic. It
    never renormalizes, clips, or adds tolerance to a supplied table. Rounded
    GPU probabilities that sum above one are outside this certificate contract.
    A caller may explicitly construct another normalized reference table, but
    must identify it as such rather than claiming a certificate for raw scores.
    """

    def __init__(
        self,
        candidates: Mapping[str, Iterable[Sequence[Token]]],
        conditionals: Mapping[Prefix, Mapping[Token, Probability]],
    ):
        if not candidates:
            raise ValueError("at least one semantic class is required")
        self.classes = tuple(candidates)
        if not all(isinstance(label, str) for label in self.classes):
            raise TypeError("semantic class IDs must be strings")
        self._children: dict[Prefix, dict[Token, Prefix]] = {(): {}}
        self._descendants: dict[Prefix, set[str]] = {(): set()}
        self._terminal: dict[Prefix, str] = {}
        self._sequences: dict[str, tuple[Prefix, ...]] = {}
        for label, family in candidates.items():
            paths = []
            for sequence in family:
                path = tuple(sequence)
                if not path:
                    raise ValueError("complete sequences must include a termination token")
                prefix: Prefix = ()
                self._descendants[prefix].add(label)
                for token in path:
                    if prefix in self._terminal:
                        raise ValueError("complete sequences must be prefix-free")
                    try:
                        hash(token)
                    except TypeError as error:
                        raise TypeError("tokens must be hashable") from error
                    next_prefix = (*prefix, token)
                    if next_prefix not in self._children:
                        self._children[next_prefix] = {}
                        self._descendants[next_prefix] = set()
                    self._children[prefix][token] = next_prefix
                    self._descendants[next_prefix].add(label)
                    prefix = next_prefix
                if prefix in self._terminal:
                    raise ValueError("duplicate complete token sequences are not allowed")
                if self._children[prefix]:
                    raise ValueError("complete sequences must be prefix-free")
                self._terminal[prefix] = label
                paths.append(path)
            if not paths:
                raise ValueError("every semantic class needs at least one sequence")
            self._sequences[label] = tuple(paths)
        self._order = {prefix: index for index, prefix in enumerate(self._children)}
        self._conditionals: dict[Prefix, dict[Token, Fraction]] = {}
        for prefix, children in self._children.items():
            if not children:
                continue
            if prefix not in conditionals:
                raise ValueError(f"missing probability row for prefix {prefix!r}")
            row = {token: _probability(value) for token, value in conditionals[prefix].items()}
            if sum(row.values(), Fraction(0)) > 1:
                raise ValueError(f"probability row exceeds unit mass at prefix {prefix!r}")
            if any(token not in row for token in children):
                raise ValueError(f"missing trie-child probability at prefix {prefix!r}")
            self._conditionals[prefix] = row
        self._prefix_probabilities: dict[Prefix, Fraction] = {(): Fraction(1)}
        # Parents precede children by depth, independent of family insertion.
        for prefix in sorted(self._children, key=len):
            for token, child in self._children[prefix].items():
                self._prefix_probabilities[child] = (
                    self._prefix_probabilities[prefix] * self._conditionals[prefix][token]
                )

    def sequence_probability(self, sequence: Sequence[Token]) -> Fraction:
        path = tuple(sequence)
        if path not in self._terminal:
            raise ValueError("the sequence is not a declared complete candidate")
        return self._prefix_probabilities[path]

    def enumerate_masses(self) -> dict[str, Fraction]:
        """Independent of the audit frontier: sum all declared complete leaves."""
        return {
            label: sum(
                (self.sequence_probability(path) for path in paths), Fraction(0)
            )
            for label, paths in self._sequences.items()
        }

    def _snapshot(self, lower, frontier, expanded_prefix=None) -> BoundSnapshot:
        upper = dict(lower)
        ordered_frontier = tuple(sorted(frontier, key=self._order.__getitem__))
        for prefix in ordered_frontier:
            mass = self._prefix_probabilities[prefix]
            for label in self._descendants[prefix]:
                upper[label] += mass
        return BoundSnapshot(
            tuple((label, lower[label]) for label in self.classes),
            tuple((label, upper[label]) for label in self.classes),
            ordered_frontier,
            expanded_prefix,
        )

    def audit(self, *, max_expansions: int | None = None) -> CertificateAudit:
        """Audit largest-prefix-mass-first expansion of this already supplied table.

        A strict bound separation certifies the finite semantic MAP winner.
        Exact ties require exhausting the frontier and use declared class order.
        The trace's expansion count is an audit statistic, never GPU work saved.
        """
        if max_expansions is not None and (
            isinstance(max_expansions, bool)
            or not isinstance(max_expansions, int)
            or max_expansions < 0
        ):
            raise ValueError("max_expansions must be a nonnegative integer or None")
        lower = dict.fromkeys(self.classes, Fraction(0))
        frontier = {()}
        history = [self._snapshot(lower, frontier)]
        expansions = 0
        while True:
            snapshot = history[-1]
            upper = dict(snapshot.upper)
            if not frontier:
                winner = max(self.classes, key=lower.__getitem__)
                return CertificateAudit(winner, "exhaustive", expansions, tuple(history))
            for label in self.classes:
                if all(
                    lower[label] > upper[other]
                    for other in self.classes
                    if other != label
                ):
                    return CertificateAudit(label, "strict_bounds", expansions, tuple(history))
            if max_expansions is not None and expansions >= max_expansions:
                return CertificateAudit(None, "expansion_limit", expansions, tuple(history))
            prefix = max(
                frontier,
                key=lambda value: (
                    self._prefix_probabilities[value], -self._order[value]
                ),
            )
            frontier.remove(prefix)
            for child in self._children[prefix].values():
                if child in self._terminal:
                    label = self._terminal[child]
                    lower[label] += self._prefix_probabilities[child]
                else:
                    frontier.add(child)
            expansions += 1
            history.append(self._snapshot(lower, frontier, prefix))


def certify_semantic_map(
    candidates: Mapping[str, Iterable[Sequence[Token]]],
    conditionals: Mapping[Prefix, Mapping[Token, Probability]],
    *,
    max_expansions: int | None = None,
) -> CertificateAudit:
    """Convenience entry point for a pure finite-table audit."""
    return FiniteProbabilityTree(candidates, conditionals).audit(
        max_expansions=max_expansions
    )
