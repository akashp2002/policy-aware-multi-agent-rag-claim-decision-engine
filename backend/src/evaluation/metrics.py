"""Metric helpers for the evaluation harness."""
from __future__ import annotations

from typing import Iterable, List, Sequence


def decision_accuracy(expected: Sequence[str], predicted: Sequence[str]) -> float:
    if not expected:
        return 0.0
    hits = sum(1 for e, p in zip(expected, predicted) if e == p)
    return hits / len(expected)


def is_correct(expected: str, predicted: str) -> bool:
    return expected == predicted


def citation_hit_rate(expected_chunk_ids: Iterable[str], cited_chunk_ids: Iterable[str]) -> float:
    """Fraction of expected supporting chunks that were cited.

    A case is only meaningful when it declares expected chunk ids; an empty
    expectation returns 1.0 (nothing to miss) so it does not distort averages.
    """
    expected = set(expected_chunk_ids)
    if not expected:
        return 1.0
    cited = set(cited_chunk_ids)
    return len(expected & cited) / len(expected)


def validation_pass_rate(statuses: Sequence[str]) -> float:
    if not statuses:
        return 0.0
    return sum(1 for s in statuses if s == "PASS") / len(statuses)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def confusion_matrix(expected: Sequence[str], predicted: Sequence[str]) -> dict:
    labels = sorted(set(expected) | set(predicted))
    matrix = {e: {p: 0 for p in labels} for e in labels}
    for e, p in zip(expected, predicted):
        matrix[e][p] += 1
    return matrix
