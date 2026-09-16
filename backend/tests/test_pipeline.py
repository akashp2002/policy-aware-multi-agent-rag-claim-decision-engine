"""Integration tests: hybrid retrieval and the full multi-agent pipeline.

These exercise the real pipeline (embedding model + BM25 + reranker), so
they take a few seconds each. They assert the core behaviours rather than
exact numbers, to stay robust to policy/retrieval tuning.
"""
from __future__ import annotations

import pytest

from src.evaluation.expected_outcomes import PUBLIC_EXPECTED
from src.models.schemas import ClaimCase, DecisionStatus

from conftest import load_case


def test_hybrid_search_returns_stable_evidence(search):
    hits = search.search("waiting period pre existing disease 48 months", top_k=5)
    assert len(hits) >= 1
    assert all(h.rank and h.rank >= 1 for h in hits)  # ranked
    assert all(h.score == h.score for h in hits)  # not NaN
    assert len({h.chunk_id for h in hits}) == len(hits)  # unique


def test_rerank_prefers_top_candidates(search):
    hits = search.search("room sub-limit one percent sum insured per day", top_k=4, with_rerank=True)
    assert all(h.rank is not None for h in hits)


@pytest.mark.parametrize(
    "case_id,expected",
    [(cid, exp) for cid, exp in PUBLIC_EXPECTED.items()],
)
def test_public_cases_match_ground_truth(orchestrator, case_id, expected):
    raw = load_case(case_id)
    result = orchestrator.analyze(ClaimCase(**raw))
    assert result.decision.value == expected, result.key_findings
    assert result.validation.status.value == "PASS"
    assert len(result.citations) >= 1


def test_needs_review_cases_have_low_confidence(orchestrator):
    for cid in ("PUB-006", "PUB-011"):
        raw = load_case(cid)
        result = orchestrator.analyze(ClaimCase(**raw))
        assert result.decision == DecisionStatus.NEEDS_REVIEW
        assert result.confidence < 0.6


def test_custom_cases_match_ground_truth(orchestrator):
    from src.evaluation.expected_outcomes import CUSTOM_EXPECTED

    for cid, expected in CUSTOM_EXPECTED.items():
        result = orchestrator.analyze(ClaimCase(**load_case(cid)))
        assert result.decision.value == expected