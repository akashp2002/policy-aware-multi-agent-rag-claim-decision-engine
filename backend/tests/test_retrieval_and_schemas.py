"""Unit tests for policy ingestion, text matching, and schema contracts."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.text_match import best_matching, overlap_ratio, significant_tokens
from src.models.schemas import ClaimCase, DecisionStatus, Dimension
from src.retrieval.policy_ingestion import load_chunks


def test_policy_index_is_non_trivial(chunks):
    assert len(chunks) >= 40
    metas = [c.section for c in chunks]
    assert "WHAT WE EXCLUDE" in metas
    assert "DEFINITIONS" in metas
    assert all(c.text.strip() for c in chunks)


def test_sections_are_tagged(chunks):
    by_section = {}
    for c in chunks:
        by_section.setdefault(c.section, 0)
        by_section[c.section] += 1
    for name in ("DEFINITIONS", "STANDARD TERMS AND CONDITIONS", "WHAT WE EXCLUDE"):
        assert by_section.get(name, 0) >= 1


def test_significant_tokens_and_overlap():
    assert "the" not in significant_tokens("the cat")
    assert "cat" in significant_tokens("the cat")
    assert overlap_ratio("cat dog", "dog bird") == 0.5
    assert overlap_ratio("apple", "banana") == 0.0


def test_best_matching_filters_low_overlap():
    top = best_matching(
        "hospital eligibility minimum criteria",
        [("C-1", "room sub-limit applies per day"), ("C-2", "hospital must be registered")],
        min_overlap=0.3,
    )
    assert len(top) == 0


def test_claim_case_tolerates_unknown_extra_fields():
    case = ClaimCase(
        case_id="X-1",
        sum_insured_inr=100000,
        custom_future_field="ignored",
        nested={"a": 1},
    )
    assert case.sum_insured_inr == 100000


def test_claim_case_rejects_bad_types():
    with pytest.raises(ValidationError):
        ClaimCase(case_id="X-1", sum_insured_inr="ten")


def test_enum_members_present():
    names = {d.value for d in Dimension}
    for want in ("waiting_period", "pre_existing_disease", "exclusion", "sub_limit"):
        assert want in names
    assert DecisionStatus.NEEDS_REVIEW.value == "NEEDS_REVIEW"


def test_vendor_public_case_parses():
    # The exact vendor fixture must parse cleanly (regression guard).
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "data" / "test_cases" / "public_test_cases.json"
    data = json.load(open(path, encoding="utf-8"))
    for raw in data:
        case = ClaimCase(**raw)
        assert case.case_id