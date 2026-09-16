"""Focused tests for final-decision citation validation."""
from __future__ import annotations

from src.agents.validation import ValidationAgent
from src.models.schemas import (
    Citation,
    DecisionState,
    EvidenceState,
    FinalDecision,
    InvestigationPlan,
    PolicyEvidence,
    DecisionStatus,
)


def test_validation_rejects_unsupported_final_key_finding():
    evidence = EvidenceState(
        plan=InvestigationPlan(),
        all_evidence=[PolicyEvidence(
            chunk_id="C-1",
            page=1,
            section="SCOPE OF COVER",
            text="Hospitalization requires a minimum stay of 24 hours.",
        )],
    )
    decision = DecisionState(decision=FinalDecision(
        status=DecisionStatus.ADMISSIBLE,
        confidence=0.8,
        key_findings=["Cosmetic dental treatment is covered without limitation"],
        rationale="Decision ADMISSIBLE.",
    ))
    citations = [Citation(
        claim="Policy finding: hospitalization definition",
        chunk_id="C-1",
        page=1,
        section="SCOPE OF COVER",
    )]

    result = ValidationAgent().run(evidence, decision, citations)

    assert result.status.value == "FAIL"
    assert any("Final key finding" in failure.claim for failure in result.unsupported_claims)