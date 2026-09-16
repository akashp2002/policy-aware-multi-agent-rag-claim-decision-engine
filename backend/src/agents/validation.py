"""Validation Agent.

Verifies that every material decision statement is supported by the
retrieved policy evidence. Each citation is cross-checked against the
text of its source chunk.

A citation passes when the cited chunk shares enough significant
(shorter, content-bearing) terms with the claimed statement. Claims that
fail validation cause the orchestrator to either retry retrieval with
enriched queries or downgrade the decision to NEEDS_REVIEW, so the
system never emits a confidently unsupported decision.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List

from src.agents.base import BaseAgent
from src.agents.text_match import overlap_ratio
from src.models.schemas import (
    Citation,
    DecisionState,
    EvidenceState,
    ValidationFailure,
    ValidationResult,
    ValidationStatus,
)

logger = logging.getLogger(__name__)

_PREFIX_RX = re.compile(r"^Policy finding(?: \((?:not applicable|unknown[^)]*)\))?:\s*", re.I)
_RATIONALE_LABELS = ("Blocking:", "Requires review:", "Partly payable:", "Limits applied:")


class ValidationAgent(BaseAgent):
    name = "ValidationAgent"

    # Minimum share of a claim's significant tokens that must appear in the
    # cited chunk for the citation to be considered supported.
    MIN_OVERLAP = 0.30

    def run(self, evidence_state: EvidenceState, decision_state: DecisionState,
            citations: List[Citation]) -> ValidationResult:
        chunk_texts: Dict[str, str] = {e.chunk_id: e.text for e in evidence_state.all_evidence}
        failures: List[ValidationFailure] = []

        if not citations:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                unsupported_claims=[ValidationFailure(
                    claim="No citations produced",
                    reason="A decision with zero citations cannot be validated",
                )],
            )

        for cit in citations:
            if cit.chunk_id not in chunk_texts:
                failures.append(ValidationFailure(
                    claim=cit.claim,
                    reason=f"Cited chunk {cit.chunk_id} was not retrieved as evidence",
                ))
                continue

            claim_text = _PREFIX_RX.sub("", cit.claim)
            ov = overlap_ratio(claim_text, chunk_texts[cit.chunk_id])
            if ov < self.MIN_OVERLAP:
                failures.append(ValidationFailure(
                    claim=cit.claim,
                    reason=f"Only {ov:.0%} of claim terms found in chunk {cit.chunk_id} "
                           f"(threshold {self.MIN_OVERLAP:.0%})",
                ))

        cited_dimensions = {cit.dimension for cit in citations if cit.dimension}
        for finding in decision_state.decision.key_findings:
            dimension = finding.split(":", 1)[0].strip() if ":" in finding else ""
            if dimension not in cited_dimensions:
                failures.append(ValidationFailure(
                    claim=f"Final key finding: {finding}",
                    reason=f"No citation is associated with decision dimension {dimension!r}",
                ))

        for clause in self._material_rationale_clauses(decision_state.decision.rationale):
            best_overlap = max(
                (
                    overlap_ratio(clause, finding)
                    for finding in (
                        decision_state.decision.key_findings
                        + decision_state.decision.missing_evidence
                    )
                ),
                default=0.0,
            )
            if best_overlap < self.MIN_OVERLAP:
                failures.append(ValidationFailure(
                    claim=f"Final rationale: {clause}",
                    reason=(
                        f"Only {best_overlap:.0%} of rationale terms were found in "
                        f"the final key findings (threshold {self.MIN_OVERLAP:.0%})"
                    ),
                ))

        status = ValidationStatus.PASS if not failures else ValidationStatus.FAIL
        return ValidationResult(status=status, unsupported_claims=failures)

    @staticmethod
    def _material_rationale_clauses(rationale: str) -> List[str]:
        """Extract policy-bearing rationale clauses, excluding status boilerplate."""
        clauses: List[str] = []
        for sentence in rationale.split("."):
            sentence = sentence.strip()
            for label in _RATIONALE_LABELS:
                if sentence.startswith(label):
                    clause = sentence[len(label):].strip()
                    if clause:
                        clauses.append(clause)
                    break
        return clauses