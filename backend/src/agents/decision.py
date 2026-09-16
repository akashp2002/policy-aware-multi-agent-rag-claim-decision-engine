"""Decision Agent.

Synthesizes the specialist findings into the final structured decision.

The agent classifies each finding as:
  * blocking  -> a policy exclusion / unsatisfied waiting period applies
  * review    -> a required condition cannot be established (abstain)
  * partial   -> part of the claim (e.g. out-of-window expenses) is not payable
  * limiting  -> an applicable cap/sub-limit reduces the payable amount
  * clean     -> no material issue found

The resulting DecisionStatus follows directly from those classes,
which keeps the mapping auditable and avoids fragile score arithmetic.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from src.agents.base import BaseAgent
from src.models.schemas import (
    CaseState,
    CoverageFindings,
    DecisionState,
    DecisionStatus,
    EvidenceState,
    FinalDecision,
)

logger = logging.getLogger(__name__)

_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

# Documents that materially affect whether a claim can be decided.
_CRITICAL_DOCS = ["itemized_bill"]


class DecisionAgent(BaseAgent):
    name = "DecisionAgent"

    def run(
        self,
        case_state: CaseState,
        evidence_state: EvidenceState,
        coverage_findings: CoverageFindings,
    ) -> DecisionState:
        facts = case_state.extracted_facts
        findings = coverage_findings.findings

        blocking, review, partial, limiting = self._classify(findings)
        doc_gaps = self._document_gaps(case_state)
        if doc_gaps:
            review.extend(doc_gaps)

        status = self._assign_status(blocking, review, partial, limiting)
        limits = coverage_findings.applicable_limits
        key_findings = self._key_findings(findings)
        missing = self._missing_evidence(case_state, evidence_state, review, doc_gaps)
        confidence = self._confidence(status, blocking, review, partial, evidence_state)
        rationale = self._rationale(status, blocking, review, partial, limiting)

        decision = FinalDecision(
            status=status,
            confidence=round(confidence, 2),
            key_findings=key_findings,
            applicable_limits=limits,
            missing_evidence=missing,
            rationale=rationale,
        )
        return DecisionState(decision=decision)

    # ------------------------------------------------------------------
    def _classify(self, findings) -> Tuple[List[str], List[str], List[str], List[str]]:
        blocking: List[str] = []
        review: List[str] = []
        partial: List[str] = []
        limiting: List[str] = []

        for f in findings:
            a = f.assessment.lower()
            dim = f.dimension

            # Blocking conditions (explicit NOT ADMISSIBLE)
            if f.status == "not_applicable" and "not admissible" in a:
                blocking.append(f"{dim}: {f.assessment}")
                continue
            if f.status == "not_applicable":
                blocking.append(f"{dim}: {f.assessment}")
                continue

            # Abstain conditions
            if f.status == "unknown" or _INSUFFICIENT.lower() in a:
                review.append(f"{dim}: {f.assessment}")
                continue

            # Partial (out-of-window expenses)
            if "outside" in a and "window" in a:
                partial.append(f"{dim}: {f.assessment}")
                continue

            # Limiting (caps / sub-limits)
            if dim in ("sub_limit", "domiciliary", "ambulance") and f.status == "applicable":
                limiting.append(f"{dim}: {f.assessment}")
                continue

        return blocking, review, partial, limiting

    # ------------------------------------------------------------------
    def _document_gaps(self, cs: CaseState) -> List[str]:
        c = cs.case
        docs = set(d.lower() for d in (c.documents or []))
        gaps: List[str] = []
        for need in _CRITICAL_DOCS:
            if need not in docs:
                gaps.append(f"missing required document: {need}")
        # medical necessity unknown + minimal documents => review
        evc = c.evidence_context
        if evc is not None and evc.medical_necessity_confirmed is None:
            if len(docs) <= 2:
                gaps.append("medical necessity not confirmed")
        return gaps

    # ------------------------------------------------------------------
    def _assign_status(self, blocking, review, partial, limiting) -> DecisionStatus:
        if blocking:
            return DecisionStatus.NOT_ADMISSIBLE
        if review:
            return DecisionStatus.NEEDS_REVIEW
        if partial:
            return DecisionStatus.PARTIALLY_ADMISSIBLE
        if limiting:
            return DecisionStatus.ADMISSIBLE_WITH_LIMITS
        return DecisionStatus.ADMISSIBLE

    # ------------------------------------------------------------------
    def _confidence(self, status, blocking, review, partial, es: EvidenceState) -> float:
        evidence_bonus = min(len(es.all_evidence) / 12.0, 1.0) * 0.12
        base = {
            DecisionStatus.NOT_ADMISSIBLE: 0.80,
            DecisionStatus.ADMISSIBLE: 0.85,
            DecisionStatus.ADMISSIBLE_WITH_LIMITS: 0.78,
            DecisionStatus.PARTIALLY_ADMISSIBLE: 0.60,
            DecisionStatus.NEEDS_REVIEW: 0.35,
        }[status]
        return min(base + evidence_bonus, 0.97)

    # ------------------------------------------------------------------
    def _key_findings(self, findings) -> List[str]:
        out = []
        for f in findings:
            out.append(f"{f.dimension}: {f.assessment[:220]}")
        return out[:12]

    # ------------------------------------------------------------------
    def _missing_evidence(self, cs: CaseState, es: EvidenceState,
                          review: List[str], doc_gaps: List[str]) -> List[str]:
        missing = list(cs.missing_fields)
        missing.extend(doc_gaps)
        for r in review:
            missing.append(r)
        # de-dup, keep order
        seen = set()
        out = []
        for m in missing:
            if m and m not in seen:
                seen.add(m)
                out.append(m)
        return out

    # ------------------------------------------------------------------
    def _rationale(self, status, blocking, review, partial, limiting) -> str:
        parts = [f"Decision {status.value}."]
        if blocking:
            parts.append("Blocking: " + "; ".join(b.split(": ", 1)[-1][:140] for b in blocking[:3]) + ".")
        if review:
            parts.append("Requires review: " + "; ".join(r.split(": ", 1)[-1][:140] for r in review[:3]) + ".")
        if partial:
            parts.append("Partly payable: " + "; ".join(p.split(": ", 1)[-1][:140] for p in partial[:3]) + ".")
        if limiting:
            parts.append("Limits applied: " + "; ".join(l.split(": ", 1)[-1][:120] for l in limiting[:3]) + ".")
        if status == DecisionStatus.NEEDS_REVIEW:
            parts.append("Abstaining because a required condition or evidence could not be established.")
        return " ".join(parts)