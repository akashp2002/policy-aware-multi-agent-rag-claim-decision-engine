"""AgentOrchestrator – runs the five-agent pipeline on a ClaimCase.

Flow:
  ClaimCase (API)
    -> CaseAnalysisAgent      -> CaseState (plan + facts)
    -> PolicyEvidenceAgent    -> EvidenceState (queries + chunks)
    -> CoverageExclusionAgent -> CoverageFindings
    -> DecisionAgent          -> DecisionState
    -> ValidationAgent        -> ValidationResult + citations
    -> DecisionResponse

Retry / validation loop:
  If the Validation Agent reports FAIL, we do at most N re-passes with
  enriched queries (adding dimension keywords to retrieval) before
  returning a NEEDS_REVIEW decision rather than an unsupported one.

All agent outputs are typed Pydantic objects (structured state), and
every step is recorded to the `trace` list of AgentTraceEntry.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from src.agents.case_analyst import CaseAnalysisAgent
from src.agents.coverage_exclusion import CoverageExclusionAgent
from src.agents.decision import DecisionAgent
from src.agents.policy_evidence import PolicyEvidenceAgent
from src.agents.validation import ValidationAgent
from src.llm.provider import LLMProvider
from src.models.schemas import (
    AgentTraceEntry,
    CaseState,
    Citation,
    CoverageFindings,
    DecisionResponse,
    DecisionState,
    DecisionStatus,
    EvidenceState,
    ValidationResult,
    ValidationStatus,
)
from src.retrieval.hybrid_search import HybridSearch

logger = logging.getLogger(__name__)

MAX_VALIDATION_PASSES = 2


class AgentOrchestrator:
    def __init__(
        self,
        search_engine: HybridSearch,
        llm_provider: Optional[LLMProvider] = None,
        additional_queries: Optional[List[str]] = None,
    ) -> None:
        self.search = search_engine
        self.llm = llm_provider or LLMProvider(provider=os_environ("CLAIM_LLM_PROVIDER", "auto"))
        self.extra_queries = additional_queries or []

        self.case_agent = CaseAnalysisAgent()
        self.evidence_agent = PolicyEvidenceAgent(search_engine, self.llm)
        self.coverage_agent = CoverageExclusionAgent(self.llm)
        self.decision_agent = DecisionAgent()
        self.validation_agent = ValidationAgent()
        self.trace: List[AgentTraceEntry] = []

    # ------------------------------------------------------------------
    def analyze(self, claim_case) -> DecisionResponse:  # noqa: F821
        self.trace = []

        # 1) Case Analysis
        case_state, t1 = self.case_agent.run_with_trace(
            claim_case,
            input_summary=f"Claim {claim_case.case_id} plan extraction",
        )
        self.trace.append(t1)
        logger.info("CaseAnalysis done: %d dimensions", len(case_state.plan.dimensions))

        # 2) Policy Evidence
        evidence_state, t2 = self.evidence_agent.run_with_trace(
            case_state,
            input_summary=f"{len(case_state.plan.dimensions)} dimensional queries",
        )
        self.trace.append(t2)
        logger.info("PolicyEvidence done: %d pooled chunks", len(evidence_state.all_evidence))

        # 3) Coverage & Exclusion
        coverage, t3 = self.coverage_agent.run_with_trace(
            case_state, evidence_state,
            input_summary=f"Evaluate {len(evidence_state.evidence_by_dimension)} dimensions",
        )
        self.trace.append(t3)

        # 4) Decision
        decision_state, t4 = self.decision_agent.run_with_trace(
            case_state, evidence_state, coverage,
            input_summary="Synthesize findings into decision",
        )
        self.trace.append(t4)

        # Build real citations: reuse decision agent's citation builder via coverage
        citations = self._recover_citations(coverage, evidence_state)
        decision_state.decision.rationale = (
            decision_state.decision.rationale
            + f" Citations: {len(citations)}"
        )

        # 5) Validation
        validation, t5 = self.validation_agent.run_with_trace(
            evidence_state, decision_state, citations,
            input_summary=f"Validate {len(citations)} citations",
        )
        self.trace.append(t5)

        # Retry loop: if validation FAIL, enrich queries and re-run coverage->decision
        passes = 0
        while validation.status == ValidationStatus.FAIL and passes < MAX_VALIDATION_PASSES:
            passes += 1
            logger.warning("Validation FAIL (pass %d); enriching retrieval", passes)
            extra = self._derive_extra_queries(validation)
            mid_state = self._run_enriched_pass(case_state, evidence_state, extra)
            if mid_state is None:
                break
            case_state, evidence_state, coverage, decision_state, citations = mid_state
            validation, t5 = self.validation_agent.run_with_trace(
                evidence_state, decision_state, citations,
                input_summary=f"Validate {len(citations)} citations (post-retry)",
            )
            self.trace.append(t5)

        # If validation still fails, downgrade to NEEDS_REVIEW rather than emit unsupported claims.
        if validation.status == ValidationStatus.FAIL:
            decision_state.decision.status = DecisionStatus.NEEDS_REVIEW
            decision_state.decision.confidence = min(decision_state.decision.confidence, 0.35)
            decision_state.decision.missing_evidence.append(
                "Validation could not confirm supporting policy evidence for all material statements"
            )

        return DecisionResponse(
            case_id=claim_case.case_id,
            decision=decision_state.decision.status,
            confidence=round(decision_state.decision.confidence, 2),
            key_findings=decision_state.decision.key_findings,
            applicable_limits=decision_state.decision.applicable_limits,
            missing_evidence=decision_state.decision.missing_evidence,
            citations=self._dedupe_citations(citations),
            validation=validation,
            trace=self.trace,
            policy={
                "model": self.llm.model_id,
                "provider": self.llm.provider,
                "chunks_indexed": len(self.search.chunks),
                "retrieval_metadata": evidence_state.retrieval_metadata,
            },
        )

    # ------------------------------------------------------------------
    def _run_enriched_pass(self, case_state, evidence_state, extra_queries):
        """Re-run the retrieval portion with additional queries."""
        es = self.evidence_agent.run(case_state, extra_queries=list(extra_queries))
        cs = case_state
        coverage = self.coverage_agent.run(cs, es)
        ds = self.decision_agent.run(cs, es, coverage)
        citations = self._recover_citations(coverage, es)
        return cs, es, coverage, ds, citations

    def _derive_extra_queries(self, validation) -> List[str]:
        out: List[str] = []
        for fail in validation.unsupported_claims:
            words = fail.claim.split()
            keywords = [w for w in words if len(w) > 4][:5]
            if keywords:
                out.append(" ".join(keywords))
        return out[:5]

    def _recover_citations(self, coverage: CoverageFindings, evidence_state: EvidenceState) -> List[Citation]:
        chunk_map = {e.chunk_id: e for e in evidence_state.all_evidence}
        citations: List[Citation] = []
        seen = set()
        for f in coverage.findings:
            if f.status == "not_applicable":
                claim = f"Policy finding (not applicable): {f.finding}"
            elif f.status == "unknown":
                claim = f"Policy finding (unknown / needs review): {f.finding}"
            else:
                claim = f"Policy finding: {f.finding}"
            if claim in seen:
                continue
            seen.add(claim)
            chunk_ids = list(f.policy_chunk_ids or [])[:1]
            if not chunk_ids:
                dimension_evidence = evidence_state.evidence_by_dimension.get(f.dimension, [])
                fallback = dimension_evidence or evidence_state.all_evidence
                chunk_ids = [fallback[0].chunk_id] if fallback else []
            for cid in chunk_ids:
                chunk = chunk_map.get(cid)
                if chunk is None:
                    continue
                citations.append(Citation(
                    claim=claim,
                    source="policy.pdf",
                    page=chunk.page,
                    section=chunk.section,
                    chunk_id=chunk.chunk_id,
                    dimension=f.dimension,
                    excerpt=chunk.text,
                ))
        # Keep one or more citations available for every bounded final finding
        # so validation can verify the complete response, not only its first
        # few dimensions.
        return citations[:12]

    def _dedupe_citations(self, citations: List[Citation]) -> List[Citation]:
        seen = set()
        out = []
        for c in citations:
            key = (c.claim, c.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out


def os_environ(key: str, default: str = "") -> str:
    import os
    return os.environ.get(key, default)