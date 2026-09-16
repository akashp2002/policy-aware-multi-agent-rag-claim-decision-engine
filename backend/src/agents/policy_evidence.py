"""Policy Evidence Agent.

Translates the investigation plan's dimensional questions into focused
retrieval queries, executes the hybrid search pipeline, and returns the
pooled evidence bundle. When the LLM provider is available, the agent
uses it to refine queries; otherwise a deterministic keyword-expansion
strategy is used.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.agents.base import BaseAgent
from src.models.schemas import (
    CaseState,
    ClaimCase,
    EvidenceState,
    InvestigationItem,
    PolicyEvidence,
)
from src.retrieval.hybrid_search import HybridSearch

logger = logging.getLogger(__name__)

# Deterministic query expansion: dimension -> base queries
_DIM_QUERIES: Dict[str, List[str]] = {
    "hospitalization_definition": [
        "Hospital definition hospitalization covered",
        "What is a hospital minimum requirements beds",
        "Inpatient day care domiciliary eligible hospital",
    ],
    "waiting_period": [
        "waiting period initial",
        "thirty days waiting period",
        "first year operation waiting period treatment",
    ],
    "specific_disease_waiting": [
        "first year operation of the insurance cover treatment of the following diseases",
        "one year waiting period list of diseases cataract hernia piles tonsils",
    ],
    "pre_existing_disease": [
        "pre-existing disease definition",
        "pre-existing disease waiting period duration",
        "continuous coverage prior insurer waiting period reduction",
    ],
    "domiciliary": [
        "Domiciliary Hospitalisation definition",
        "domiciliary treatment conditions",
        "domiciliary sub-limit amount payable",
    ],
    "day_care": [
        "day care procedure covered",
        "less than twenty four hours hospitalization",
        "day care surgery eye cataract",
    ],
    "sub_limit": [
        "room boarding nursing expense limit",
        "room category cap sub-limit",
        "sum insured room limit percentage",
    ],
    "sum_insured": [
        "sum insured maximum payable",
        "cumulative bonus increase sum insured",
        "cumulative bonus conditions percentage",
    ],
    "pre_post_hospitalization_window": [
        "pre-hospitalization days coverage",
        "post-hospitalization days coverage",
        "pre and post hospitalisation treatment expenses",
    ],
    "exclusion": [
        "exclusion list what is not covered",
        "experimental unproven treatment excluded",
        "cosmetic surgery excluded",
        "war terrorism nuclear excluded",
    ],
    "hospital_eligibility": [
        "hospital registered minimum criteria",
        "hospital definition registration requirements",
    ],
    "portability": [
        "portability provisions continuous coverage",
        "portability waiting period credit",
        "prior continuous insurance coverage waiting period reduction",
    ],
}

# LLM query refinement prompt
_LLM_QSYS = """You are a policy-evidence retrieval assistant. Given a list of policy investigation
dimensions and case facts, output a JSON array of exactly N search queries (one per dimension)
that would locate the most relevant text in an Indian health-insurance policy document. Each
query should be a concise keyword-style phrase of 6-12 words. Return ONLY the JSON array
(no explanation)."""

_LLM_QUSR = """Dimensions to investigate: {dims}
Case facts: {facts}
Output {n} queries as a JSON array of strings."""


class PolicyEvidenceAgent(BaseAgent):
    name = "PolicyEvidenceAgent"

    def __init__(self, search_engine: HybridSearch, llm_provider=None) -> None:
        super().__init__()
        self.search = search_engine
        self.llm = llm_provider

    # ------------------------------------------------------------------
    def run(self, case_state: CaseState, extra_queries: Optional[List[str]] = None) -> EvidenceState:
        """Build retrieval queries and execute hybrid search.

        Each investigation dimension gets its own dedicated retrieval
        pass so the per-dimension evidence is precise. A global pooled
        set is also produced for case-level coverage and validation.
        """
        queries = self._build_queries(case_state)
        if extra_queries:
            queries = list(dict.fromkeys(queries + list(extra_queries)))
        logger.info("Built %d queries for case %s", len(queries), case_state.case.case_id)

        # Per-dimension retrieval: precise evidence for each dimension.
        ev_by_dim: Dict[str, List[PolicyEvidence]] = {}
        pooled_map: Dict[str, PolicyEvidence] = {}
        for dim_item in case_state.plan.dimensions:
            dim = dim_item.dimension
            dim_queries = list(_DIM_QUERIES.get(dim, []))
            if extra_queries:
                dim_queries = dim_queries + [q for q in extra_queries if q]
            if not dim_queries:
                dim_queries = [dim.replace("_", " ")]
            candidates: Dict[str, PolicyEvidence] = {}
            for dq in dim_queries:
                for ev in self.search.search(dq, top_k=4):
                    prev = candidates.get(ev.chunk_id)
                    if prev is None or (ev.rerank_score or 0) > (prev.rerank_score or 0):
                        candidates[ev.chunk_id] = ev
            ranked = sorted(candidates.values(), key=lambda e: -(e.rerank_score or 0))
            ev_by_dim[dim] = ranked[:4]
            for ev in ev_by_dim[dim]:
                prev = pooled_map.get(ev.chunk_id)
                if prev is None or (ev.rerank_score or 0) > (prev.rerank_score or 0):
                    pooled_map[ev.chunk_id] = ev

        pooled = sorted(pooled_map.values(), key=lambda e: -(e.rerank_score or 0))

        return EvidenceState(
            plan=case_state.plan,
            queries=queries,
            evidence_by_dimension=ev_by_dim,
            all_evidence=pooled,
            retrieval_metadata={
                "num_queries": len(queries),
                "pooled_candidates": len(pooled),
                "dimensions_with_evidence": len([v for v in ev_by_dim.values() if v]),
                "dimensions_total": len(case_state.plan.dimensions),
            },
        )

    # ------------------------------------------------------------------
    def _build_queries(self, cs: CaseState) -> List[str]:
        if self.llm is not None and not self.llm.is_fallback:
            return self._llm_queries(cs)
        return self._deterministic_queries(cs)

    # ------------------------------------------------------------------
    def _deterministic_queries(self, cs: CaseState) -> List[str]:
        c = cs.case
        queries: List[str] = []

        for dim_item in cs.plan.dimensions:
            base = _DIM_QUERIES.get(dim_item.dimension, [])
            for q in base[:2]:  # up to 2 per dimension
                enriched = q
                diag = (c.treatment.diagnosis if c.treatment else "") or ""
                if "pre-existing" in q.lower() and "pre_existing" in dim_item.dimension.lower():
                    enriched += f" {diag[:40]}"
                queries.append(enriched)

        # Also add a general full-case query
        ttype = (c.treatment.type if c.treatment else "") or "hospitalization"
        diag = (c.treatment.diagnosis if c.treatment else "") or ""
        queries.append(f"{ttype} {diag} coverage scope covered expenses payable")

        return queries

    # ------------------------------------------------------------------
    def _llm_queries(self, cs: CaseState) -> List[str]:
        facts_str = json.dumps(cs.extracted_facts, default=str)[:2000]
        dims_str = ", ".join(cs.decision_dimensions)
        user = _LLM_QUSR.format(dims=dims_str, facts=facts_str, n=len(cs.plan.dimensions) + 1)
        try:
            res = self.llm.complete(_LLM_QSYS, user, max_tokens=500)
            arr = json.loads(res.text)
            if isinstance(arr, list) and len(arr) > 0:
                return [str(x) for x in arr]
        except Exception as e:
            logger.warning("LLM query refinement failed (%s); falling back to deterministic.", e)
        return self._deterministic_queries(cs)

    # ------------------------------------------------------------------
    def _find_evidence_for_dimension_all(
        self,
        plan: Any,
        pooled: List[PolicyEvidence],
        cs: CaseState,
    ) -> Dict[str, List[PolicyEvidence]]:
        """Helper: recompute per-dimension evidence for a fresh pool."""
        ev_by_dim: Dict[str, List[PolicyEvidence]] = {}
        for dim_item in plan.dimensions:
            ev_by_dim[dim_item.dimension] = self._find_evidence_for_dimension(
                dim_item.dimension, pooled, cs
            )
        return ev_by_dim

    # ------------------------------------------------------------------
    def _find_evidence_for_dimension(
        self,
        dim: str,
        pooled: List[PolicyEvidence],
        cs: CaseState,
    ) -> List[PolicyEvidence]:
        """Return the subset of pooled evidence relevant to the given dimension.

        Heuristic: match section/heading keywords against the dimension name,
        and check if any query keyword from _DIM_QUERIES appears in the
        evidence text.
        """
        dim_queries = _DIM_QUERIES.get(dim, [])
        dim_keywords = set()
        for q in dim_queries:
            dim_keywords.update(q.lower().split())

        # Also add keywords derived from the dimension name
        dim_keywords.update(dim.replace("_", " ").split())

        dim_keywords.discard("the")
        dim_keywords.discard("a")
        dim_keywords.discard("an")
        dim_keywords.discard("and")
        dim_keywords.discard("or")
        dim_keywords.discard("of")
        dim_keywords.discard("in")
        dim_keywords.discard("to")
        dim_keywords.discard("for")

        relevant: List[PolicyEvidence] = []
        for ev in pooled:
            text_lower = ev.text.lower()
            # check if any dimension keyword appears in the evidence text
            if any(kw in text_lower for kw in dim_keywords if len(kw) > 3):
                relevant.append(ev)
            # also match by section
            elif dim.lower().split("_")[0] in (ev.section or "").lower():
                relevant.append(ev)
        return relevant[:5]  # cap at 5 per dimension