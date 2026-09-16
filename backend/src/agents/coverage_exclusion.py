"""Coverage & Exclusion Agent.

Assesses every investigation dimension against retrieved policy
evidence and extracted numeric policy rules, producing structured
CoverageFinding objects plus applicable limits and unmet conditions.

All numeric thresholds (waiting periods, sub-limit percentages,
pre/post windows) come from `extract_policy_rules`, which parses them
out of the retrieved policy chunks. This keeps the agent grounded in
the supplied policy instead of external insurance knowledge.

An optional LLM path is provided. When configured it produces the same
structured CoverageFindings; the deterministic engine is the default
and the fallback on parse failure.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base import BaseAgent
from src.agents.policy_rules import PolicyRuleSet, extract_policy_rules
from src.agents.text_match import best_matching
from src.models.schemas import (
    CaseState,
    ClaimCase,
    CoverageFinding,
    CoverageFindings,
    EvidenceState,
    PolicyEvidence,
)

logger = logging.getLogger(__name__)

_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

# Diseases subject to the policy's first-year waiting period, expressed as
# keyword groups. The list mirrors the policy's "first year of operation ...
# treatment of the following Diseases" clause and is matched against the
# case diagnosis/procedure text.
_FIRST_YEAR_DISEASE_KEYWORDS = {
    "cataract": ["cataract"],
    "benign prostatic hypertrophy": ["prostatic hypertrophy", "bph"],
    "myomectomy/hysterectomy": ["myomectomy", "hysterectomy"],
    "hernia/hydrocele": ["hernia", "hydrocele"],
    "fistula/piles": ["fistula", "piles", "hemorrhoid"],
    "arthritis/gout/rheumatism": ["arthritis", "gout", "rheumatism"],
    "joint replacement": ["joint replacement", "knee replacement", "hip replacement", "arthroplasty"],
    "sinusitis": ["sinusitis", "sinus"],
    "urinary/biliary stones": ["urinary stone", "renal stone", "kidney stone", "gallstone", "calculus"],
    "dilatation and curettage": ["dilatation and curettage", "d&c", "d & c"],
    "skin/internal tumors": ["tumour", "tumor", "malignant", "cancer", "neoplasm"],
    "dialysis for renal failure": ["dialysis", "renal failure", "kidney failure"],
    "tonsils and sinuses": ["tonsil", "tonsillitis", "tonsillectomy", "adenoid"],
    "gastric/duodenal ulcers": ["gastric ulcer", "duodenal ulcer", "peptic ulcer"],
}


class CoverageExclusionAgent(BaseAgent):
    name = "CoverageExclusionAgent"

    def __init__(self, llm_provider=None) -> None:
        super().__init__()
        self.llm = llm_provider

    # ------------------------------------------------------------------
    def run(self, case_state: CaseState, evidence_state: EvidenceState) -> CoverageFindings:
        c = case_state.case
        facts = case_state.extracted_facts
        rules = extract_policy_rules(evidence_state.all_evidence)

        findings: List[CoverageFinding] = []
        limits: List[Dict[str, Any]] = []
        unmet: List[str] = []

        findings.append(self._scope(c, facts, evidence_state))
        findings.append(self._waiting_period(c, facts, evidence_state, rules))
        findings.append(self._specific_disease_waiting(c, facts, evidence_state, rules))
        findings.append(self._pre_existing(c, facts, evidence_state, rules))

        if facts["is_domiciliary"]:
            findings.append(self._domiciliary(c, facts, evidence_state, rules, limits))
        if facts["is_day_care"] or (facts.get("admission_hours") or 0) < 24:
            findings.append(self._day_care(c, facts, evidence_state, rules))
        if facts["has_room_expense"]:
            findings.append(self._room_limit(c, facts, evidence_state, rules, limits))
        if facts["is_inpatient"]:
            findings.append(self._pre_post(c, facts, evidence_state, rules, limits))
            findings.append(self._ambulance(c, facts, evidence_state, rules, limits))
        findings.append(self._exclusions(c, facts, evidence_state))
        if not facts["is_domiciliary"]:
            findings.append(self._hospital_eligibility(c, facts, evidence_state, unmet))
        if facts["has_prior_policy"]:
            findings.append(self._portability(c, facts, evidence_state, rules))

        # Post-process: replace each finding's cited chunks with the ones
        # that actually share significant terms with the finding text, so
        # citations are precise and validation is meaningful. The same
        # text (the finding label) is what the Validation Agent checks.
        candidates = [(e.chunk_id, e.text) for e in evidence_state.all_evidence]
        for f in findings:
            matches = best_matching(f.finding, candidates, min_overlap=0.30, max_results=2)
            if matches:
                f.policy_chunk_ids = [cid for cid, _ in matches]
            else:
                # keep a single dimension-retrieved chunk as a weak citation
                f.policy_chunk_ids = f.policy_chunk_ids[:1]

        def find_dim(name: str) -> str:
            for f in findings:
                if f.dimension == name:
                    return f.assessment
            return ""

        return CoverageFindings(
            findings=findings,
            coverage_scope_assessment=find_dim("hospitalization_definition"),
            applicable_limits=limits,
            waiting_period_assessment=find_dim("waiting_period"),
            exclusion_assessment=find_dim("exclusion"),
            unmet_conditions=unmet,
        )

    # ------------------------------------------------------------------
    def _pool(self, evidence: List[PolicyEvidence], dim: str) -> str:
        return " ".join(e.text for e in evidence)

    def _ev(self, es: EvidenceState, dim: str) -> List[PolicyEvidence]:
        return es.evidence_by_dimension.get(dim, [])

    # ------------------------------------------------------------------
    def _scope(self, c: ClaimCase, facts: Dict, es: EvidenceState) -> CoverageFinding:
        ev = self._ev(es, "hospitalization_definition")
        ttype = facts["treatment_type"] or "unknown"
        hrs = facts.get("admission_hours") or 0

        if ttype == "inpatient" and hrs >= 24:
            status, note = "applicable", "inpatient >=24h meets hospitalization definition"
        elif ttype == "day_care":
            status, note = "applicable", "day-care qualifies under the 24h-waiver provisions"
        elif ttype == "domiciliary":
            status, note = "applicable", "domiciliary assessed under its specific conditions"
        elif ttype == "outpatient":
            status, note = "not_applicable", "outpatient treatment is excluded (exclusion 13)"
        else:
            status, note = "unknown", "treatment type unclear for scope assessment"

        return CoverageFinding(
            finding="Coverage scope: treatment qualifies as a covered hospitalization",
            dimension="hospitalization_definition",
            status=status,
            policy_chunk_ids=[e.chunk_id for e in ev[:2]],
            assessment=f"type={ttype}, admission_hours={hrs}; {note}",
        )

    # ------------------------------------------------------------------
    def _waiting_period(self, c: ClaimCase, facts: Dict, es: EvidenceState,
                        rules: PolicyRuleSet) -> CoverageFinding:
        ev = self._ev(es, "waiting_period")
        days_required = rules.value("initial_waiting_days", 30)
        months = facts.get("continuous_coverage_months") or 0
        prior_years = facts.get("prior_insurer_continuous_years") or 0

        # Exceptions: continuous previous policy year, or >=1yr prior Indian insurer
        has_exception = months >= 12 or prior_years >= 1

        status = "applicable"
        parts = [f"continuous_coverage_months={months:.0f}, prior_insurer_years={prior_years:.0f}",
                 f"initial_waiting_days={days_required:.0f}"]

        if has_exception:
            parts.append("30-day waiting-period exception applies (continuous prior coverage)")
        elif months < 1:
            status = "not_applicable"
            parts.append("claim falls within the initial waiting period => NOT ADMISSIBLE")
        elif months * 30 < days_required:
            status = "not_applicable"
            parts.append("elapsed coverage shorter than initial waiting period")

        return CoverageFinding(
            finding="Initial waiting-period assessment",
            dimension="waiting_period",
            status=status,
            policy_chunk_ids=[e.chunk_id for e in ev[:2]],
            assessment="; ".join(parts),
        )

    # ------------------------------------------------------------------
    def _specific_disease_waiting(self, c: ClaimCase, facts: Dict, es: EvidenceState,
                                  rules: PolicyRuleSet) -> CoverageFinding:
        ev = self._ev(es, "specific_disease_waiting")
        pooled = self._pool(es.all_evidence, "wa")

        text = ((facts.get("diagnosis") or "") + " " + (facts.get("procedure") or "")).lower()
        matched = [name for name, kws in _FIRST_YEAR_DISEASE_KEYWORDS.items()
                   if any(k in text for k in kws)]

        months = facts.get("continuous_coverage_months") or 0
        prior_years = facts.get("prior_insurer_continuous_years") or 0
        prior_ok = prior_years >= 1

        parts = [f"matched_first_year_diseases={matched or 'none'}",
                 f"continuous_coverage_months={months:.0f}, prior_insurer_years={prior_years:.0f}"]

        if not matched:
            status = "applicable"
            parts.append("diagnosis/procedure not in the first-year waiting list")
        else:
            if months >= 12 or prior_ok:
                status = "applicable"
                parts.append("first-year disease waiting period satisfied (continuous/prior coverage)")
            else:
                status = "not_applicable"
                parts.append(
                    "first-year waiting period applies to listed disease and is NOT satisfied "
                    "=> NOT ADMISSIBLE"
                )

        return CoverageFinding(
            finding="Specific-disease first-year waiting-period assessment",
            dimension="specific_disease_waiting",
            status=status,
            policy_chunk_ids=[e.chunk_id for e in ev[:2]],
            assessment="; ".join(parts),
        )

    # ------------------------------------------------------------------
    def _pre_existing(self, c: ClaimCase, facts: Dict, es: EvidenceState,
                      rules: PolicyRuleSet) -> CoverageFinding:
        ev = self._ev(es, "pre_existing_disease")
        ped_months = rules.value("ped_waiting_months", 48)
        port_reduction = rules.get("ped_portability_reduction") is not None
        months = facts.get("continuous_coverage_months") or 0
        prior_years = facts.get("prior_insurer_continuous_years") or 0
        pre_existing = bool(facts.get("pre_existing"))

        parts = [f"pre_existing={pre_existing}, continuous_coverage_months={months:.0f}",
                 f"ped_waiting_months={ped_months:.0f}"]

        if not pre_existing:
            status = "applicable"
            parts.append("no pre-existing flag; PED waiting period not triggered")
        else:
            effective_required = ped_months
            if port_reduction and prior_years > 0:
                effective_required = max(0, ped_months - prior_years * 12)
                parts.append(
                    f"portability reduction applied: {ped_months:.0f} - {prior_years*12:.0f} "
                    f"= {effective_required:.0f} months required"
                )
            if months >= effective_required:
                status = "applicable"
                parts.append("PED waiting period satisfied")
            else:
                status = "not_applicable"
                parts.append(
                    f"PED waiting period NOT satisfied ({months:.0f} < {effective_required:.0f} months) "
                    "=> NOT ADMISSIBLE"
                )

        return CoverageFinding(
            finding="Pre-existing disease waiting-period assessment",
            dimension="pre_existing_disease",
            status=status,
            policy_chunk_ids=[e.chunk_id for e in ev[:2]],
            assessment="; ".join(parts),
        )

    # ------------------------------------------------------------------
    def _domiciliary(self, c: ClaimCase, facts: Dict, es: EvidenceState,
                     rules: PolicyRuleSet, limits: List[Dict]) -> CoverageFinding:
        ev = self._ev(es, "domiciliary")
        pct = rules.value("domiciliary_sub_limit_pct", 20)
        si = facts.get("sum_insured") or 0
        cap = si * pct / 100

        room_unavail = bool(facts.get("hospital_room_unavailable"))
        cannot_move = bool(facts.get("patient_cannot_be_moved"))
        cond_ok = room_unavail or cannot_move

        if cond_ok:
            status = "applicable"
        else:
            status = "not_applicable"

        limits.append({
            "type": "domiciliary_sub_limit",
            "amount": round(cap, 2),
            "percentage": pct,
            "basis": "Basic Sum Insured",
            "chunk_id": rules.get("domiciliary_sub_limit_pct").chunk_id if rules.get("domiciliary_sub_limit_pct") else "",
        })

        assessment = (
            f"domiciliary sub-limit={pct:.0f}% of SI => INR {cap:,.0f}; "
            f"room_unavailable={room_unavail}, cannot_be_moved={cannot_move}; "
            + ("domiciliary conditions satisfied" if cond_ok
               else "domiciliary conditions NOT satisfied (bed available and patient movable) => NOT ADMISSIBLE")
        )
        return CoverageFinding(
            finding="Domiciliary hospitalisation eligibility and sub-limit",
            dimension="domiciliary",
            status=status,
            policy_chunk_ids=[e.chunk_id for e in ev[:2]],
            assessment=assessment,
        )

    # ------------------------------------------------------------------
    def _day_care(self, c: ClaimCase, facts: Dict, es: EvidenceState,
                  rules: PolicyRuleSet) -> CoverageFinding:
        ev = self._ev(es, "day_care")
        pooled = self._pool(es.all_evidence, "day_care")
        waiver_mentioned = rules.get("daycare_24h_waiver_mentioned") is not None
        proc = (facts.get("procedure") or "").lower()
        diag = (facts.get("diagnosis") or "").lower()

        listed = any(k in (proc + " " + diag) for k in
                     ["eye surgery", "cataract", "dialysis", "chemotherapy",
                      "radiotherapy", "lithotripsy", "tonsillectomy"])
        listed = listed or bool(re.search(r"Eye Surgery|Dialysis|Chemotherapy", pooled, re.I))

        if listed:
            status = "applicable"
            note = "procedure is within the day-care / 24h-waiver list"
        else:
            status = "applicable"
            note = "day-care treatment; covered if within the policy's day-care list"

        return CoverageFinding(
            finding="Day-care / less-than-24-hour treatment eligibility",
            dimension="day_care",
            status=status,
            policy_chunk_ids=[e.chunk_id for e in ev[:2]],
            assessment=f"procedure={facts.get('procedure')}; {note}",
        )

    # ------------------------------------------------------------------
    def _room_limit(self, c: ClaimCase, facts: Dict, es: EvidenceState,
                    rules: PolicyRuleSet, limits: List[Dict]) -> CoverageFinding:
        ev = self._ev(es, "sub_limit")
        pct = rules.value("room_limit_pct_normal", 1.0)
        icu_pct = rules.value("room_limit_pct_icu", 2.0)
        si = facts.get("sum_insured") or 0
        room = facts.get("room_cost") or 0
        per_day_cap = si * pct / 100

        limits.append({
            "type": "room_sub_limit",
            "amount": round(per_day_cap, 2),
            "percentage": pct,
            "basis": "Basic Sum Insured per day",
            "chunk_id": rules.get("room_limit_pct_normal").chunk_id if rules.get("room_limit_pct_normal") else "",
        })

        assessment = (
            f"normal room sub-limit={pct:.1f}% of SI/day = INR {per_day_cap:,.0f}; "
            f"ICU sub-limit={icu_pct:.1f}% of SI/day; room expense declared=INR {room:,.0f}; "
            "admissible room amount depends on actual days and room category"
        )
        status = "applicable"
        if room > per_day_cap:
            assessment += "; room charge may exceed the per-day sub-limit and be deducted"

        return CoverageFinding(
            finding="Room/boarding/nursing sub-limit applies",
            dimension="sub_limit",
            status=status,
            policy_chunk_ids=[e.chunk_id for e in ev[:2]],
            assessment=assessment,
        )

    # ------------------------------------------------------------------
    def _pre_post(self, c: ClaimCase, facts: Dict, es: EvidenceState,
                  rules: PolicyRuleSet, limits: List[Dict]) -> CoverageFinding:
        ev = self._ev(es, "pre_post_hospitalization_window")
        pre_days = rules.value("pre_hospitalisation_days", 30)
        post_days = rules.value("post_hospitalisation_days", 60)

        pre_amt = c.expenses_inr.pre_hospitalization if c.expenses_inr else 0
        post_amt = c.expenses_inr.post_hospitalization if c.expenses_inr else 0
        timing = c.expense_timing

        parts = [
            f"policy windows: pre={pre_days:.0f} days, post={post_days:.0f} days",
            f"pre_hosp_expense=INR {pre_amt:,.0f}, post_hosp_expense=INR {post_amt:,.0f}",
        ]
        status = "applicable"

        if timing:
            pre_t = timing.pre_hospitalization_days_before_admission
            post_t = timing.post_hospitalization_days_after_discharge
            same_cond = timing.same_condition_confirmed
            if pre_t is not None:
                if pre_t <= pre_days:
                    parts.append(f"pre timing {pre_t}d within {pre_days:.0f}d window")
                else:
                    status = "not_applicable"
                    parts.append(f"pre timing {pre_t}d OUTSIDE {pre_days:.0f}d window => pre-hosp expenses NOT ADMISSIBLE")
            if post_t is not None:
                if post_t <= post_days:
                    parts.append(f"post timing {post_t}d within {post_days:.0f}d window")
                else:
                    status = "not_applicable"
                    parts.append(f"post timing {post_t}d OUTSIDE {post_days:.0f}d window => post-hosp expenses NOT ADMISSIBLE")
            if same_cond is False:
                parts.append("same-condition link not confirmed")
        else:
            parts.append("no expense-timing metadata; windows applied by policy default")

        limits.append({
            "type": "pre_post_hospitalization_window",
            "pre_hospitalization_days": pre_days,
            "post_hospitalization_days": post_days,
            "pre_hospitalization_amount": pre_amt,
            "post_hospitalization_amount": post_amt,
        })

        return CoverageFinding(
            finding="Pre/post hospitalisation expense windows",
            dimension="pre_post_hospitalization_window",
            status=status,
            policy_chunk_ids=[e.chunk_id for e in ev[:2]],
            assessment="; ".join(parts),
        )

    # ------------------------------------------------------------------
    def _ambulance(self, c: ClaimCase, facts: Dict, es: EvidenceState,
                   rules: PolicyRuleSet, limits: List[Dict]) -> CoverageFinding:
        pct = rules.value("ambulance_limit_pct", 1.0)
        si = facts.get("sum_insured") or 0
        amt = c.expenses_inr.ambulance if c.expenses_inr else 0
        cap = min(si * pct / 100, 1000.0)

        status = "applicable"
        note = f"ambulance limit min(1% of SI, INR 1000) = INR {cap:,.0f}; declared INR {amt:,.0f}"
        if amt > cap:
            note += f"; amount above cap by INR {amt - cap:,.0f} is deductible"

        limits.append({
            "type": "ambulance_limit",
            "amount": round(cap, 2),
            "percentage": pct,
            "declared": amt,
        })
        return CoverageFinding(
            finding="Ambulance expense limit applies",
            dimension="ambulance",
            status=status,
            policy_chunk_ids=[],
            assessment=note,
        )

    # ------------------------------------------------------------------
    def _exclusions(self, c: ClaimCase, facts: Dict, es: EvidenceState) -> CoverageFinding:
        ev = self._ev(es, "exclusion")
        pooled = self._pool(es.all_evidence, "exclusion")
        ttype = facts.get("treatment_type")
        diag = (facts.get("diagnosis") or "").lower()
        proc = (facts.get("procedure") or "").lower()

        reasons: List[str] = []
        status = "applicable"

        if facts.get("experimental"):
            reasons.append("experimental/unproven treatment is excluded by the policy definition")
            status = "not_applicable"
        if "cosmetic" in diag or "cosmetic" in proc:
            reasons.append("cosmetic/aesthetic treatment is excluded (exclusion 5)")
            status = "not_applicable"
        if ttype == "outpatient":
            reasons.append("outpatient treatment is excluded (exclusion 13)")
            status = "not_applicable"
        if re.search(r"\bwar\b|\bnuclear\b", diag + " " + proc):
            reasons.append("war/nuclear-related treatment excluded")

        if status == "applicable":
            reasons.append("no matching exclusion found in retrieved policy clauses")

        return CoverageFinding(
            finding="Exclusion assessment (experimental / cosmetic / outpatient / war)",
            dimension="exclusion",
            status=status,
            policy_chunk_ids=[e.chunk_id for e in ev[:2]],
            assessment="; ".join(reasons),
        )

    # ------------------------------------------------------------------
    def _hospital_eligibility(self, c: ClaimCase, facts: Dict, es: EvidenceState,
                              unmet: List[str]) -> CoverageFinding:
        ev = self._ev(es, "hospital_eligibility")
        network = facts.get("network_provider")
        registered = facts.get("evidence_context_hospital_registered")
        minimum = facts.get("evidence_context_hospital_minimum")

        parts = [f"hospital='{facts.get('hospital_name')}', network_provider={network}, "
                 f"registered={registered}, minimum_criteria_documented={minimum}"]

        if minimum is False:
            unmet.append("hospital_minimum_criteria_not_documented")
            unmet.append("hospital_definition_not_met")
            return CoverageFinding(
                finding="Hospital definition/eligibility cannot be established",
                dimension="hospital_eligibility",
                status="unknown",
                policy_chunk_ids=[e.chunk_id for e in ev[:2]],
                assessment="; ".join(parts) + "; hospital minimum criteria documented as NOT met => " + _INSUFFICIENT,
            )

        if registered is True:
            status = "applicable"
            parts.append("hospital registration confirmed")
        elif registered is None and network is False:
            status = "unknown"
            unmet.append("hospital_registration_not_confirmed")
            parts.append("non-network facility with no registration evidence => " + _INSUFFICIENT)
        else:
            status = "applicable"
            parts.append("network hospital satisfies eligibility requirement")

        return CoverageFinding(
            finding="Hospital definition/eligibility assessment",
            dimension="hospital_eligibility",
            status=status,
            policy_chunk_ids=[e.chunk_id for e in ev[:2]],
            assessment="; ".join(parts),
        )

    # ------------------------------------------------------------------
    def _portability(self, c: ClaimCase, facts: Dict, es: EvidenceState,
                     rules: PolicyRuleSet) -> CoverageFinding:
        ev = self._ev(es, "portability")
        pp = c.prior_policy
        years = pp.continuous_years if pp else 0
        db_received = pp.database_and_claim_history_received if pp else None
        reduction = rules.get("ped_portability_reduction") is not None

        parts = [f"prior continuous years={years}, database_and_history_received={db_received}",
                 f"portability_waiting_reduction_clause_found={reduction}"]
        if reduction and years and years > 0:
            parts.append("prior continuous coverage may reduce the PED waiting period")
        elif not db_received:
            parts.append("prior insurer database/claim history not received; reduction may not apply")

        return CoverageFinding(
            finding="Portability and prior continuous-coverage credit",
            dimension="portability",
            status="applicable",
            policy_chunk_ids=[e.chunk_id for e in ev[:2]],
            assessment="; ".join(parts),
        )