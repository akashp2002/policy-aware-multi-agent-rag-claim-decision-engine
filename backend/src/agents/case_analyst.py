"""Case Analysis Agent – extracts facts, identifies decision dimensions, and produces an InvestigationPlan.

This agent runs a structured extraction over the raw ClaimCase JSON,
pulling out the information the downstream agents need (waiting-period
checks, pre-existing disease flags, expense totals, time windows) and
signalling dimensions where data is missing.

The plan is consumed verbatim by the Policy Evidence Agent to build
targeted retrieval queries, and by the Coverage & Exclusion Agent to
check each dimension.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

from src.agents.base import BaseAgent
from src.models.schemas import (
    ClaimCase,
    CaseState,
    CoverageFinding,
    Dimension,
    InvestigationItem,
    InvestigationPlan,
)


class CaseAnalysisAgent(BaseAgent):
    name = "CaseAnalysisAgent"

    # ------------------------------------------------------------------
    def run(self, claim_case: ClaimCase) -> CaseState:
        """Analyse a ClaimCase and produce a structured CaseState."""
        facts = self._extract_facts(claim_case)
        dims = self._determine_dimensions(claim_case, facts)
        plan = self._build_plan(dims, claim_case, facts)
        missing = self._find_missing(claim_case, facts)

        return CaseState(
            case=claim_case,
            plan=plan,
            extracted_facts=facts,
            decision_dimensions=[d.value for d in dims],
            missing_fields=missing,
        )

    # ------------------------------------------------------------------
    def _extract_facts(self, c: ClaimCase) -> Dict[str, Any]:
        total_expense = 0.0
        if c.expenses_inr:
            total_expense = sum(
                getattr(c.expenses_inr, f, 0) or 0
                for f in ["room", "doctor_fees", "medicines_diagnostics",
                           "pre_hospitalization", "post_hospitalization", "ambulance"]
            )

        days_between = None
        policy_months_elapsed = None
        if c.policy_start_date and c.claim_date:
            try:
                psd = datetime.strptime(c.policy_start_date, "%Y-%m-%d")
                cd  = datetime.strptime(c.claim_date,      "%Y-%m-%d")
                days_between = (cd - psd).days
                policy_months_elapsed = days_between // 30
            except ValueError:
                pass

        return {
            "total_expense": total_expense,
            "room_cost": (c.expenses_inr.room if c.expenses_inr else 0),
            "has_room_expense": (c.expenses_inr.room if c.expenses_inr else 0) > 0,
            "treatment_type": c.treatment.type if c.treatment else None,
            "is_inpatient": (c.treatment.type if c.treatment else "") == "inpatient",
            "is_day_care": (c.treatment.type if c.treatment else "") == "day_care",
            "is_domiciliary": (c.treatment.type if c.treatment else "") == "domiciliary",
            "admission_hours": c.treatment.admission_hours if c.treatment else 0,
            "pre_existing": c.treatment.pre_existing if c.treatment else False,
            "experimental": c.treatment.experimental if c.treatment else False,
            "network_provider": c.hospital.network_provider if c.hospital else None,
            "diagnosis": (c.treatment.diagnosis if c.treatment else None) or "",
            "procedure": (c.treatment.procedure if c.treatment else None) or "",
            "days_between_policy_start_and_claim": days_between,
            "policy_months_elapsed": policy_months_elapsed,
            "continuous_coverage_months": c.continuous_coverage_months or 0,
            "prior_insurer_continuous_years": c.prior_insurer_continuous_years or 0,
            "sum_insured": c.sum_insured_inr or 0,
            "has_prior_policy": c.prior_policy is not None,
            "documents_count": len(c.documents),
            "hospital_name": c.hospital.name if c.hospital else None,
            "hospital_room_unavailable": (c.treatment.hospital_room_unavailable if c.treatment else None),
            "patient_cannot_be_moved": (c.treatment.patient_cannot_be_moved if c.treatment else None),
            "evidence_context_hospital_registered": (c.evidence_context.hospital_registered if c.evidence_context else None),
            "evidence_context_hospital_minimum": (c.evidence_context.hospital_minimum_criteria_documented if c.evidence_context else None),
        }

    # ------------------------------------------------------------------
    def _determine_dimensions(self, c: ClaimCase, facts: Dict[str, Any]) -> List[Dimension]:
        dims: List[Dimension] = []

        # Always check scope/definition
        dims.append(Dimension.HOSPITALIZATION)

        if facts["pre_existing"]:
            dims.append(Dimension.PRE_EXISTING)

        # Waiting period (policy <1 year)
        if facts["continuous_coverage_months"] < 12:
            dims.append(Dimension.WAITING_PERIOD)

        # Specific-disease 1-year waiting list is always checked because the
        # diagnosis/procedure may fall within the policy's listed diseases.
        dims.append(Dimension.SPECIFIC_DISEASE_WAITING)

        if facts["is_domiciliary"]:
            dims.append(Dimension.DOMICILIARY)

        if facts["is_day_care"] or (facts["admission_hours"] is not None and facts["admission_hours"] < 24):
            dims.append(Dimension.DAY_CARE)

        if facts["has_room_expense"]:
            dims.append(Dimension.SUB_LIMIT)

        if facts["is_inpatient"] or facts["is_day_care"]:
            dims.append(Dimension.SUM_INSURED)

        if facts["is_inpatient"]:
            dims.append(Dimension.PRE_POST_EXPENSE)

        if facts["experimental"]:
            dims.append(Dimension.EXCLUSION)

        if not facts["network_provider"] and facts["network_provider"] is not None:
            if not facts["is_domiciliary"]:
                dims.append(Dimension.HOSPITAL_ELIGIBILITY)

        if facts["has_prior_policy"]:
            dims.append(Dimension.PORTABILITY)

        if ((facts["evidence_context_hospital_minimum"] is False) or
                (facts["evidence_context_hospital_minimum"] is None
                 and not facts["network_provider"] and not facts["is_domiciliary"])):
            dims.append(Dimension.HOSPITAL_ELIGIBILITY)

        if not dims:
            dims.append(Dimension.EXCLUSION)

        return dims

    # ------------------------------------------------------------------
    def _build_plan(self, dims: List[Dimension], c: ClaimCase, facts: Dict[str, Any]) -> InvestigationPlan:
        items: List[InvestigationItem] = []
        diag = facts["diagnosis"][:80] if facts["diagnosis"] else "N/A"
        ttype = facts["treatment_type"] or "unknown"

        _Q = {
            Dimension.HOSPITALIZATION: [
                "Hospital definition and minimum requirements",
                "What constitutes a covered Hospitalization",
                "Inpatient day-care domiciliary eligibility criteria",
            ],
            Dimension.WAITING_PERIOD: [
                "Initial waiting period duration and applicability",
                "First-year treatment waiting period",
                "What counts as day one of policy for waiting period",
            ],
            Dimension.SPECIFIC_DISEASE_WAITING: [
                "First-year operation treatment of listed diseases",
                "One-year waiting period disease list",
            ],
            Dimension.PRE_EXISTING: [
                "Pre-existing disease definition",
                "Pre-existing disease waiting period duration",
                "How prior continuous coverage affects PED waiting period",
            ],
            Dimension.DOMICILIARY: [
                "Domiciliary Hospitalisation definition and conditions",
                "Domiciliary sub-limit amount",
                "When domiciliary treatment is admissible",
            ],
            Dimension.DAY_CARE: [
                "Day-care procedure definition",
                "List of covered day-care procedures or exclusion criteria",
                "24-hour hospitalization rule exceptions",
            ],
            Dimension.SUB_LIMIT: [
                "Room, boarding and nursing expense sub-limit",
                "Room category cap or percentage limit",
                "How room limit affects other expenses",
            ],
            Dimension.SUM_INSURED: [
                "Maximum sum insured payable",
                "Cumulative bonus effect on sum insured",
                "Cumulative bonus conditions",
            ],
            Dimension.PRE_POST_EXPENSE: [
                "Pre-hospitalization coverage window (days before admission)",
                "Post-hospitalization coverage window (days after discharge)",
                "What qualifies as pre/post hospitalization expenses",
            ],
            Dimension.EXCLUSION: [
                "List of exclusions",
                "Experimental or unproven treatment exclusion",
                "Cosmetic treatment exclusion",
                "War terrorism nuclear exclusion",
            ],
            Dimension.HOSPITAL_ELIGIBILITY: [
                "Hospital registration requirements",
                "Hospital minimum criteria (beds, staff, OT, etc.)",
                "Network vs non-network hospital effect on claim",
            ],
            Dimension.PORTABILITY: [
                "Portability provisions",
                "Waiting period credit for prior continuous coverage",
                "Portability eligibility requirements",
            ],
            Dimension.CO_PAY: [
                "Co-payment clause applicability",
                "Co-pay percentage",
            ],
        }

        for dim in dims:
            questions = _Q.get(dim, ["General coverage question"])
            req_fields = self._required_fields_for_dim(dim)
            items.append(
                InvestigationItem(
                    dimension=dim.value,
                    question="; ".join(questions),
                    required_fields=req_fields,
                )
            )

        return InvestigationPlan(
            dimensions=items,
            case_summary=(
                f"Claim {c.case_id}: {facts['diagnosis'] or 'N/A'}, "
                f"{facts['treatment_type']} treatment, "
                f"total expense INR {facts['total_expense']:.0f}, "
                f"continuous coverage {facts['continuous_coverage_months']:.0f} months, "
                f"sum insured INR {facts['sum_insured']:.0f}"
            ),
            missing_fields=[],
        )

    # ------------------------------------------------------------------
    def _required_fields_for_dim(self, dim: Dimension) -> List[str]:
        m = {
            Dimension.HOSPITALIZATION: ["hospital.name", "hospital.network_provider", "treatment.type"],
            Dimension.WAITING_PERIOD: ["continuous_coverage_months", "policy_start_date"],
            Dimension.SPECIFIC_DISEASE_WAITING: ["treatment.diagnosis", "treatment.procedure", "continuous_coverage_months"],
            Dimension.PRE_EXISTING: ["treatment.pre_existing"],
            Dimension.DOMICILIARY: ["treatment.type", "expenses_inr"],
            Dimension.DAY_CARE: ["treatment.type", "treatment.admission_hours"],
            Dimension.SUB_LIMIT: ["expenses_inr.room", "sum_insured_inr"],
            Dimension.SUM_INSURED: ["sum_insured_inr", "continuous_coverage_months"],
            Dimension.PRE_POST_EXPENSE: ["expenses_inr.pre_hospitalization", "expenses_inr.post_hospitalization"],
            Dimension.EXCLUSION: ["treatment.diagnosis", "treatment.experimental"],
            Dimension.HOSPITAL_ELIGIBILITY: ["hospital.name", "evidence_context.hospital_registered"],
            Dimension.PORTABILITY: ["prior_policy"],
        }
        return m.get(dim, [])

    # ------------------------------------------------------------------
    def _find_missing(self, c: ClaimCase, facts: Dict[str, Any]) -> List[str]:
        missing: List[str] = []
        if not c.sum_insured_inr:
            missing.append("sum_insured_inr")
        if not c.policy_start_date:
            missing.append("policy_start_date")
        if not c.claim_date:
            missing.append("claim_date")
        if c.hospital and c.hospital.name and c.hospital.network_provider is None:
            missing.append("hospital.network_provider")
        if not c.documents:
            missing.append("documents")
        if (facts.get("evidence_context_hospital_registered") is None
                and facts.get("network_provider") is not True
                and not facts.get("is_domiciliary")):
            missing.append("evidence_context.hospital_registered")
        return missing