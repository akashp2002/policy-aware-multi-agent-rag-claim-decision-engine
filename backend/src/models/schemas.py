"""Pydantic schema definitions for the claim decision engine.

These define the structured contracts exchanged between agents, the
input data model, and the final decision response. All agent-to-agent
communication flows through these typed structures rather than
free-form text.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class DecisionStatus(str, Enum):
    ADMISSIBLE = "ADMISSIBLE"
    ADMISSIBLE_WITH_LIMITS = "ADMISSIBLE_WITH_LIMITS"
    PARTIALLY_ADMISSIBLE = "PARTIALLY_ADMISSIBLE"
    NOT_ADMISSIBLE = "NOT_ADMISSIBLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class EvidenceType(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    RERANKED = "reranked"


class AgentName(str, Enum):
    CASE_ANALYST = "CaseAnalysisAgent"
    POLICY_EVIDENCE = "PolicyEvidenceAgent"
    COVERAGE_EXCLUSION = "CoverageExclusionAgent"
    DECISION = "DecisionAgent"
    VALIDATION = "ValidationAgent"


class Dimension(str, Enum):
    HOSPITALIZATION = "hospitalization_definition"
    WAITING_PERIOD = "waiting_period"
    SPECIFIC_DISEASE_WAITING = "specific_disease_waiting"
    PRE_EXISTING = "pre_existing_disease"
    EXCLUSION = "exclusion"
    SUB_LIMIT = "sub_limit"
    PRE_POST_EXPENSE = "pre_post_hospitalization_window"
    PORTABILITY = "portability"
    SUM_INSURED = "sum_insured"
    CUMULATIVE_BONUS = "cumulative_bonus"
    HOSPITAL_ELIGIBILITY = "hospital_eligibility"
    DAY_CARE = "day_care"
    DOMICILIARY = "domiciliary"
    CO_PAY = "co_pay"
    DEDUCTIBLE = "deductible"
    REQUIRED_EVIDENCE = "required_evidence"


# ---------------------------------------------------------------------------
# Input data model
# ---------------------------------------------------------------------------
class Patient(BaseModel):
    age: Optional[int] = None


class Hospital(BaseModel):
    name: Optional[str] = None
    network_provider: Optional[bool] = None


class Treatment(BaseModel):
    type: Optional[str] = None  # inpatient, day_care, domiciliary, outpatient
    admission_hours: Optional[int] = None
    diagnosis: Optional[str] = None
    procedure: Optional[str] = None
    pre_existing: Optional[bool] = None
    experimental: Optional[bool] = None
    hospital_room_unavailable: Optional[bool] = None
    patient_cannot_be_moved: Optional[bool] = None
    # Extra fields tolerated for robustness
    model_config = {"extra": "allow"}


class Expenses(BaseModel):
    room: float = 0
    doctor_fees: float = 0
    medicines_diagnostics: float = 0
    pre_hospitalization: float = 0
    post_hospitalization: float = 0
    ambulance: float = 0
    model_config = {"extra": "allow"}


class PriorPolicy(BaseModel):
    insurer_type: Optional[str] = None
    continuous_years: Optional[float] = None
    database_and_claim_history_received: Optional[bool] = None
    previous_sum_insured_inr: Optional[float] = None
    model_config = {"extra": "allow"}


class EvidenceContext(BaseModel):
    """Additional context that may or may not be present."""
    hospital_registered: Optional[bool] = None
    medical_necessity_confirmed: Optional[bool] = None
    hospital_minimum_criteria_documented: Optional[bool] = None
    model_config = {"extra": "allow"}


class ExpenseTiming(BaseModel):
    pre_hospitalization_days_before_admission: Optional[int] = None
    post_hospitalization_days_after_discharge: Optional[int] = None
    same_condition_confirmed: Optional[bool] = None
    model_config = {"extra": "allow"}


class ClaimCase(BaseModel):
    case_id: str
    policy_id: Optional[str] = None
    policy_start_date: Optional[str] = None
    claim_date: Optional[str] = None
    sum_insured_inr: Optional[float] = None
    continuous_coverage_months: Optional[float] = None
    prior_insurer_continuous_years: Optional[float] = None
    patient: Optional[Patient] = None
    hospital: Optional[Hospital] = None
    treatment: Optional[Treatment] = None
    expenses_inr: Optional[Expenses] = None
    documents: List[str] = Field(default_factory=list)
    task: Optional[str] = None
    evidence_context: Optional[EvidenceContext] = None
    expense_timing: Optional[ExpenseTiming] = None
    prior_policy: Optional[PriorPolicy] = None
    # Robustness: tolerate unknown/extraneous fields
    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------
class PolicyChunk(BaseModel):
    chunk_id: str
    page: int
    section: str
    heading: Optional[str] = None
    text: str
    char_len: int = 0


class PolicyEvidence(BaseModel):
    chunk_id: str
    page: int
    section: str
    heading: Optional[str] = None
    text: str
    score: float = 0.0
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    rank: Optional[int] = None


class RetrievedEvidenceBundle(BaseModel):
    query: str = ""
    evidence: List[PolicyEvidence] = Field(default_factory=list)
    retrieval_method: str = "hybrid"
    total_candidates: int = 0
    final_count: int = 0


# ---------------------------------------------------------------------------
# Agent state contracts (structured, typed)
# ---------------------------------------------------------------------------
class InvestigationItem(BaseModel):
    dimension: str
    question: str
    required_fields: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending | satisfied | blocked


class Finding(BaseModel):
    """A single piece of a case-level finding (used in final response)."""
    claim: str = ""
    detail: str = ""
    dimension: str = ""
    policy_chunk_ids: List[str] = Field(default_factory=list)
    severity: str = "info"  # info | warning | critical


class InvestigationPlan(BaseModel):
    dimensions: List[InvestigationItem] = Field(default_factory=list)
    case_summary: str = ""
    missing_fields: List[str] = Field(default_factory=list)


class CaseState(BaseModel):
    """Produced by the Case Analysis Agent."""
    case: ClaimCase
    plan: InvestigationPlan
    extracted_facts: Dict[str, Any] = Field(default_factory=dict)
    decision_dimensions: List[str] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)


class EvidenceState(BaseModel):
    """Produced by the Policy Evidence Agent."""
    plan: InvestigationPlan
    queries: List[str] = Field(default_factory=list)
    evidence_by_dimension: Dict[str, List[PolicyEvidence]] = Field(default_factory=dict)
    all_evidence: List[PolicyEvidence] = Field(default_factory=list)
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict)


class CoverageFinding(BaseModel):
    finding: str
    dimension: str
    status: str = "applicable"  # applicable | not_applicable | unknown
    policy_chunk_ids: List[str] = Field(default_factory=list)
    assessment: str = ""


class CoverageFindings(BaseModel):
    """Produced by the Coverage & Exclusion Agent."""
    findings: List[CoverageFinding] = Field(default_factory=list)
    coverage_scope_assessment: str = ""
    applicable_limits: List[Dict[str, Any]] = Field(default_factory=list)
    waiting_period_assessment: str = ""
    exclusion_assessment: str = ""
    unmet_conditions: List[str] = Field(default_factory=list)


class FinalDecision(BaseModel):
    status: DecisionStatus
    confidence: float
    key_findings: List[str] = Field(default_factory=list)
    applicable_limits: List[Dict[str, Any]] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    rationale: str = ""


class DecisionState(BaseModel):
    """Produced by the Decision Agent."""
    decision: FinalDecision


class ValidationFailure(BaseModel):
    claim: str
    reason: str


class ValidationResult(BaseModel):
    """Produced by the Validation Agent."""
    status: ValidationStatus = ValidationStatus.PASS
    unsupported_claims: List[ValidationFailure] = Field(default_factory=list)


class Citation(BaseModel):
    claim: str
    source: str = "policy.pdf"
    page: Optional[int] = None
    section: str = ""
    chunk_id: str = ""
    dimension: str = ""
    excerpt: str = ""


class AgentTraceEntry(BaseModel):
    agent: str
    action: str
    detail: str = ""
    elapse_ms: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    score: Optional[float] = None


# ---------------------------------------------------------------------------
# Stage-specific intermediate bundles
# ---------------------------------------------------------------------------
class StageOutput(BaseModel):
    """Bundles per-agent outputs, un-validated decision pieces."""
    decision_status: DecisionStatus
    confidence: float
    key_findings: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    applicable_limits: List[Dict[str, Any]] = Field(default_factory=list)
    rationale: str = ""
    citations: List[Citation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Final API response
# ---------------------------------------------------------------------------
class DecisionResponse(BaseModel):
    case_id: str
    decision: DecisionStatus
    confidence: float
    key_findings: List[str] = Field(default_factory=list)
    applicable_limits: List[Dict[str, Any]] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    validation: ValidationResult
    trace: List[AgentTraceEntry] = Field(default_factory=list)
    policy: Dict[str, Any] = Field(default_factory=dict)


class AnalysisRequest(BaseModel):
    case: ClaimCase


class AnalysisResponse(BaseModel):
    result: DecisionResponse


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = ""
    chunks_indexed: int = 0
    version: str = ""