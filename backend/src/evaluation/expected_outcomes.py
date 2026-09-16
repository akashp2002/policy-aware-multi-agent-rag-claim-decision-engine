"""Expected outcomes for the public and candidate-created test cases.

These are the ground-truth labels used by the evaluation harness. The
public outcomes are the supplied target labels; the custom outcomes are
labels we assigned while designing the additional cases.
"""
from __future__ import annotations

from typing import Dict

# Supplied public cases (data/test_cases/public_test_cases.json)
PUBLIC_EXPECTED: Dict[str, str] = {
    "PUB-001": "ADMISSIBLE_WITH_LIMITS",
    "PUB-002": "NOT_ADMISSIBLE",
    "PUB-003": "NOT_ADMISSIBLE",
    "PUB-004": "ADMISSIBLE_WITH_LIMITS",
    "PUB-005": "ADMISSIBLE",
    "PUB-006": "NEEDS_REVIEW",
    "PUB-007": "ADMISSIBLE_WITH_LIMITS",
    "PUB-008": "NOT_ADMISSIBLE",
    "PUB-009": "ADMISSIBLE_WITH_LIMITS",
    "PUB-010": "ADMISSIBLE",
    "PUB-011": "NEEDS_REVIEW",
    "PUB-012": "NOT_ADMISSIBLE",
}

# Cases we created (custom_cases/candidate_test_cases.json)
CUSTOM_EXPECTED: Dict[str, str] = {
    "CUS-001": "ADMISSIBLE_WITH_LIMITS",
    "CUS-002": "NOT_ADMISSIBLE",
    "CUS-003": "ADMISSIBLE_WITH_LIMITS",
    "CUS-004": "NEEDS_REVIEW",
    "CUS-005": "NOT_ADMISSIBLE",
}

# Curated "gold clause" chunk ids per case. These are the policy chunks that
# contain the decisive clause(s) for the case. They drive the citation
# recall@k metric. Curated by inspecting the section structure of the policy
# (see data/policy/policy_chunks.json) and the task description of each case.
GOLD_CHUNKS: Dict[str, list] = {
    "PUB-001": ["P08-0023", "P08-0024"],            # room / ICU sub-limits
    "PUB-002": ["P09-0026"],                        # 30-day / first-year waiting
    "PUB-003": ["P05-0014", "P08-0023"],            # PED 48-month waiting
    "PUB-004": ["P07-0020", "P07-0021"],            # domiciliary scope
    "PUB-005": ["P09-0026", "P07-0021"],            # day-care / cataract list
    "PUB-006": ["P11-0032", "P10-0031"],            # claims procedure / evidence
    "PUB-007": ["P08-0024"],                        # category-specific cap
    "PUB-008": ["P09-0027", "P09-0028"],            # cosmetic exclusion
    "PUB-009": ["P08-0024"],                        # pre/post expense windows
    "PUB-010": ["P13-0038", "P12-0037"],            # portability waiting relief
    "PUB-011": ["P11-0032", "P10-0031"],            # claims procedure / evidence
    "PUB-012": ["P06-0017", "P09-0028"],            # experimental/unproven
    "CUS-001": ["P08-0023", "P08-0024"],            # sub-limits + ambulance
    "CUS-002": ["P09-0026"],                        # first-year disease list
    "CUS-003": ["P08-0023"],                        # room sub-limit
    "CUS-004": ["P02-0005", "P08-0023"],            # hospital definition + SI cap
    "CUS-005": ["P10-0029"],                        # outpatient exclusion
}

