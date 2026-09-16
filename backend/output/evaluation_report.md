# Evaluation Report

| Metric | Value |
| --- | --- |
| Cases evaluated | 17 |
| Decision accuracy | 100.0% |
| Validation pass rate | 100.0% |
| Mean citation recall@k (curated gold clauses) | 61.8% |
| Citation coverage (>=1 grounded citation) | 100.0% |
| Mean confidence | 0.84 |
| Mean latency | 3.35s |
| Correct / total | 17/17 |

## Per-case results

| Case | Split | Expected | Predicted | OK | Conf | Validation | Recall | Cites | Limits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PUB-001 | public | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | yes | 0.90 | PASS | 100% | 9 | 3 |
| PUB-002 | public | NOT_ADMISSIBLE | NOT_ADMISSIBLE | yes | 0.92 | PASS | 100% | 9 | 3 |
| PUB-003 | public | NOT_ADMISSIBLE | NOT_ADMISSIBLE | yes | 0.92 | PASS | 50% | 9 | 3 |
| PUB-004 | public | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | yes | 0.90 | PASS | 100% | 7 | 1 |
| PUB-005 | public | ADMISSIBLE | ADMISSIBLE | yes | 0.97 | PASS | 50% | 7 | 0 |
| PUB-006 | public | NEEDS_REVIEW | NEEDS_REVIEW | yes | 0.47 | PASS | 0% | 9 | 3 |
| PUB-007 | public | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | yes | 0.90 | PASS | 100% | 9 | 3 |
| PUB-008 | public | NOT_ADMISSIBLE | NOT_ADMISSIBLE | yes | 0.92 | PASS | 50% | 9 | 3 |
| PUB-009 | public | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | yes | 0.90 | PASS | 100% | 9 | 3 |
| PUB-010 | public | ADMISSIBLE | ADMISSIBLE | yes | 0.97 | PASS | 0% | 8 | 0 |
| PUB-011 | public | NEEDS_REVIEW | NEEDS_REVIEW | yes | 0.47 | PASS | 0% | 9 | 3 |
| PUB-012 | public | NOT_ADMISSIBLE | NOT_ADMISSIBLE | yes | 0.92 | PASS | 50% | 9 | 3 |
| CUS-001 | custom | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | yes | 0.90 | PASS | 100% | 9 | 3 |
| CUS-002 | custom | NOT_ADMISSIBLE | NOT_ADMISSIBLE | yes | 0.92 | PASS | 100% | 9 | 3 |
| CUS-003 | custom | ADMISSIBLE_WITH_LIMITS | ADMISSIBLE_WITH_LIMITS | yes | 0.90 | PASS | 100% | 9 | 3 |
| CUS-004 | custom | NEEDS_REVIEW | NEEDS_REVIEW | yes | 0.47 | PASS | 50% | 9 | 3 |
| CUS-005 | custom | NOT_ADMISSIBLE | NOT_ADMISSIBLE | yes | 0.90 | PASS | 0% | 7 | 0 |
