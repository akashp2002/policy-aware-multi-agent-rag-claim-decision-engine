# Policy-Aware Multi-Agent RAG Claim Decision Engine

An end-to-end take-home implementation that decides whether a health-insurance
claim is admissible under a specific policy PDF, by combining **hybrid
retrieval** (BM25 + dense embeddings + cross-encoder reranking) with a
**genuine multi-agent workflow** that passes **structured, typed state**
between agents and grounds every decision in **citable policy clauses**.

```
                ┌───────────────────────────────────────────────────────────┐
                │                    FastAPI (src/api/main.py)              │
                │      POST /analyze   ·   GET /health   ·   GET /          │
                └───────────────┬───────────────────────────────────────────┘
                                │ claim payload
                                ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 1. Case      │──>│ 2. Policy    │──>│ 3. Coverage  │──>│ 4. Decision  │
│    Analysis  │   │    Evidence  │   │    & Excl.   │   │    Agent     │
│    Agent     │   │    Agent     │   │    Agent     │   │              │
└──────────────┘   └──────────────┘   └──────────────┘   └──────┬───────┘
   plan+facts          query+evidence   findings+limits        status
        ▲                    ▲             │                     │
        │        hybrid retrieval        │ validation retry loop │
        │   ┌────┴─────┐ ┌──┴───┐        ▼                     │
        └── │ BM25     │ │ bge- │  ┌──────────────┐             │
            │ (rank-   │ │ small│  │ 5. Validation│<─────────────┘
            │  bm25)   │ │ -en  │  │    Agent     │
            └────┬─────┘ └──┬───┘  └──────────────┘
                 │ reranker │        citations + PASS/FAIL
     50 policy chunks ─────┴─> cross-encoder rerank ─> grounded decision
```

All inter-agent communication uses Pydantic models (`src/models/schemas.py`)
rather than loose text, so every stage is verifiable and every material
finding carries the policy chunk ids that support it.

## Requirements satisfied

| Requirement | Where |
| --- | --- |
| Read & index the policy PDF | `src/retrieval/policy_ingestion.py` (17 pages → 50 section-tagged chunks) |
| Hybrid retrieval (sparse + dense) | `src/retrieval/hybrid_search.py` — BM25 + `BAAI/bge-small-en-v1.5` + RRF |
| ≥ 3 specialized agents | 5 agents: CaseAnalysis, PolicyEvidence, Coverage&Exclusion, Decision, Validation (`src/agents/`) |
| Structured/shared agent state | Typed Pydantic contracts: `CaseState → EvidenceState → CoverageFindings → DecisionState → ValidationResult` |
| FastAPI `/analyze` + `/health` | `src/api/main.py` |
| Streamlit frontend | `frontend/app.py` (bundled cases or paste JSON) |
| ≥ 5 custom test cases | `custom_cases/candidate_test_cases.json` (CUS-001..005) |
| Evaluation harness | `src/evaluation/` → `output/evaluation_report.{json,md}` |
| Deployment config | `Dockerfile` + `docker-compose.yml` (see below) |

## Current evaluation results

Run with: `python -m src.evaluation.run_evaluation`

- **Decision accuracy: 12/12 public + 5/5 custom = 17/17 (100%)**
- Validation (citation-grounding) PASS rate: **100%**
- Mean confidence: 0.84 · Mean latency: 3.35 s/case (single process)
- Mean citation recall@k against curated gold clauses: **61.8%** (see "Known limits")

## Live Demo

- **Frontend:** [Streamlit reviewer application](https://policy-aware-multi-agent-rag-claim-decision-engine-qxfyty8m9ws.streamlit.app/)
- **Backend API:** [FastAPI service](https://claim-decision-api.onrender.com/)
- **API health:** [Live health check](https://claim-decision-api.onrender.com/health)
- **API documentation:** [Swagger UI](https://claim-decision-api.onrender.com/docs)

The hosted demo uses the deterministic policy-rule fallback and does not
require an external LLM API key. The free Render instance may sleep after
inactivity, so the first request can take longer while the embedding model
loads.

## Quickstart

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate      # Windows
# or: source .venv/bin/activate                     # macOS / Linux
pip install -r requirements.txt
python -m src.retrieval.policy_ingestion            # (re)build policy chunk index
python -m src.evaluation.run_evaluation             # run the harness
python -m pytest tests -q                           # unit + integration tests
```

### API

```bash
cd backend
uvicorn src.api.main:app --reload --port 8000
curl http://localhost:8000/health
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
   -d '{"case": {"case_id": "CUS-005", "treatment": {"type": "outpatient"}}}'
```

### Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

By default the local frontend calls `http://localhost:8000`. For hosted
Streamlit, set the `CLAIM_API_BASE` secret to the deployed API URL and set
`ALLOW_IN_PROCESS_FALLBACK=false`; the hosted frontend should call the public
backend rather than attempt to load backend files locally.

## LLM configuration

The pipeline runs with **no external dependencies** by default. With no API
key, `LLMProvider` (`src/llm/provider.py`) uses a deterministic heuristic
path — the evaluation above runs entirely in this mode.

To enable LLM-based reasoning quality:

```bash
# Groq (free tier)
$env:GROQ_API_KEY = "..." ; $env:CLAIM_LLM_PROVIDER = "auto"

# or any OpenAI-compatible endpoint
$env:OPENAI_API_KEY = "..." ; $env:OPENAI_BASE_URL = "..." ; $env:OPENAI_MODEL = "gpt-4o-mini"
```

The provider is used only where an actual LLM adds value (query expansion in
the Policy Evidence agent). Rule application, decision synthesis and
validation remain deterministic, so results stay reproducible.

## Project layout

```
backend/                            # FastAPI service + engine + data
├── src/
│   ├── models/schemas.py           # typed contracts for all agent state
│   ├── retrieval/                  # ingestion, embeddings, hybrid search
│   ├── agents/                     # 5 specialized agents + orchestrator
│   ├── llm/provider.py             # groq/openai/heuristic fallback
│   ├── api/main.py                 # FastAPI application
│   └── evaluation/                 # harness + metrics
├── data/
│   ├── policy/policy_chunks.json   # 50 indexed, section-tagged chunks
│   └── test_cases/public_test_cases.json   # the 12 supplied cases (unmodified)
├── custom_cases/candidate_test_cases.json   # 5 additional cases we designed
├── tests/                          # unit + integration + API tests
├── output/evaluation_report.md     # generated evaluation report
├── requirements.txt
├── Dockerfile
└── .env.example

frontend/                           # Streamlit UI
├── app.py
├── requirements.txt

docker-compose.yml
README.md
```

## Design decisions & trade-offs

1. **Typed agent state instead of free-form tool calls.** Each agent consumes
   and emits Pydantic models (`CaseState`, `EvidenceState`, …). This makes the
   pipeline testable, inspectable, and — importantly — lets the Validation
   agent verify that *every material claim is backed by a cited chunk* using
   token-overlap grounding.

2. **Cross-encoder rerank on top of RRF fusion.** BM25 (sparse) and bge-small
   (dense) are fused with Reciprocal Rank Fusion, then re-ranked with
   `ms-marco-MiniLM-L-6-v2`. This materially improves precision on clause-heavy
   queries without needing a GPU (all models run in ONNX on CPU). The hosted
   free-tier deployment sets `RERANK_ENABLED=false` to stay below the 512 MB
   memory limit; dense+sparse retrieval and RRF remain active there.

3. **Deterministic, reproducible decisions.** The coverage rules
   (sub-limits, waiting periods, windows, domiciliary/day-care treatment,
   category caps) are extracted from the policy text by regex
   (`src/agents/policy_rules.py`) and applied heuristically. A real LLM is an
   optional layer only. Trade-off: heuristic coverage is narrower than an LLM's
   reading comprehension, but it is deterministic and auditable.

4. **Validation-driven retry loop.** If a material finding cannot be backed by
   retrieved chunks, the orchestrator enriches retrieval and re-runs
   coverage→decision (up to 2 passes). Persistent failure downgrades the claim
   to `NEEDS_REVIEW` instead of emitting an unsupported decision.

5. **Per-dimension targeted retrieval.** The Case Analysis agent emits an
   investigation plan; the Policy Evidence agent issues one query per
   dimension (with deterministic expansion), so each rule dimension reads from
   a focused evidence pool.

6. **Citation selection is consistent with validation.** Both use the same
   token-overlap utilities (`src/agents/text_match.py`), so what the Coverage
   agent cites is what the Validation agent checks.

### Known limits

- **Citation recall@k (61.8%).** Gold-clause recall is limited because the
   orchestrator reserves one citation slot per final finding, with a maximum
   of 12 findings. Retrieval itself recovers the gold clauses in 74% of
   curated cases; the cap is a deliberate conciseness/safety trade-off.
- **Curated gold clauses are a proxy.** `GOLD_CHUNKS` in
  `src/evaluation/expected_outcomes.py` is hand-curated from the section map;
  treat the recall figure as indicative, not canonical.
- **Heuristic rules cover the sections parsed so far** (definitions, waiting
  periods, sub-limits, day-care, domiciliary, exclusions, pre/post windows,
  portability). Obscure clauses may be retrieved but not yet reduced to rules.
- **NEEDS_REVIEW cases are conservative.** When evidence (e.g. hospital
  registration, medical necessity) is missing, the engine prefers review over
  a false admission/denial.

### Failure analysis

The following cases were used to identify reliability gaps and guide the
retrieval, citation, and abstention safeguards. A correct decision label does
not mean that the evidence path was perfect; the evaluation report records
both outcomes separately.

1. **PUB-006 — correct abstention, incomplete evidence retrieval.** The
   engine returned `NEEDS_REVIEW`, as expected, because the case did not
   include enough claim or hospital evidence to establish all conditions.
   However, citation recall was 0%: the retrieved citations did not include
   the claims-procedure chunks identified as the curated gold evidence. The
   root cause was that the dimensional retrieval plan favored coverage and
   limit clauses, while the missing-document/claims-procedure path was not
   guaranteed to receive a dedicated high-priority query. The implemented
   mitigation is structured missing-field detection in the Case Analysis and
   Decision agents, plus conservative `NEEDS_REVIEW` classification when a
   required condition is unknown. The remaining improvement is to add a
   mandatory `required_evidence` retrieval dimension for every abstention
   candidate.

2. **PUB-012 — correct exclusion decision, unsupported gold citation path.**
   The engine correctly returned `NOT_ADMISSIBLE` for the experimental or
   unproven treatment, but citation recall was 0% against the curated gold
   chunks for that exclusion. The root cause was competition between generic
   exclusion queries and more strongly matching hospitalization, waiting
   period, and sub-limit results in the pooled top-k evidence. The implemented
   mitigation is hybrid BM25+dense retrieval, RRF fusion, cross-encoder
   reranking, and validation-triggered retry with enriched queries. The
   remaining improvement is to reserve citation capacity for the decisive
   finding instead of allowing every dimension to compete equally for the
   previous eight-citation cap. The implemented mitigation now reserves one
   citation slot per final finding, up to twelve findings, so the decisive
   dimensions remain inspectable.

3. **CUS-004 — deliberate abstention caused by missing facility evidence.**
   The bill exceeds the sum insured and the non-network facility's
   registration and medical-necessity evidence are undocumented. Returning
   `NEEDS_REVIEW` is the safe result, but it demonstrates that the system
   cannot make a final eligibility decision from billing data alone. The
   implemented mitigation is the typed `EvidenceContext` contract and
   Decision Agent document-gap checks, which preserve the known gap in
   `missing_evidence` rather than guessing. The remaining improvement is to
   expose a reviewer checklist or upload flow for the missing registration and
   medical-necessity documents.

These cases explain why the system reports 100% decision accuracy while mean
curated citation recall remains 61.8%: decision correctness and evidence
retrieval quality are measured independently.

## Deployment

For local or self-hosted deployment, copy the placeholder settings from
`backend/.env.example` into your deployment platform's environment settings.
Do not commit a real `.env` file or API keys. At minimum, configure the
frontend origin for the deployed API:

```text
CORS_ORIGINS=https://policy-aware-multi-agent-rag-claim-decision-engine-qxfyty8m9ws.streamlit.app
MAX_REQUEST_BYTES=1048576
```

With Docker Desktop running, `docker compose up --build -d` starts the API on
`:8000` and the Streamlit UI on `:8501`. The API health check waits for the
lightweight `/health` endpoint before starting the frontend. Verify the
deployment with `GET /health` and then a representative `POST /analyze`.

For a public deployment, place the API and frontend behind HTTPS and configure
the hosting provider's secret store for `GROQ_API_KEY` or `OPENAI_API_KEY`.
Restrict `CORS_ORIGINS` to the actual frontend URL, keep the default request
limit unless a larger claim schema is required, and add platform-level
authentication/rate limiting before exposing `/analyze` to untrusted users.

### Hosted deployment settings

For the current live deployment:

**Render API environment variables**

```text
CLAIM_LLM_PROVIDER=auto
CORS_ORIGINS=https://policy-aware-multi-agent-rag-claim-decision-engine-qxfyty8m9ws.streamlit.app
MAX_REQUEST_BYTES=1048576
RERANK_ENABLED=false
```

**Streamlit Cloud secrets**

```toml
CLAIM_API_BASE = "https://claim-decision-api.onrender.com"
ALLOW_IN_PROCESS_FALLBACK = "false"
```

No LLM key is required for the live demo. Provider keys should be added only
through the hosting platform's secret manager and never committed to GitHub.

## Submission checklist

- Live frontend and backend URLs are listed above.
- `/health` and `/docs` are publicly available for API verification.
- Twelve supplied public cases and five additional candidate cases are
   evaluated by the reproducible harness.
- The system exposes five specialized agents, typed state contracts, hybrid
   retrieval, citations, validation, abstention, and an execution trace.
- Failure analysis and the architecture note are included in `docs/` and this
   README.
- Generated evaluation reports are included under `backend/output/`.
- No credentials are committed; `.env.example` contains placeholders only.