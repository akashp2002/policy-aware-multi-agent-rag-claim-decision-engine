"""FastAPI application: POST /analyze, GET /health.

Loads the policy index and hybrid search engine at startup (lazy, so the
/health endpoint works without heavy model loading), validates inputs,
and returns the structured DecisionResponse.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.agents.orchestrator import AgentOrchestrator
from src.llm.provider import LLMProvider
from src.models.schemas import (
    AgentTraceEntry,
    AnalysisRequest,
    AnalysisResponse,
    DecisionResponse,
    HealthResponse,
)
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.policy_ingestion import ingest_policy, load_chunks

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
POLICY_PDF = Path(os.environ.get("POLICY_PDF", "data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"))
CHUNKS_PATH = Path(os.environ.get("POLICY_CHUNKS", "data/policy/policy_chunks.json"))

VERSION = "0.1.0"
MAX_REQUEST_BYTES = int(os.environ.get("MAX_REQUEST_BYTES", str(1024 * 1024)))
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("CORS_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]


# Global singletons populated during lifespan
_orchestrator: Optional[AgentOrchestrator] = None
_chunk_count = 0
_health: HealthResponse | None = None


def _build_orchestrator() -> AgentOrchestrator:
    """Build the search index + agent orchestrator (lazy singleton)."""
    global _orchestrator, _chunk_count
    if _orchestrator is not None:
        return _orchestrator

    # 1. Chunks
    chunks = None
    if CHUNKS_PATH.exists():
        try:
            chunks = load_chunks(CHUNKS_PATH)
        except Exception:
            chunks = None
    if chunks is None:
        raw = ingest_policy(POLICY_PDF, out_path=CHUNKS_PATH, force=False)
        chunks = load_chunks(CHUNKS_PATH)

    _chunk_count = len(chunks)

    # 2. Embeddings + hybrid search
    rerank_enabled = os.environ.get("RERANK_ENABLED", "true").lower() == "true"
    emb = EmbeddingService(rerank_enabled=rerank_enabled)
    search = HybridSearch(chunks, emb)

    # 3. LLM provider (falls back to heuristic if no key)
    llm = LLMProvider(provider=os.environ.get("CLAIM_LLM_PROVIDER", "auto"))

    _orchestrator = AgentOrchestrator(search_engine=search, llm_provider=llm)
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Claim-Engine starting (version %s)", VERSION)
    yield
    logger.info("Claim-Engine stopped")


app = FastAPI(
    title="Policy-Aware Multi-Agent RAG Claim Decision Engine",
    version=VERSION,
    description=(
        "Analyzes health-insurance claim cases against a policy document "
        "using hybrid retrieval and a genuine multi-agent workflow. "
        "Local LLM fallback is used when no API key is configured."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_size_guard(request: Request, call_next):
    """Reject oversized requests before they reach Pydantic or the pipeline."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body is too large"},
                )
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length header"},
            )
    return await call_next(request)


@app.get("/")
def root():
    return {
        "service": "Policy-Aware Multi-Agent RAG Claim Decision Engine",
        "version": VERSION,
        "endpoints": [{"method": "POST", "path": "/analyze"}, {"method": "GET", "path": "/health"}],
    }


@app.get("/health")
def health() -> HealthResponse:
    """Lightweight liveness/readiness check without loading ML models."""
    chunks_indexed = 0
    status = "ok"
    if CHUNKS_PATH.exists():
        try:
            with open(CHUNKS_PATH, encoding="utf-8") as fh:
                chunks_indexed = len(json.load(fh))
        except Exception as exc:
            status = f"degraded: invalid policy index ({exc})"
    else:
        status = "degraded: policy index not found"

    provider = os.environ.get("CLAIM_LLM_PROVIDER", "auto")
    model = os.environ.get("OPENAI_MODEL", "heuristic") if provider != "groq" else "groq"
    return HealthResponse(
        status=status,
        model=model,
        chunks_indexed=chunks_indexed,
        version=VERSION,
    )


@app.post("/analyze")
def analyze(payload: AnalysisRequest) -> AnalysisResponse:
    """Analyze a single claim case."""
    t0 = time.time()
    orch = _build_orchestrator()

    # Pydantic already validated the payload; extra unknown fields are tolerated.
    result: DecisionResponse = orch.analyze(payload.case)

    # Add wall-clock timing to the tail of the trace for transparency.
    result.trace.append(AgentTraceEntry(
        agent="API",
        action="total",
        detail=f"total wall time {(time.time()-t0)*1000:.0f} ms",
    ))
    return AnalysisResponse(result=result)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid claim case payload", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )