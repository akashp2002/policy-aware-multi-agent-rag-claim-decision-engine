"""Shared fixtures: build the search pipeline once per test session."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agents.orchestrator import AgentOrchestrator
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.policy_ingestion import load_chunks

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def _pipeline():
    chunks = load_chunks()
    search = HybridSearch(chunks, EmbeddingService())
    orchestrator = AgentOrchestrator(search_engine=search)
    return chunks, search, orchestrator


@pytest.fixture(scope="session")
def chunks(_pipeline):
    return _pipeline[0]


@pytest.fixture(scope="session")
def search(_pipeline):
    return _pipeline[1]


@pytest.fixture(scope="session")
def orchestrator(_pipeline):
    return _pipeline[2]


def load_case(case_id: str) -> dict:
    for rel in ("data/test_cases/public_test_cases.json", "custom_cases/candidate_test_cases.json"):
        path = ROOT / rel
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            for c in json.load(fh):
                if c.get("case_id") == case_id:
                    return c
    raise FileNotFoundError(case_id)