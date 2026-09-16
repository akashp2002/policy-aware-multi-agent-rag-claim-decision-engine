"""API smoke tests (FastAPI TestClient, in-process)."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app

ROOT = Path(__file__).resolve().parents[1]

client = TestClient(app)


def test_root_and_health():
    assert client.get("/").status_code == 200
    assert client.get("/health").status_code == 200


def test_analyze_returns_structured_response():
    with open(ROOT / "custom_cases" / "candidate_test_cases.json", encoding="utf-8") as fh:
        case = json.load(fh)[0]
    resp = client.post("/analyze", json={"case": case})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert {"decision", "confidence", "key_findings", "citations", "validation", "trace"} <= set(result)
    assert result["validation"]["status"] in ("PASS", "FAIL")
    assert isinstance(result["citations"], list)


def test_analyze_rejects_bad_payload():
    case = {"case_id": "BAD", "sum_insured_inr": "NaN-ish"}
    resp = client.post("/analyze", json={"case": case})
    assert resp.status_code == 422


def test_rejects_oversized_request():
    resp = client.post(
        "/analyze",
        content=b"{}",
        headers={"Content-Length": str(1024 * 1024 + 1)},
    )
    assert resp.status_code == 413