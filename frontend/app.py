"""Streamlit frontend for the Policy-Aware Multi-Agent RAG Claim Decision Engine.

Two interaction modes:
  1. Pick a bundled test case (public or candidate-created).
  2. Paste a claim case as JSON.

The app calls the FastAPI backend (default http://localhost:8000). If the
backend is unreachable it falls back to loading the pipeline in-process
(heavier startup, but self-contained for quick local demos).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import requests
import streamlit as st

# Path to the backend package (relative to this file's location)
FRONTEND_DIR = Path(__file__).resolve().parents[0]  # frontend/
REPO_ROOT    = FRONTEND_DIR.parent                  # repo root
BACKEND_DIR  = REPO_ROOT / "backend"
BACKEND_SRC  = BACKEND_DIR / "src"

# Make backend imports available for the in-process fallback path
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

def _configured_api_base() -> str:
    """Read the API URL from hosting secrets, environment, or local default."""
    try:
        secret_value = st.secrets.get("CLAIM_API_BASE")
    except Exception:
        secret_value = None
    return (secret_value or os.environ.get("CLAIM_API_BASE") or "http://localhost:8000").rstrip("/")


API_BASE = _configured_api_base()
ALLOW_IN_PROCESS_FALLBACK = (
    os.environ.get("ALLOW_IN_PROCESS_FALLBACK", "true").lower() == "true"
)

st.set_page_config(page_title="Claim Decision Engine", page_icon=":clipboard:", layout="wide")


def load_bundled_cases() -> dict[str, dict]:
    cases: dict[str, dict] = {}
    paths = [
        BACKEND_DIR / "data" / "test_cases" / "public_test_cases.json",
        BACKEND_DIR / "custom_cases" / "candidate_test_cases.json",
    ]
    for path in paths:
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                for c in json.load(fh):
                    cases[c.get("case_id", str(len(cases)))] = c
    return cases


def run_via_api(case: dict) -> dict:
    resp = requests.post(f"{API_BASE}/analyze", json={"case": case}, timeout=600)
    resp.raise_for_status()
    return resp.json()["result"]


def _build_orchestrator():
    from src.agents.orchestrator import AgentOrchestrator
    from src.retrieval.embeddings import EmbeddingService
    from src.retrieval.hybrid_search import HybridSearch
    from src.retrieval.policy_ingestion import load_chunks

    chunks = load_chunks()
    return AgentOrchestrator(search_engine=HybridSearch(chunks, EmbeddingService()))


@st.cache_resource
def get_orchestrator():
    return _build_orchestrator()


def run_via_process(case: dict) -> dict:
    from src.models.schemas import ClaimCase

    orch = get_orchestrator()
    result = orch.analyze(ClaimCase(**case))
    return result.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Sidebar: input selection
# ---------------------------------------------------------------------------
bundled = load_bundled_cases()

st.sidebar.header("Input")
mode = st.sidebar.radio("Source", ["Bundled test case", "Paste JSON"], index=0)
use_api = st.sidebar.toggle("Use FastAPI backend", value=API_BASE != "")

case: Optional[dict] = None
if mode == "Bundled test case":
    label = st.sidebar.selectbox("Case", sorted(bundled.keys()))
    case = bundled[label]
    if case.get("task"):
        st.sidebar.caption(case["task"])
else:
    raw = st.sidebar.text_area("Claim case (JSON)", height=240)
    if raw.strip():
        try:
            case = json.loads(raw)
        except Exception as exc:
            st.sidebar.error(f"Invalid JSON: {exc}")

if case is None:
    st.warning("Select a case or paste valid JSON to begin.")
    st.stop()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
st.title("Policy-Aware Multi-Agent RAG Claim Decision Engine")
st.caption(f"Case **{case.get('case_id', 'custom')}** · policy "
           f"{case.get('policy_id', '-')} · validated by the structured-state pipeline")

with st.expander("Claim case input", expanded=False):
    st.json(case)

if st.button("Analyze claim", type="primary"):
    backend = "FastAPI" if use_api else "in-process"
    with st.status(f"Running {backend} pipeline…", expanded=False) as status:
        try:
            result = run_via_api(case) if use_api else run_via_process(case)
            status.update(label="Analysis complete", state="complete")
        except Exception:
            if use_api and ALLOW_IN_PROCESS_FALLBACK:
                st.info("API unreachable — falling back to in-process pipeline.")
                status.update(label="Running in-process fallback", state="running")
                try:
                    result = run_via_process(case)
                    status.update(label="Analysis complete (in-process fallback)", state="complete")
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")
                    st.stop()
            elif use_api:
                st.error(
                    f"Could not reach the configured API at {API_BASE}. "
                    "Check CLAIM_API_BASE in the hosting settings."
                )
                st.stop()
            else:
                st.error("Pipeline error.")
                raise

    # Decision banner
    decision = result["decision"]
    conf = result["confidence"]
    colors = {
        "ADMISSIBLE": "green",
        "ADMISSIBLE_WITH_LIMITS": "green",
        "PARTIALLY_ADMISSIBLE": "orange",
        "NEEDS_REVIEW": "orange",
        "NOT_ADMISSIBLE": "red",
    }
    st.markdown(
        f"### :{colors.get(decision, 'blue')}[ {decision} ]  ·  "
        f"confidence **{conf:.0%}**"
    )
    if decision == "NEEDS_REVIEW":
        st.warning(
            "Abstained: the available claim or policy evidence is insufficient "
            "for a safe final decision."
        )
    st.progress(int(conf * 100))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key findings")
        for f in result.get("key_findings", []):
            st.markdown(f"- {f}")

        lims = result.get("applicable_limits", [])
        if lims:
            st.subheader("Applicable limits / deductions")
            for l in lims:
                name = l.get("name") or l.get("limit_name") or "limit"
                amount = l.get("amount") or l.get("value") or l.get("limit")
                note = l.get("note") or ""
                st.markdown(f"- **{name}**: {amount} {f'— {note}' if note else ''}")

    with col2:
        st.subheader("Missing evidence")
        miss = result.get("missing_evidence", [])
        if miss:
            for m in miss:
                st.markdown(f"- {m}")
        else:
            st.success("None — evidence sufficient.")

        v = result.get("validation", {})
        st.subheader("Validation")
        st.metric("Status", v.get("status", "-"))
        unsupported = v.get("unsupported_claims") or []
        for u in unsupported:
            st.caption(f":red[Unsupported:] {u.get('claim')} — {u.get('reason')}")

    # Citations
    st.subheader("Policy citations")
    citing = result.get("citations", [])
    if citing:
        for index, c in enumerate(citing, start=1):
            sec = c.get("section", "")
            page = c.get("page")
            with st.expander(
                f"{index}. {c.get('claim', '')} · {sec} (p{page})"
            ):
                st.caption(
                    f"Chunk: {c.get('chunk_id', '')} · "
                    f"Source: {c.get('source', 'policy.pdf')}"
                )
                excerpt = c.get("excerpt")
                if excerpt:
                    st.markdown(f"> {excerpt}")
                else:
                    st.info("No policy excerpt returned by the backend.")
    else:
        st.info("No citations returned.")

    # Trace
    with st.expander("Agent trace"):
        for t in result.get("trace", []):
            dt = f"{t.get('elapse_ms', 0):.0f} ms"
            st.markdown(f"- **{t.get('agent')}** · {dt} · `{t.get('action')}`")
            if t.get("input_summary"):
                st.caption(f"Input: {t['input_summary']}")
            if t.get("detail"):
                st.caption(f"Action: {t['detail']}")
            if t.get("output_summary"):
                st.caption(f"Output: {t['output_summary']}")

    policy_meta = result.get("policy", {})
    if policy_meta:
        with st.expander("Pipeline metadata"):
            st.json(policy_meta)

st.divider()
st.caption("Deterministic heuristic fallback is active when no LLM API key is configured.")