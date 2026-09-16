"""Batch-run all public cases and print decisions."""
import json
import time

from src.retrieval.policy_ingestion import load_chunks
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.hybrid_search import HybridSearch
from src.agents.orchestrator import AgentOrchestrator
from src.models.schemas import ClaimCase

chunks = load_chunks()
emb = EmbeddingService()
search = HybridSearch(chunks, emb)
orch = AgentOrchestrator(search_engine=search)

with open("data/test_cases/public_test_cases.json", encoding="utf-8") as f:
    cases = json.load(f)

t0 = time.time()
for raw in cases:
    case = ClaimCase(**raw)
    res = orch.analyze(case)
    print(f"{res.case_id}: {res.decision.value:26s} conf={res.confidence:.2f} "
          f"val={res.validation.status.value:4s} cites={len(res.citations)} "
          f"limits={len(res.applicable_limits)}")
    print(f"      missing: {res.missing_evidence[:3]}")
print(f"\nTotal time: {time.time()-t0:.1f}s for {len(cases)} cases")