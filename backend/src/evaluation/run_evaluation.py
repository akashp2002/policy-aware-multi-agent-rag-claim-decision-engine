"""Evaluation harness.

Runs the full multi-agent pipeline over the public test cases and the
candidate-created cases, then reports:

  * decision accuracy (exact match to ground-truth labels)
  * validation pass rate (citation-grounding consistency)
  * citation recall@k against curated gold clauses
  * citation coverage (every case returns at least one grounded citation)
  * mean confidence and latency
  * a confusion matrix and per-case breakdown

Outputs ``output/evaluation_report.json`` and ``output/evaluation_report.md``.

Usage:
    python -m src.evaluation.run_evaluation
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

from src.agents.orchestrator import AgentOrchestrator
from src.evaluation.expected_outcomes import (
    CUSTOM_EXPECTED,
    GOLD_CHUNKS,
    PUBLIC_EXPECTED,
)
from src.evaluation.metrics import (
    citation_hit_rate,
    confusion_matrix,
    decision_accuracy,
    mean,
    validation_pass_rate,
)
from src.models.schemas import ClaimCase
from src.retrieval.embeddings import EmbeddingService
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.policy_ingestion import load_chunks

logging.basicConfig(level=logging.WARNING)

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_CASES = ROOT / "data" / "test_cases" / "public_test_cases.json"
CUSTOM_CASES = ROOT / "custom_cases" / "candidate_test_cases.json"
OUTPUT_DIR = ROOT / "output"


def _load(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run(split: str = "all") -> dict:
    chunks = load_chunks()
    embeddings = EmbeddingService()
    search = HybridSearch(chunks, embeddings)
    orchestrator = AgentOrchestrator(search_engine=search)

    datasets: List[Tuple[str, List[dict], Dict[str, str]]] = []
    if split in ("all", "public"):
        datasets.append(("public", _load(PUBLIC_CASES), PUBLIC_EXPECTED))
    if split in ("all", "custom"):
        datasets.append(("custom", _load(CUSTOM_CASES), CUSTOM_EXPECTED))

    per_case: List[dict] = []
    for split_name, cases, expected_map in datasets:
        for raw in cases:
            case_id = raw.get("case_id", "?")
            t0 = time.perf_counter()
            result = orchestrator.analyze(ClaimCase(**raw))
            elapsed = time.perf_counter() - t0

            expected = expected_map.get(case_id, "")
            cited_ids = [c.chunk_id for c in result.citations]
            gold = GOLD_CHUNKS.get(case_id, [])
            recall = citation_hit_rate(gold, cited_ids)

            per_case.append({
                "split": split_name,
                "case_id": case_id,
                "expected": expected,
                "predicted": result.decision.value,
                "correct": expected == result.decision.value,
                "confidence": result.confidence,
                "validation": result.validation.status.value,
                "num_citations": len(result.citations),
                "num_limits": len(result.applicable_limits),
                "citation_recall": round(recall, 3),
                "gold_chunks": gold,
                "cited_chunks": cited_ids,
                "latency_s": round(elapsed, 2),
                "key_findings": result.key_findings,
            })

    expected_list = [c["expected"] for c in per_case]
    predicted_list = [c["predicted"] for c in per_case]

    report = {
        "summary": {
            "num_cases": len(per_case),
            "decision_accuracy": round(decision_accuracy(expected_list, predicted_list), 3),
            "validation_pass_rate": round(
                validation_pass_rate([c["validation"] for c in per_case]), 3
            ),
            "mean_citation_recall": round(mean([c["citation_recall"] for c in per_case]), 3),
            "citation_coverage": round(
                sum(1 for c in per_case if c["num_citations"] > 0) / len(per_case), 3
            ),
            "mean_confidence": round(mean([c["confidence"] for c in per_case]), 3),
            "mean_latency_s": round(mean([c["latency_s"] for c in per_case]), 2),
            "num_correct": sum(1 for c in per_case if c["correct"]),
        },
        "confusion_matrix": confusion_matrix(expected_list, predicted_list),
        "cases": per_case,
        "failures": [c["case_id"] for c in per_case if not c["correct"]],
    }
    return report


def _write_reports(report: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "evaluation_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    s = report["summary"]
    lines = [
        "# Evaluation Report",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Cases evaluated | {s['num_cases']} |",
        f"| Decision accuracy | {s['decision_accuracy']:.1%} |",
        f"| Validation pass rate | {s['validation_pass_rate']:.1%} |",
        f"| Mean citation recall@k (curated gold clauses) | {s['mean_citation_recall']:.1%} |",
        f"| Citation coverage (>=1 grounded citation) | {s['citation_coverage']:.1%} |",
        f"| Mean confidence | {s['mean_confidence']:.2f} |",
        f"| Mean latency | {s['mean_latency_s']:.2f}s |",
        f"| Correct / total | {s['num_correct']}/{s['num_cases']} |",
        "",
        "## Per-case results",
        "",
        "| Case | Split | Expected | Predicted | OK | Conf | Validation | Recall | Cites | Limits |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in report["cases"]:
        lines.append(
            f"| {c['case_id']} | {c['split']} | {c['expected']} | {c['predicted']} | "
            f"{'yes' if c['correct'] else 'NO'} | {c['confidence']:.2f} | {c['validation']} | "
            f"{c['citation_recall']:.0%} | {c['num_citations']} | {c['num_limits']} |"
        )
    if report["failures"]:
        lines += ["", f"**Failures:** {', '.join(report['failures'])}"]

    with open(OUTPUT_DIR / "evaluation_report.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the evaluation harness")
    parser.add_argument("--split", choices=["all", "public", "custom"], default="all")
    args = parser.parse_args()

    report = run(split=args.split)
    _write_reports(report)

    s = report["summary"]
    print(f"Cases            : {s['num_cases']}")
    print(f"Decision accuracy: {s['decision_accuracy']:.1%} ({s['num_correct']}/{s['num_cases']})")
    print(f"Validation PASS  : {s['validation_pass_rate']:.1%}")
    print(f"Citation recall@k: {s['mean_citation_recall']:.1%}")
    print(f"Mean confidence  : {s['mean_confidence']:.2f}")
    print(f"Mean latency     : {s['mean_latency_s']:.2f}s")
    if report["failures"]:
        print("Failures         : " + ", ".join(report["failures"]))
    print("Reports written to output/evaluation_report.{json,md}")


if __name__ == "__main__":
    main()
