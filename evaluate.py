"""Evaluate retrieval quality and export CSV/JSON reports."""

import csv
import json
import statistics
import time

from config import PROJECT_ROOT
from retriever import Retriever


def evaluate() -> dict[str, float | int]:
    """Run retrieval test cases and write report artifacts."""
    cases = json.loads(
        (PROJECT_ROOT / "data" / "test_questions.json").read_text(encoding="utf-8")
    )
    retriever = Retriever()
    rows = []
    for case in cases:
        started = time.perf_counter()
        matches, rejected = retriever.retrieve(case["question"])
        latency = (time.perf_counter() - started) * 1000
        ids = [item["id"] for item in matches]
        expected = case.get("expected_faq_id")
        rows.append(
            {
                "test_id": case["id"],
                "question": case["question"],
                "type": case["type"],
                "expected_faq_id": expected,
                "predicted_faq_id": None if rejected else ids[0],
                "top1_score": round(matches[0]["score"], 4),
                "hit_at_1": expected is not None and ids[0] == expected,
                "hit_at_3": expected is not None and expected in ids[:3],
                "expected_reject": case["should_reject"],
                "actual_reject": rejected,
                "latency_ms": round(latency, 2),
            }
        )
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with (reports / "test_results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    positive = [row for row in rows if row["expected_faq_id"] is not None]
    correct_scores = [
        row["top1_score"] for row in positive if row["hit_at_1"]
    ]
    summary = {
        "total_tests": len(rows),
        "hit_at_1": sum(row["hit_at_1"] for row in positive) / len(positive),
        "hit_at_3": sum(row["hit_at_3"] for row in positive) / len(positive),
        "correct_rejects": sum(
            row["expected_reject"] and row["actual_reject"] for row in rows
        ),
        "false_rejects": sum(
            not row["expected_reject"] and row["actual_reject"] for row in rows
        ),
        "reject_accuracy": sum(
            row["expected_reject"] == row["actual_reject"] for row in rows
        )
        / len(rows),
        "average_latency_ms": statistics.fmean(row["latency_ms"] for row in rows),
        "average_correct_similarity": (
            statistics.fmean(correct_scores) if correct_scores else 0.0
        ),
    }
    (reports / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    evaluate()
