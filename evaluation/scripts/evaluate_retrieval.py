"""
Evaluates retrieval quality (search only, no LLM) against the question set
in evaluation/dataset/questions.json and evaluation/dataset/expected_answers.json.

Requires: Qdrant running, the referenced documents already indexed. Could
not be executed end-to-end while writing this (no live Qdrant available) —
syntax and JSON validity were checked, not the actual retrieval calls. Run
it in your real environment.

Usage:
    cd backend
    python ../evaluation/scripts/evaluate_retrieval.py
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.services.retrieval_service import RetrievalService  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).parent.parent / "dataset" / "questions.json"
EXPECTED_ANSWERS_PATH = Path(__file__).parent.parent / "dataset" / "expected_answers.json"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

TOP_K = 5


# --- Metrics (pure functions, no I/O — kept inline rather than in a
# separate metrics.py, to match this project's existing file list) ---

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & set(relevant)) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for r in top_k if r in relevant_set) / len(top_k)


def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    relevant_set = set(relevant)
    for i, r in enumerate(retrieved, start=1):
        if r in relevant_set:
            return 1.0 / i
    return 0.0


def hit_rate_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    return 1.0 if (set(retrieved[:k]) & set(relevant)) else 0.0


def aggregate_metrics(per_query_results: List[Dict[str, List[str]]], k: int) -> Dict[str, Any]:
    n = len(per_query_results)
    if n == 0:
        return {"num_queries": 0, "k": k}
    return {
        "recall_at_k": sum(recall_at_k(r["retrieved"], r["relevant"], k) for r in per_query_results) / n,
        "precision_at_k": sum(precision_at_k(r["retrieved"], r["relevant"], k) for r in per_query_results) / n,
        "mrr": sum(mean_reciprocal_rank(r["retrieved"], r["relevant"]) for r in per_query_results) / n,
        "hit_rate_at_k": sum(hit_rate_at_k(r["retrieved"], r["relevant"], k) for r in per_query_results) / n,
        "k": k,
        "num_queries": n,
    }


# --- Evaluation runner ---

def load_questions() -> List[dict]:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_expected_answers() -> Dict[str, dict]:
    """Expected answers keyed by question id, so questions.json and
    expected_answers.json can evolve independently without duplicating
    question text in both files."""
    with open(EXPECTED_ANSWERS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {item["id"]: item for item in raw}


def run_evaluation(top_k: int = TOP_K) -> dict:
    questions = load_questions()
    expected = load_expected_answers()
    retrieval = RetrievalService()

    per_query_results = []
    details = []

    for q in questions:
        expected_item = expected.get(q["id"], {})
        relevant_filenames = expected_item.get("relevant_filenames", [])

        results = retrieval.search(q["question"], limit=top_k)
        retrieved_filenames = [r.get("metadata", {}).get("filename", "unknown") for r in results]

        per_query_results.append({"retrieved": retrieved_filenames, "relevant": relevant_filenames})
        details.append({
            "id": q["id"],
            "question": q["question"],
            "retrieved_filenames": retrieved_filenames,
            "relevant_filenames": relevant_filenames,
        })

    metrics = aggregate_metrics(per_query_results, k=top_k)
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    metrics["details"] = details
    return metrics


def write_report(metrics: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "history").mkdir(parents=True, exist_ok=True)

    slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / "history" / f"retrieval_{slug}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(REPORTS_DIR / "retrieval_latest.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return json_path


def main():
    logger.info(f"Loading questions: {QUESTIONS_PATH}")
    logger.info(f"Loading expected answers: {EXPECTED_ANSWERS_PATH}")
    metrics = run_evaluation()

    logger.info("=" * 50)
    logger.info("RETRIEVAL EVALUATION RESULTS")
    logger.info(f"  Recall@{metrics['k']}: {metrics['recall_at_k']:.3f}")
    logger.info(f"  Precision@{metrics['k']}: {metrics['precision_at_k']:.3f}")
    logger.info(f"  MRR: {metrics['mrr']:.3f}")
    logger.info(f"  Hit rate@{metrics['k']}: {metrics['hit_rate_at_k']:.3f}")
    logger.info(f"  Number of queries: {metrics['num_queries']}")
    logger.info("=" * 50)

    report_path = write_report(metrics)
    logger.info(f"Report written: {report_path}")


if __name__ == "__main__":
    main()