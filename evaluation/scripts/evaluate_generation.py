"""
Evaluates generation quality (full RAG: retrieval + LLM) against the
question set in evaluation/dataset/questions.json and
evaluation/dataset/expected_answers.json, via a simple keyword-match count
on the generated answer — no LLM-judge, to avoid adding API cost/latency
to the evaluation itself.

Requires: Qdrant running, documents indexed, valid GROQ_API_KEY. Could not
be executed end-to-end while writing this (no live services available) —
syntax and JSON validity were checked, not the actual RAG calls. Run it in
your real environment.

Usage:
    cd backend
    python ../evaluation/scripts/evaluate_generation.py
"""
import sys
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.services.rag_service import IndustrialRAGService  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).parent.parent / "dataset" / "questions.json"
EXPECTED_ANSWERS_PATH = Path(__file__).parent.parent / "dataset" / "expected_answers.json"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def keyword_match_rate(answer: str, expected_keywords: List[str]) -> float:
    """Proportion of expected keywords present in the generated answer
    (case-insensitive). Deliberately simple — catches gross regressions
    (RAG stopped finding the right context) without an LLM-judge call."""
    if not expected_keywords:
        return 0.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return found / len(expected_keywords)


def load_questions() -> List[dict]:
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_expected_answers() -> Dict[str, dict]:
    with open(EXPECTED_ANSWERS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {item["id"]: item for item in raw}


def run_evaluation() -> dict:
    questions = load_questions()
    expected = load_expected_answers()
    rag = IndustrialRAGService()

    details = []
    total_rate = 0.0

    for q in questions:
        expected_item = expected.get(q["id"], {})
        expected_keywords = expected_item.get("expected_keywords", [])

        result = rag.run(q["question"])
        answer = result.get("answer", "")
        rate = keyword_match_rate(answer, expected_keywords)
        total_rate += rate

        details.append({
            "id": q["id"],
            "question": q["question"],
            "answer": answer,
            "expected_keywords": expected_keywords,
            "keyword_match_rate": rate,
            "num_sources": len(result.get("source_documents", [])),
        })

    n = len(questions)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_queries": n,
        "avg_keyword_match_rate": total_rate / n if n else 0.0,
        "details": details,
    }


def write_report(results: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "history").mkdir(parents=True, exist_ok=True)

    slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / "history" / f"generation_{slug}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    with open(REPORTS_DIR / "generation_latest.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return json_path


def main():
    logger.info(f"Loading questions: {QUESTIONS_PATH}")
    logger.info(f"Loading expected answers: {EXPECTED_ANSWERS_PATH}")
    results = run_evaluation()

    logger.info("=" * 50)
    logger.info("GENERATION EVALUATION RESULTS")
    logger.info(f"  Avg keyword match rate: {results['avg_keyword_match_rate']:.1%}")
    logger.info(f"  Number of queries: {results['num_queries']}")
    logger.info("=" * 50)

    report_path = write_report(results)
    logger.info(f"Report written: {report_path}")


if __name__ == "__main__":
    main()