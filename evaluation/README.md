# Evaluation

Measures RAG quality instead of relying on "it looks like it works".

## Structure

- `dataset/questions.json` — question text only, keyed by `id`.
- `dataset/expected_answers.json` — expected relevant documents (for
  retrieval) and expected keywords (for generation), keyed by the same
  `id`. Kept separate from `questions.json` so ground-truth answers can be
  reviewed/edited independently from the question list.
- `dataset/documents/` — reference copies of the documents used to build
  the questions (not indexed automatically — index them via
  `backend/scripts/ingest_data.py` as usual).
- `scripts/evaluate_retrieval.py` — runs `RetrievalService.search()` for
  each question, computes recall@k, precision@k, MRR, hit rate@k.
- `scripts/evaluate_generation.py` — runs `IndustrialRAGService.run()` for
  each question, computes `keyword_match_rate` on the generated answer.
- `reports/` — `retrieval_latest.json` / `generation_latest.json` (most
  recent run) and `reports/history/` (timestamped runs, to track quality
  over time after a change — e.g. switching to BGE-M3, enabling hybrid
  search).

## Why keyword matching instead of an LLM-judge

An LLM-judge (asking a model to score the answer) is more nuanced, but
adds an API call, cost, and latency to every evaluation run — and
introduces its own margin of error. A keyword-match count is cruder but
sufficient to catch a gross regression (the RAG stopped finding the right
context at all), with no extra dependency.

## Prerequisites to run the scripts

- Qdrant running (`docker start qdrant`)
- The documents referenced in `expected_answers.json` already indexed
- `GROQ_API_KEY` valid in `.env` (for `evaluate_generation.py` only)

Neither script could be run end-to-end while being written — only syntax
and JSON validity were checked (no live Qdrant available in that
environment). Run them for real here:

```bash
cd backend
python ../evaluation/scripts/evaluate_retrieval.py
python ../evaluation/scripts/evaluate_generation.py
```