# Evaluation Methodology

`evaluation/scripts/` measures retrieval and generation quality against a fixed dataset (`evaluation/dataset/questions.json`, `expected_answers.json`).

## Retrieval (`evaluate_retrieval.py`)

For each question, retrieves top-k chunks and checks their source filenames against the expected relevant filename.

| Metric | Meaning |
|---|---|
| `recall_at_k` | Fraction of questions where a relevant document appeared in the top k |
| `precision_at_k` | Fraction of top-k results that were actually relevant, averaged |
| `mrr` | Mean Reciprocal Rank — rewards relevant results ranked higher |
| `hit_rate_at_k` | Same as recall_at_k under the current single-relevant-document setup |

## Generation (`evaluate_generation.py`)

Compares generated answers against `expected_answers.json`. Current method is keyword/string-based, not an LLM judge — see Future Improvements in the root README for that option.

## Running

```bash
cd backend
python ../evaluation/scripts/evaluate_retrieval.py
python ../evaluation/scripts/evaluate_generation.py
```

Writes `evaluation/reports/{retrieval,generation}_latest.json` plus a timestamped copy under `evaluation/reports/history/`.

## Caveat

These scripts query whichever collection `QDRANT_COLLECTION_NAME` currently points to. If indexed documents have changed since `expected_answers.json` was written (new uploads, a `/clear`, a different `EMBEDDING_MODEL`), results won't be comparable to a previously committed `*_latest.json` and may show near-zero recall simply because the expected documents aren't indexed anymore. Re-ingest the documents referenced in `expected_answers.json` before treating a run as a regression check.
