# API Reference

Base URL (local): `http://localhost:8000`

🔒 = requires `X-API-Key` header when `API_KEY` is set server-side (unset by default for local dev).

## GET /health
No auth. Checks real Qdrant connectivity, not just process liveness.
```json
{ "status": "ok", "rag_service": true, "qdrant": true }
```

## 🔒 POST /upload
`multipart/form-data`, field `file`. One file per request (the frontend loops client-side for multi-file upload). Accepted: `.pdf .txt .md .markdown .doc .docx .odt .xls .xlsx .csv .ppt .pptx .html .htm .xml .jpg .jpeg .png .gif .bmp .tiff .json .yaml .yml .rtf .epub`. Max 100 MB.
```json
{ "message": "Success", "filename": "doc.pdf", "chunks": 6 }
```

## GET /documents
No auth. All indexed documents (text + image collections combined), with chunk counts.
```json
{ "documents": [{ "name": "doc.pdf", "chunks": 6 }] }
```

## 🔒 DELETE /documents
```json
{ "filename": "doc.pdf" }
```
Removes all chunks for that filename from both collections.

## 🔒 POST /ask
```json
{
  "question": "Quelle est la date d'expiration ?",
  "history": [
    { "role": "user", "content": "Resume ce document" },
    { "role": "assistant", "content": "..." }
  ]
}
```
`history` optional, defaults to `[]`; only the last 6 turns are used by the LLM. `role` is `"user"` or `"assistant"`.
```json
{
  "answer": "...",
  "source_documents": [
    { "content": "...", "filename": "doc.pdf", "page": 0, "score": 0.82 }
  ],
  "sources_details": [],
  "processing_time_ms": null
}
```
When nothing relevant is found, `answer` says so explicitly (in the question's language) and `source_documents` is empty.

## GET /stats
No auth. Point counts for both Qdrant collections.

## 🔒 POST /clear
```json
{ "confirm": true }
```
Wipes both collections. Required before switching to an `EMBEDDING_MODEL` with a different vector dimension.
