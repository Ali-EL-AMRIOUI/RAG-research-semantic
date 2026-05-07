# Semantic RAG - Advanced Document Search and Retrieval

## Overview

Semantic RAG is a production-ready Retrieval-Augmented Generation system that enables intelligent semantic search and question answering across multiple document formats. The system combines state-of-the-art vector embeddings, multimodal understanding (text + images), and large language models to provide accurate, source-cited responses.

## Key Features

- Multi-format document support (PDF, DOCX, TXT, MD, RTF, HTML, JPG, PNG, TIFF)
- Multimodal search with CLIP embeddings (text and images)
- OCR extraction for images and scanned documents
- Hybrid search (vector + keyword) with Qdrant
- LLM integration with Groq (Llama 3.3)
- Reranking for improved result relevance
- FastAPI backend with async support
- Next.js frontend with modern UI
- Docker containerization with GPU support
- Full-text search with result citations

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python 3.10 |
| Vector Database | Qdrant |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2), CLIP (ViT-B/32) |
| LLM | Groq (Llama 3.3 70B) |
| OCR | Tesseract, Docling |
| Frontend | Next.js, TypeScript, Tailwind CSS |
| Deployment | Docker, Docker Compose |
| Monitoring | Prometheus, Grafana |

## Architecture
Document Upload
↓
Document Loader (Docling + Unstructured + Tesseract)
↓
Chunking (Intelligent with title preservation)
↓
Embeddings (Text: MiniLM / Images: CLIP)
↓
Vector Storage (Qdrant - hybrid search)
↓
Query Processing
↓
Retrieval + Reranking
↓
LLM Generation (Groq)
↓
Response with Citations


## Project Structure
semantic-search-rag/
├── .gitignore # Git ignore rules
├── .dockerignore # Docker ignore rules
├── README.md # Project documentation
├── docker-compose.yml # Docker Compose configuration
├── git_push.sh # Git automation script
├── backend/
│ ├── Dockerfile # Backend container
│ ├── requirements.txt # Python dependencies
│ ├── test_rag.py # Test script
│ ├── app/
│ │ ├── api/ # API endpoints
│ │ ├── core/ # Configuration
│ │ ├── db/ # Qdrant vector store
│ │ ├── services/ # RAG, embedding, reranking
│ │ ├── utils/ # Document loader, chunking
│ │ └── main.py # FastAPI entry point
│ └── data/ # Data directory
├── frontend/
│ ├── Dockerfile # Frontend container
│ ├── package.json # Node dependencies
│ ├── next.config.ts # Next.js configuration
│ ├── tailwind.config.js # Tailwind CSS config
│ ├── tsconfig.json # TypeScript config
│ ├── app/ # Next.js pages
│ ├── lib/ # API client
│ └── public/ # Static assets
└── kubernetes/
├── backend.yaml # K8s backend deployment
├── frontend.yaml # K8s frontend deployment
├── ingress.yaml # K8s ingress
├── namespace.yaml # K8s namespace
├── qdrant.yaml # K8s Qdrant deployment
└── secret.yaml # K8s secrets


## Supported File Formats

| Category | Formats |
|----------|---------|
| Documents | PDF, DOCX, TXT, MD, RTF, HTML, ODT |
| Spreadsheets | XLS, XLSX, CSV |
| Presentations | PPT, PPTX |
| Images | JPG, JPEG, PNG, GIF, BMP, TIFF |
| Data | JSON, YAML |

## Installation

### Local Development

# Clone repository
git clone https://github.com/Ali-EL-AMRIOUI/RAG-research-semantic.git
cd RAG-research-semantic

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev

# Qdrant (Docker)
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
Docker Deployment
bash
docker-compose up --build
API Endpoints
Method	Endpoint	Description
POST	/upload	Upload document (PDF, image, text)
POST	/ask	Ask question with RAG
GET	/documents	List indexed documents
DELETE	/documents	Delete document
GET	/health	Health check
GET	/stats	Collection statistics
Usage Examples
Upload Document
bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
Ask Question
bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
Response Format
json
{
  "answer": "This document discusses...",
  "source_documents": [
    {
      "content": "Extracted text chunk...",
      "filename": "document.pdf",
      "page": 5,
      "score": 0.89
    }
  ],
  "processing_time_ms": 1250
}


Performance Metrics

Metric	Value
Embedding dimension	384 (text), 512 (image)
Chunk size	1000 characters
Chunk overlap	200 characters
Top K retrieval	10 chunks
Top N for response	3 chunks
Response time	1-3 seconds (typical)
Environment Variables
env
GROQ_API_KEY=your_groq_api_key
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=documents
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=llama-3.3-70b-versatile
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K=10
TOP_N=3

# Future Improvements
Add support for more vector databases (Weaviate, Pinecone)

Implement multi-modal reranking

Add user feedback loop for continuous improvement

Deploy to cloud (AWS, GCP, Azure)

Add authentication and rate limiting

Implement document versioning

License
MIT License

Author
ALI EL AMRIOUI - GitHub: https://github.com/Ali-EL-AMRIOUI