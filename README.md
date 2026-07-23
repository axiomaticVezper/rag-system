# RAG System

A production-grade **Retrieval-Augmented Generation (RAG)** system built from scratch — no tutorial shortcuts. It ingests documents, chunks and indexes them into a vector database, retrieves the most relevant content using hybrid search, and generates grounded, cited answers using a local LLM. Access to documents is enforced at every layer via Role-Based Access Control (RBAC).

Built as a portfolio project targeting senior engineering roles at AI-first companies.

---

## Architecture

```
Documents (PDF / HTML / Markdown)
         │
         ▼
┌─────────────────────┐
│   Ingestion Layer   │  Loaders + RBAC access_level tagging
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Chunking Layer    │  Paragraph-aware semantic chunking (512 tokens, 50 overlap)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Embedding Layer   │  BAAI/bge-small-en-v1.5 (local, free, 384-dim)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Qdrant (Docker)   │  Vector store — access_level in payload for RBAC filtering
└────────┬────────────┘
         │
    Query time
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                 Hybrid Retrieval                     │
│  Dense search (Qdrant)  +  BM25 sparse search       │
│         └──────── RRF Fusion ────────┘              │
│              Cross-encoder Reranking                 │
│         (BAAI/bge-reranker-base, local)             │
└────────┬────────────────────────────────────────────┘
         │  RBAC filter enforced at every step
         ▼
┌─────────────────────┐
│  Generation Layer   │  Ollama llama3.1:8b — structured JSON output with citations
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌─────────────────────┐
│   FastAPI Backend   │────▶│  Streamlit Frontend  │
│  /ask  /ingest      │     │  browser UI + RBAC   │
└─────────────────────┘     └─────────────────────┘
```

---

## Tech Stack

| Component | Tool | Rationale |
|---|---|---|
| Language | Python 3.11+ | Strong AI/ML ecosystem |
| API | FastAPI | Async-ready, auto-generates OpenAPI docs |
| Vector store | Qdrant (Docker) | Native payload filtering for RBAC, production-scalable |
| Embeddings | BAAI/bge-small-en-v1.5 | Free, local, strong retrieval performance (384-dim) |
| Sparse search | BM25 via rank-bm25 | Catches exact keyword matches dense search misses |
| Reranker | BAAI/bge-reranker-base | Cross-encoder precision pass on fused candidates |
| Generation | Ollama llama3.1:8b | Free local LLM, structured JSON output via format mode |
| UI | Streamlit | Fast interactive demo frontend |
| Containerization | Docker Compose | One-command full stack startup |
| CI/CD | GitHub Actions | Runs 41 tests on every push to main |
| Testing | pytest + pytest-asyncio | Unit + integration tests across all layers |

---

## Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- [Ollama](https://ollama.com/download/windows) installed with `llama3.1:8b` pulled:
```bash
  ollama pull llama3.1:8b
```

### Run with Docker Compose
```bash
git clone https://github.com/axiomaticVezper/rag-system.git
cd rag-system
docker compose up --build
```

This starts:
- **Qdrant** on `http://localhost:6333`
- **FastAPI API** on `http://localhost:8000` (docs at `/docs`)
- **Streamlit UI** on `http://localhost:8501`

### Populate the vector database
Once containers are running, trigger ingestion via the UI ("Re-index documents" expander) or the API:
```bash
curl -X POST http://localhost:8000/ingest
```

### Ask a question
Open `http://localhost:8501`, select your clearance level, and ask away.

Or via the API:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the office hours?", "clearance": "public"}'
```

---

## Local Development

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker run -d -p 6333:6333 -v ./qdrant_storage:/qdrant/storage --name qdrant-rag qdrant/qdrant

# Index sample documents
python scripts/run_ingestion_pipeline.py

# Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Start the UI (separate terminal)
streamlit run app/streamlit_app.py

# Interactive CLI demo
python scripts/ask.py
```

---

## RBAC Design

Every document is tagged with an `access_level` at ingestion time:

| Level | Documents |
|---|---|
| `public` | General company info (employee handbook) |
| `internal` | Engineering wiki, internal processes |
| `confidential` | Financial reports, sensitive data |

Access levels are **cumulative** — `internal` clearance sees `public` + `internal` content; `confidential` sees everything.

RBAC is enforced at the **retrieval layer**, not the generation layer — confidential chunks are never sent to the LLM for a public-clearance user, regardless of semantic relevance. Test this:

```bash
# Public user — confidential finance data is blocked
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the Q3 revenue?", "clearance": "public"}'
# → confidence_score: 0.0

# Confidential user — same question, grounded answer
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What was the Q3 revenue?", "clearance": "confidential"}'
# → confidence_score: 1.0, citation: [0]
```

---

## Evaluation

Custom RAG evaluation metrics over 6 golden question/answer pairs (2 per access level):

| Metric | Score | Description |
|---|---|---|
| Faithfulness | 0.67 | Answer only contains claims supported by retrieved chunks |
| Answer Relevancy | 0.79 | Answer actually addresses what was asked |
| Contextual Recall | 0.79 | Retrieved chunks contain the information needed to answer |

*Scores measured with llama3.1:8b as judge. Raise faithfulness threshold to 0.7+ with a stronger judge (e.g. gpt-4o-mini).*

Run evals:
```bash
pytest tests/test_evaluation.py -v -s
```

---

## Running Tests

```bash
# Fast deterministic tests — no LLM required, runs in CI
pytest -v --ignore=tests/test_ollama_generator.py \
          --ignore=tests/test_rag_pipeline.py \
          --ignore=tests/test_evaluation.py

# Full suite (requires Qdrant + Ollama running locally)
pytest -v
```

41 deterministic tests run automatically on every push to `main` via GitHub Actions.

---

## Project Structure

```
rag-system/
├── app/
│   ├── core/           # Pydantic models (Document, Chunk, Answer, AccessLevel)
│   ├── ingestion/      # PDF, HTML, Markdown loaders + pipeline orchestrator
│   ├── chunking/       # Semantic chunker, local embedder, Qdrant indexer
│   ├── retrieval/      # Dense search, BM25, RRF fusion, cross-encoder reranker
│   ├── generation/     # Generator abstraction, OllamaGenerator, RAG pipeline
│   ├── evaluation/     # Faithfulness, relevancy, recall metrics
│   └── main.py         # FastAPI application
├── data/               # Sample documents (PDF, HTML, Markdown)
├── scripts/
│   ├── run_ingestion_pipeline.py
│   ├── ask.py          # Interactive CLI demo
│   └── golden_questions.json
├── tests/              # Full test suite (47 tests)
├── .github/workflows/  # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Switching to a Paid LLM

The `Generator` abstraction in `app/generation/base.py` makes this a one-file addition. To use `gpt-4o-mini`:

1. Create `app/generation/openai_generator.py` implementing `Generator`
2. Set `OPENAI_API_KEY` in `.env`
3. Swap the import in `app/generation/rag_pipeline.py`

No other code changes needed.

---

## Author

**Sahil** — [github.com/axiomaticVezper](https://github.com/axiomaticVezper)