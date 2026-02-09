# SageMind - Multimodal RAG for Research Papers

A production-ready Retrieval-Augmented Generation (RAG) system designed for academic research papers. Supports **text, images, and tables** with LLM-powered enrichment and hybrid search.

## Features

- **Multimodal Ingestion**: Extract and process text, images, and tables from PDFs
- **OCR Processing**: Mistral OCR for high-quality PDF-to-markdown conversion
- **LLM Enrichment**: GPT-4o-mini generates descriptions for images and tables
- **Parallel Processing**: ThreadPoolExecutor for concurrent LLM calls with rate limit handling
- **Hybrid Search**: Combines vector similarity (Pinecone) with structured data (PostgreSQL)
- **Conversational AI**: LangGraph-powered chat with context-aware responses
- **Observability**: LangSmith tracing integration

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INGESTION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│  PDF/arXiv URL                                                      │
│       │                                                             │
│       ▼                                                             │
│  [Mistral OCR] ──► Markdown + Images + Tables                      │
│       │                                                             │
│       ├──► [Chunker] ──► Semantic chunks ──► PostgreSQL            │
│       │                         │                                   │
│       │                         ▼                                   │
│       │              [OpenAI Embeddings] ──► Pinecone              │
│       │                                                             │
│       ├──► [Enricher] ──► Image descriptions ──► PostgreSQL        │
│       │    (parallel)                                               │
│       │                                                             │
│       └──► [Enricher] ──► Table descriptions ──► PostgreSQL        │
│            (parallel)                                               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          CHAT PIPELINE                              │
├─────────────────────────────────────────────────────────────────────┤
│  User Query                                                         │
│       │                                                             │
│       ▼                                                             │
│  [Hybrid Retriever]                                                 │
│       ├──► Pinecone (vector search)                                │
│       ├──► PostgreSQL (full content + images + tables)             │
│       │                                                             │
│       ▼                                                             │
│  [LangGraph Agent] ──► GPT-4o-mini ──► Response with citations     │
└─────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend Framework | FastAPI |
| OCR | Mistral AI |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small (1536 dim) |
| Vector Database | Pinecone |
| Relational Database | PostgreSQL |
| Orchestration | LangGraph |
| Observability | LangSmith |

## Project Structure

```
RAG-Application/
├── backend/
│   ├── app.py              # FastAPI entry point
│   ├── ingestion.py        # Orchestrates PDF ingestion
│   ├── ocr_processor.py    # Mistral OCR integration
│   ├── chunker.py          # Semantic text chunking
│   ├── enricher.py         # Parallel LLM enrichment for images/tables
│   ├── retriever.py        # Hybrid search (vector + SQL)
│   ├── chat_engine.py      # LangGraph conversation workflow
│   ├── database_setup.py   # PostgreSQL models (SQLAlchemy)
│   ├── vectorstore.py      # Pinecone setup
│   ├── tracing.py          # LangSmith observability
│   ├── cleanup_database.py # Utility: clear all data
│   └── check_descriptions.py # Utility: debug stored descriptions
├── uploads/                # Uploaded PDF files
├── output_images/          # Extracted images from PDFs
├── requirements.txt
├── .env                    # Environment variables
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- PostgreSQL database
- API keys for: OpenAI, Mistral AI, Pinecone, LangSmith (optional)

### Setup

```bash
# Clone and navigate to the project
cd RAG-Application

# Create virtual environment
python -m venv myvenv
myvenv\Scripts\activate  # Windows
# source myvenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create PostgreSQL database
# psql -U postgres -c "CREATE DATABASE sagemind_db;"

# Configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required
OPENAI_API_KEY=sk-...
MISTRAL_API_KEY=...
PINECONE_API_KEY=pcsk_...
DB_URI=postgresql://user:password@localhost:5432/sagemind_db

# Optional (Observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=rag-multimodal
```

## Usage

### Start the Server

```bash
cd backend
python app.py
```

Server runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info and usage examples |
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Ingest paper from arXiv URL |
| `POST` | `/upload-file` | Upload and ingest local PDF |
| `POST` | `/chat` | Ask questions about a paper |
| `POST` | `/search` | Semantic search (debug) |

### Example: Ingest a Paper

```bash
# From arXiv URL
curl -X POST "http://localhost:8000/upload?arxiv_url=https://arxiv.org/pdf/1706.03762"

# With force re-ingestion
curl -X POST "http://localhost:8000/upload?arxiv_url=https://arxiv.org/pdf/1706.03762&force_reingest=true"
```

### Example: Chat with a Paper

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main contribution of this paper?",
    "arxiv_id": "1706.03762"
  }'
```

## Database Schema

### PostgreSQL Tables

- **papers**: Paper metadata (arxiv_id, title)
- **document_fragments**: Text chunks with page numbers
- **images**: Extracted images with LLM-generated descriptions
- **tables**: Extracted tables with LLM-generated descriptions

### Pinecone Index

- **Index**: `sagemind-research-index`
- **Dimension**: 1536 (OpenAI embeddings)
- **Metric**: Cosine similarity

## Performance Optimizations

### Parallel LLM Enrichment

Images and tables are processed in parallel using `ThreadPoolExecutor`:

- **Images**: 2 concurrent workers (token-heavy due to base64)
- **Tables**: 3 concurrent workers (lighter text payloads)

### Rate Limit Handling

Built-in retry logic with exponential backoff for OpenAI rate limits:

- Parses "retry after" from error messages
- Up to 6 retries with 2s/4s/8s/16s/32s/64s delays
- Graceful fallback to placeholder descriptions if all retries fail

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Paper already exists" | Use `force_reingest=true` |
| Rate limit errors | Reduce `max_workers` in `enricher.py` or wait |
| Pinecone dimension mismatch | Run `python cleanup_database.py` |
| Missing image descriptions | Re-ingest with `enrich_content=true` |
| Slow ingestion | OCR is the bottleneck; enrichment is parallelized |

## Development

### Cleanup All Data

```bash
cd backend
python cleanup_database.py
```

### Check Stored Descriptions

```bash
python check_descriptions.py
```

## Future Enhancements

- [ ] Streaming chat responses
- [ ] PDF viewer with text highlighting (bbox support)
- [ ] Conversation memory persistence
- [ ] Multi-model support for enrichment (Gemini, Claude)
- [ ] Batch API for high-volume ingestion
- [ ] Frontend UI (React/Next.js)
- [ ] Docker containerization
- [ ] CI/CD pipeline

## License

MIT
