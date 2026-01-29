# RAG Multimodal Application - Project Overview

## Project Structure

```
RAG-Application/
├── .env                    # Environment variables (API keys)
├── requirements.txt        # Python dependencies
├── PROJECT_OVERVIEW.md     # This file
│
├── backend/
│   ├── app.py              # FastAPI entry point (START HERE)
│   ├── tracing.py          # LangSmith observability setup
│   │
│   ├── # ===== INGESTION PIPELINE =====
│   ├── ocr_processor.py    # Step 1: PDF → Markdown (Mistral OCR)
│   ├── chunker.py          # Step 2: Markdown → Semantic chunks
│   ├── enricher.py         # Step 3: Generate image/table descriptions (OpenAI)
│   ├── ingestion.py        # Orchestrates full ingestion pipeline
│   │
│   ├── # ===== STORAGE =====
│   ├── database_setup.py   # PostgreSQL models & connections
│   ├── vectorstore.py      # Pinecone vector database setup
│   │
│   ├── # ===== RETRIEVAL & CHAT =====
│   ├── retriever.py        # Hybrid search (Pinecone + PostgreSQL)
│   ├── chat_engine.py      # LangGraph conversation workflow
│   │
│   ├── # ===== UTILITIES =====
│   ├── cleanup_database.py # Delete all data (for testing)
│   └── check_descriptions.py # Debug: view stored descriptions
│
└── uploads/                # Uploaded PDF files
```

---

## Application Flow

### 1. Starting the Server

```
app.py (Entry Point)
    │
    ├── setup_langsmith()           # Initialize tracing (optional)
    ├── Import chat_engine          # Loads LangGraph workflow
    │       └── research_agent      # Compiled conversation graph
    │
    └── FastAPI app starts on port 8000
```

### 2. Paper Ingestion Flow (`POST /upload`)

```
User uploads arXiv URL
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  ingestion.py: ingest_paper_to_production()                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. ocr_processor.py: process_arxiv_to_markdown()              │
│     └── Mistral OCR API → Returns pages with markdown + images │
│                                                                 │
│  2. chunker.py: generate_rag_chunks()                          │
│     └── Split markdown by headers → Sub-chunk for vectors      │
│                                                                 │
│  3. Store chunks in PostgreSQL (DocumentFragment table)        │
│                                                                 │
│  4. Generate embeddings → Store in Pinecone                    │
│     └── OpenAI text-embedding-3-small (1536 dimensions)        │
│                                                                 │
│  5. enricher.py: summarize_image() [if enrich_content=True]    │
│     └── GPT-4o-mini vision → Generate image descriptions       │
│                                                                 │
│  6. enricher.py: summarize_table() [if enrich_content=True]    │
│     └── GPT-4o-mini → Generate table descriptions              │
│                                                                 │
│  7. Store images/tables in PostgreSQL (ImageData, TableData)   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Chat Flow (`POST /chat`)

```
User sends query + arxiv_id
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  chat_engine.py: research_agent (LangGraph Workflow)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  START                                                          │
│    │                                                            │
│    ▼                                                            │
│  retrieve_node()                                                │
│    └── retriever.py: retrieve_hybrid_context()                 │
│        ├── Embed query → Search Pinecone (vector similarity)   │
│        ├── Fetch full content from PostgreSQL (DocumentFragment)│
│        ├── Fetch related images (ImageData)                    │
│        └── Fetch related tables (TableData)                    │
│    │                                                            │
│    ▼                                                            │
│  generate_node()                                                │
│    └── Format context (text + images + tables)                 │
│    └── GPT-4o-mini generates answer with citations             │
│    │                                                            │
│    ▼                                                            │
│  END → Return answer + thread_id                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Database Schema

### PostgreSQL Tables

```
┌─────────────────────┐     ┌─────────────────────────┐
│       papers        │     │   document_fragments    │
├─────────────────────┤     ├─────────────────────────┤
│ arxiv_id (PK)       │◄────│ paper_id (FK)           │
│ title               │     │ id (PK)                 │
└─────────────────────┘     │ content                 │
        │                   │ page_number             │
        │                   │ bbox (JSON)             │
        │                   └─────────────────────────┘
        │
        │                   ┌─────────────────────────┐
        │                   │        images           │
        │                   ├─────────────────────────┤
        └──────────────────►│ paper_id (FK)           │
                            │ id (PK)                 │
                            │ page_number             │
                            │ file_path               │
                            │ description (LLM)       │
                            │ bbox (JSON)             │
                            └─────────────────────────┘

                            ┌─────────────────────────┐
                            │        tables           │
                            ├─────────────────────────┤
                            │ paper_id (FK)           │
                            │ id (PK)                 │
                            │ page_number             │
                            │ content (markdown)      │
                            │ description (LLM)       │
                            │ bbox (JSON)             │
                            └─────────────────────────┘
```

### Pinecone Index

```
Index: sagemind-research-index
Dimension: 1536 (OpenAI embeddings)
Metric: cosine

Vectors:
  - id: fragment.id (matches PostgreSQL)
  - values: [1536 floats]
  - metadata: {arxiv_id, text (preview), page}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and usage examples |
| GET | `/health` | Health check |
| GET | `/tracing-status` | LangSmith tracing status |
| POST | `/upload?arxiv_url=...` | Ingest paper from arXiv URL |
| POST | `/upload-file` | Upload and ingest PDF file |
| POST | `/chat` | Ask questions about a paper |
| POST | `/search` | Semantic search (debug) |

---

## Environment Variables (.env)

```bash
# Required
OPENAI_API_KEY=sk-...           # For chat, embeddings, image analysis
MISTRAL_API_KEY=...             # For OCR processing
PINECONE_API_KEY=pcsk_...       # Vector database
DB_URI=postgresql://user:pass@localhost:5432/sagemind_db

# Optional (Observability)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...   # LangSmith API key
LANGCHAIN_PROJECT=rag-multimodal
```

---

## Key Functions Reference

### ocr_processor.py
- `process_arxiv_to_markdown(url)` - OCR from URL
- `process_pdf_file(path)` - OCR from local file

### chunker.py
- `generate_rag_chunks(markdown)` - Split into semantic chunks

### enricher.py
- `summarize_image(base64, page, context)` - Generate image description
- `summarize_table(content, page, context)` - Generate table description

### ingestion.py
- `ingest_paper_to_production(url, force_reingest, enrich_content)`
- `ingest_pdf_file(path, doc_id, force_reingest)`

### retriever.py
- `retrieve_hybrid_context(query, arxiv_id)` - Returns text + images + tables

### chat_engine.py
- `research_agent` - Compiled LangGraph workflow
- `retrieve_node(state)` - Fetch context
- `generate_node(state)` - Generate answer

---

## Next Steps / TODOs

### Immediate
- [ ] Fix LangSmith API key (403 error)
- [ ] Test with more papers

### Enhancements
- [ ] Add streaming responses for chat
- [ ] Add PDF viewer with highlighting (bbox support)
- [ ] Implement conversation memory persistence
- [ ] Add user authentication

### Performance
- [ ] Parallel image processing during ingestion
- [ ] Caching for repeated queries
- [ ] Batch embedding generation

### Frontend
- [ ] Build React/Next.js UI
- [ ] PDF viewer component with highlights
- [ ] Chat interface with history
- [ ] Paper upload drag-and-drop

### Deployment
- [ ] Dockerize the application
- [ ] Set up CI/CD pipeline
- [ ] Deploy to cloud (AWS/GCP/Azure)

---

## Running the Application

```bash
# 1. Activate virtual environment
cd C:\Users\amitg\Desktop\RAG-Application
myvenv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start PostgreSQL (if not running)
# Ensure your database exists: sagemind_db

# 4. Run the server
cd backend
python app.py

# Server starts at http://localhost:8000
# API docs at http://localhost:8000/docs
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Duplicate key error | Paper already exists. Use `force_reingest=true` |
| Pinecone dimension mismatch | Run `cleanup_database.py` to delete old index |
| LangSmith 403 error | Check API key at smith.langchain.com/settings |
| Image descriptions are placeholders | Re-ingest with `enrich_content=true` |
| Slow ingestion | Use `enrich_content=false` for faster processing |
