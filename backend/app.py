import uuid
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from langchain_core.messages import HumanMessage
from chat_engine import research_agent
from ingestion import ingest_paper_to_production, ingest_pdf_file
from retriever import retrieve_hybrid_context
from tracing import setup_langsmith

# Initialize LangSmith tracing (if configured)
setup_langsmith()

# Directory for uploaded files
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ==================== FastAPI Setup ====================
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation API for research papers",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Request/Response Models ====================
class ChatRequest(BaseModel):
    query: str
    arxiv_id: str
    thread_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    status: str = "success"

class SearchRequest(BaseModel):
    query: str
    arxiv_id: str
    top_k: int = 5

class SearchResult(BaseModel):
    text_chunks: list
    images: list
    tables: list
    pages_referenced: list

class HealthResponse(BaseModel):
    status: str
    version: str
    message: str

class UploadResponse(BaseModel):
    arxiv_id: str
    status: str
    message: str
    fragments_count: int
    images_count: int
    tables_count: int

# ==================== Helper Functions ====================
def chat_with_agent(user_query, arxiv_id, thread_id=None):
    """Execute chat query through LangGraph workflow"""
    if not thread_id:
        thread_id = str(uuid.uuid4())
        
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {
        "messages": [HumanMessage(content=user_query)],
        "arxiv_id": arxiv_id
    }
    
    # Stream the graph execution
    final_state = research_agent.invoke(inputs, config=config)
    
    return {
        "answer": final_state["messages"][-1].content,
        "thread_id": thread_id
    }

# ==================== Routes ====================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and availability"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "message": "RAG Chatbot API is running"
    }


@app.get("/tracing-status")
async def tracing_status():
    """Check LangSmith tracing status"""
    from tracing import get_tracing_status
    enabled = get_tracing_status()
    project = os.getenv("LANGCHAIN_PROJECT", "rag-multimodal")
    return {
        "tracing_enabled": enabled,
        "project": project if enabled else None,
        "dashboard_url": f"https://smith.langchain.com/o/default/projects/p/{project}" if enabled else None
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for question-answering on research papers
    
    Args:
        query: User question
        arxiv_id: Paper identifier (e.g., "1706.03762")
        thread_id: Optional conversation thread ID
    
    Returns:
        answer: LLM response with context
        thread_id: Conversation thread ID for multi-turn chat
    """
    try:
        if not request.query or not request.arxiv_id:
            raise HTTPException(status_code=400, detail="query and arxiv_id are required")
        
        result = chat_with_agent(request.query, request.arxiv_id, request.thread_id)
        
        return ChatResponse(
            answer=result["answer"],
            thread_id=result["thread_id"],
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/search", response_model=SearchResult)
async def search(request: SearchRequest):
    """
    Semantic search endpoint for retrieving context
    
    Args:
        query: Search query
        arxiv_id: Paper identifier
        top_k: Number of top results (default 5)
    
    Returns:
        text_chunks: Relevant text chunks with similarity scores
        images: Related images with descriptions
        tables: Related tables with content
        pages_referenced: Pages containing results
    """
    try:
        if not request.query or not request.arxiv_id:
            raise HTTPException(status_code=400, detail="query and arxiv_id are required")
        
        results = retrieve_hybrid_context(request.query, request.arxiv_id)
        
        return SearchResult(
            text_chunks=results.get("text_chunks", []),
            images=results.get("images", []),
            tables=results.get("tables", []),
            pages_referenced=results.get("pages_referenced", [])
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/upload", response_model=UploadResponse)
async def upload_paper(arxiv_url: str, force_reingest: bool = False):
    """
    Upload and ingest a research paper from arXiv

    Args:
        arxiv_url: Full arXiv URL (e.g., "https://arxiv.org/pdf/1706.03762.pdf")
        force_reingest: If True, delete existing data and re-ingest (default: False)

    Returns:
        arxiv_id: Paper identifier extracted from URL
        status: Ingestion status
        message: Ingestion details
        fragments_count: Number of text chunks created
        images_count: Number of images extracted
        tables_count: Number of tables extracted
    """
    try:
        if not arxiv_url:
            raise HTTPException(status_code=400, detail="arxiv_url is required")

        # Ingest the paper
        stats = ingest_paper_to_production(arxiv_url, force_reingest=force_reingest)
        
        # Check if paper already existed
        if stats.get("message") == "Paper already exists":
            return UploadResponse(
                arxiv_id=stats.get("arxiv_id", "unknown"),
                status="exists",
                message="Paper already exists. Use force_reingest=true to re-ingest.",
                fragments_count=stats.get("fragments_count", 0),
                images_count=stats.get("images_count", 0),
                tables_count=stats.get("tables_count", 0)
            )

        return UploadResponse(
            arxiv_id=stats.get("arxiv_id", "unknown"),
            status="success",
            message="Paper successfully ingested",
            fragments_count=stats.get("fragments_count", 0),
            images_count=stats.get("images_count", 0),
            tables_count=stats.get("tables_count", 0)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


@app.post("/upload-file", response_model=UploadResponse)
async def upload_pdf_file(file: UploadFile = File(...)):
    """
    Upload a PDF file directly for ingestion

    Args:
        file: PDF file to upload

    Returns:
        arxiv_id: Generated document identifier
        status: Ingestion status
        message: Ingestion details
        fragments_count: Number of text chunks created
        images_count: Number of images extracted
        tables_count: Number of tables extracted
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        # Generate unique document ID from filename
        doc_id = file.filename.replace('.pdf', '').replace(' ', '_')
        doc_id = f"{doc_id}_{uuid.uuid4().hex[:8]}"

        # Save uploaded file
        file_path = UPLOAD_DIR / f"{doc_id}.pdf"
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        # Ingest the PDF file
        stats = ingest_pdf_file(str(file_path), doc_id)

        # Check if document already existed
        if stats.get("message") == "Document already exists":
            return UploadResponse(
                arxiv_id=stats.get("arxiv_id", doc_id),
                status="exists",
                message="Document already exists",
                fragments_count=stats.get("fragments_count", 0),
                images_count=stats.get("images_count", 0),
                tables_count=stats.get("tables_count", 0)
            )

        return UploadResponse(
            arxiv_id=stats.get("arxiv_id", doc_id),
            status="success",
            message=f"PDF '{file.filename}' successfully ingested",
            fragments_count=stats.get("fragments_count", 0),
            images_count=stats.get("images_count", 0),
            tables_count=stats.get("tables_count", 0)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")


# ==================== Root Endpoint ====================
@app.get("/")
async def root():
    """API documentation and available endpoints"""
    return {
        "name": "RAG Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Check API health",
            "POST /chat": "Ask questions about papers",
            "POST /search": "Semantic search for context",
            "POST /upload": "Ingest new papers from arXiv URL",
            "POST /upload-file": "Upload and ingest a PDF file directly",
            "GET /docs": "Interactive API documentation (Swagger UI)",
            "GET /redoc": "ReDoc documentation"
        },
        "example_usage": {
            "chat": {
                "url": "POST /chat",
                "body": {
                    "query": "What is the Transformer architecture?",
                    "arxiv_id": "1706.03762",
                    "thread_id": "optional-conversation-id"
                }
            },
            "search": {
                "url": "POST /search",
                "body": {
                    "query": "attention mechanism",
                    "arxiv_id": "1706.03762",
                    "top_k": 5
                }
            },
            "upload": {
                "url": "POST /upload?arxiv_url=https://arxiv.org/pdf/1706.03762.pdf"
            }
        }
    }

# ==================== Server Setup ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )