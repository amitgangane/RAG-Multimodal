import os
from sqlalchemy import create_engine, Column, String, Integer, JSON, Text
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv
import uuid
load_dotenv()

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg_pool import ConnectionPool
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# Database URI - from environment variable or use hardcoded fallback
DB_URI = os.getenv('DB_URI')

# Existing SQLAlchemy Setup for Papers & Fragments
Base = declarative_base()

class Paper(Base):
    __tablename__ = 'papers'
    arxiv_id = Column(String, primary_key=True)
    title = Column(Text)

class DocumentFragment(Base):
    __tablename__ = 'document_fragments'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    paper_id = Column(String)
    content = Column(Text)
    bbox = Column(JSON)  # For PDF text highlighting
    page_number = Column(Integer)

class ImageData(Base):
    __tablename__ = 'images'
    id = Column(String, primary_key=True)
    paper_id = Column(String)  # ForeignKey to Paper.arxiv_id
    page_number = Column(Integer)  # Which page the image appears on
    file_path = Column(String)  # Local storage path or S3 URL
    image_base64 = Column(Text)  # Base64 encoded image data
    description = Column(Text)  # LLM-generated description (from enricher)
    bbox = Column(JSON)  # Image bounding box on page for highlighting
    created_at = Column(String)  # Timestamp

class TableData(Base):
    __tablename__ = 'tables'
    id = Column(String, primary_key=True)
    paper_id = Column(String)  # ForeignKey to Paper.arxiv_id
    page_number = Column(Integer)  # Which page the table appears on
    content = Column(Text)  # Markdown formatted table
    description = Column(Text)  # LLM-generated table explanation
    bbox = Column(JSON)  # Table bounding box on page for highlighting
    created_at = Column(String)  # Timestamp

# NEW: LangGraph Postgres Checkpointer Setup (Optional)
def get_graph_checkpointer():
    """Initialize LangGraph PostgreSQL checkpointer for conversation state persistence"""
    if not LANGGRAPH_AVAILABLE:
        raise ImportError("LangGraph PostgreSQL support not installed. Install with: pip install langgraph")
    
    # autocommit=True is required for the initial .setup()
    connection_kwargs = {"autocommit": True}
    pool = ConnectionPool(conninfo=DB_URI, max_size=10, kwargs=connection_kwargs)
    checkpointer = PostgresSaver(pool)
    # Run once to create LangGraph's internal tables
    checkpointer.setup()
    return checkpointer


def init_postgres():
    """Initialize and return the PostgreSQL engine."""
    engine = create_engine(DB_URI)
    Base.metadata.create_all(engine)
    return engine