import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from database_setup import init_postgres, DocumentFragment, ImageData, TableData
from vectorstore import init_pinecone
from sqlalchemy.orm import sessionmaker

def retrieve_hybrid_context(query, arxiv_id):
    # 1. Initialize
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    index = init_pinecone()
    engine = init_postgres()
    Session = sessionmaker(bind=engine)
    session = Session()

    # 2. Vector Search (Pinecone)
    query_vector = embeddings_model.embed_query(query)
    results = index.query(
        vector=query_vector,
        top_k=5,
        filter={"arxiv_id": {"$eq": arxiv_id}},
        include_metadata=True
    )

    # 3. Join with PostgreSQL for Visual Grounding (Bboxes for PDF highlighting)
    enriched_results = {
        "text_chunks": [],
        "images": [],
        "tables": [],
        "pages_referenced": set()
    }
    
    # Process text chunks with bounding boxes
    for match in results['matches']:
        fragment = session.query(DocumentFragment).filter_by(id=match.id).first()
        if fragment:
            enriched_results["text_chunks"].append({
                "id": fragment.id,
                "content": fragment.content,
                "bbox": fragment.bbox,  # For PDF text highlighting
                "page": fragment.page_number,
                "score": match.score
            })
            enriched_results["pages_referenced"].add(fragment.page_number)
    
    # Fetch related images from same pages AND nearby pages (±3 pages for broader context)
    if enriched_results["pages_referenced"]:
        pages_to_search = set()
        for page in enriched_results["pages_referenced"]:
            pages_to_search.update([page - 3, page - 2, page - 1, page, page + 1, page + 2, page + 3])
        pages_to_search = [p for p in pages_to_search if p > 0]  # Remove invalid pages
        
        images = session.query(ImageData).filter(
            ImageData.paper_id == arxiv_id,
            ImageData.page_number.in_(pages_to_search)
        ).all()
        
        enriched_results["images"] = [{
            "id": img.id,
            "page": img.page_number,
            "file_path": img.file_path,
            "description": img.description,  # LLM-generated description
            "bbox": img.bbox  # For PDF image highlighting
        } for img in images]
    
    # Fetch related tables - search entire paper for tables (they're important for understanding results)
    if enriched_results["pages_referenced"]:
        # Fetch ALL tables for this paper to provide comprehensive context
        tables = session.query(TableData).filter(
            TableData.paper_id == arxiv_id
        ).all()
        
        enriched_results["tables"] = [{
            "id": tbl.id,
            "page": tbl.page_number,
            "content": tbl.content,  # Markdown table
            "description": tbl.description,  # LLM-generated explanation
            "bbox": tbl.bbox  # For PDF table highlighting
        } for tbl in tables]
    
    # Convert set to list for JSON serialization
    enriched_results["pages_referenced"] = list(enriched_results["pages_referenced"])
    
    session.close()
    return enriched_results