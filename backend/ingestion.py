from ocr_processor import process_arxiv_to_markdown
from enricher import summarize_image_with_surroundings
from chunker import generate_rag_chunks
from database_setup import Paper, DocumentFragment, ImageData, TableData, init_postgres
from vectorstore import init_pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from sqlalchemy.orm import sessionmaker

import os
from datetime import datetime
from pathlib import Path
import base64

def save_image_file(image_base64, image_path):
    """Save base64 image to disk"""
    Path(image_path).parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image_base64, str) and image_base64.startswith("data:"):
        header, encoded = image_base64.split(",", 1)
        image_data = base64.b64decode(encoded)
    else:
        image_data = base64.b64decode(image_base64)
    with open(image_path, "wb") as f:
        f.write(image_data)

def extract_tables_from_markdown(markdown_content):
    """Extract markdown tables and their positions"""
    tables = []
    lines = markdown_content.split('\n')
    for i, line in enumerate(lines):
        if '|' in line and '-' in lines[i+1] if i+1 < len(lines) else False:
            # Simple table detection
            table_lines = [line]
            j = i + 1
            while j < len(lines) and '|' in lines[j]:
                table_lines.append(lines[j])
                j += 1
            tables.append({
                "markdown": "\n".join(table_lines),
                "page": 1  # Approximate
            })
    return tables

def ingest_paper_to_production(arxiv_url):
    # 1. Initialization
    engine = init_postgres()
    Session = sessionmaker(bind=engine)
    session = Session()
    index = init_pinecone()
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    # 2. Mistral OCR Step (The "Temporary" Data)
    ocr_data = process_arxiv_to_markdown(arxiv_url)
    
    # 3. Create Paper Record
    new_paper = Paper(arxiv_id=ocr_data.id, title="Sample RL Paper")
    session.add(new_paper)
    session.flush()
    
    # 4. Process and Chunk Text
    chunks = generate_rag_chunks(ocr_data.markdown)
    
    for chunk in chunks:
        # Create Fragment for PostgreSQL
        fragment = DocumentFragment(
            paper_id=new_paper.arxiv_id,
            content=chunk.page_content,
            page_number=chunk.metadata.get("page", 1),
            bbox=chunk.metadata.get("bbox")  # Coordinates for PDF text highlighting
        )
        session.add(fragment)
        session.flush()  # Get the fragment ID

        # Generate Embedding and Upsert to Pinecone
        vector = embeddings_model.embed_query(chunk.page_content)
        index.upsert(vectors=[(
            fragment.id, 
            vector, 
            {"arxiv_id": new_paper.arxiv_id, "text": chunk.page_content[:100], "page": chunk.metadata.get("page", 1)}
        )])

    # 5. Extract and Store Images with Descriptions
    print(f"   Processing images from {len(ocr_data.pages)} pages...")
    image_counter = 0
    for page_num, page in enumerate(ocr_data.pages, 1):
        if hasattr(page, 'images') and page.images:
            for img_idx, image in enumerate(page.images):
                image_counter += 1
                # Save image file
                image_path = f"output_images/{new_paper.arxiv_id}_p{page_num}_img{img_idx}.png"
                save_image_file(image.image_base64, image_path)
                
                # Store image metadata (skip LLM description for now - too slow)
                description = f"Figure on page {page_num}"
                
                img_data = ImageData(
                    id=f"{new_paper.arxiv_id}_img_{image_counter}",
                    paper_id=new_paper.arxiv_id,
                    page_number=page_num,
                    file_path=image_path,
                    image_base64=image.image_base64[:500],  # Store truncated base64
                    description=description,
                    bbox=None,  # Mistral OCR doesn't provide bbox for images
                    created_at=datetime.now().isoformat()
                )
                session.add(img_data)
    
    print(f"   ✅ Stored {image_counter} images")

    # 6. Extract and Store Tables with Descriptions
    print(f"   Processing tables from {len(ocr_data.pages)} pages...")
    table_counter = 0
    for page_num, page in enumerate(ocr_data.pages, 1):
        if hasattr(page, 'tables') and page.tables:
            for tbl_idx, table in enumerate(page.tables):
                table_counter += 1
                # Store table metadata (skip LLM description for now - too slow)
                description = f"Table on page {page_num}"
                
                table_data = TableData(
                    id=f"{new_paper.arxiv_id}_table_{table_counter}",
                    paper_id=new_paper.arxiv_id,
                    page_number=page_num,
                    content=table.content,
                    description=description,
                    bbox=None,  # Mistral OCR doesn't provide bbox for tables
                    created_at=datetime.now().isoformat()
                )
                session.add(table_data)
    
    print(f"   ✅ Stored {table_counter} tables")

    session.commit()
    print(f"Ingestion complete: Text chunks, images, and tables stored for {new_paper.arxiv_id}")