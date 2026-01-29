from ocr_processor import process_arxiv_to_markdown
from enricher import summarize_image, summarize_table
from chunker import generate_rag_chunks
from database_setup import Paper, DocumentFragment, ImageData, TableData, init_postgres
from vectorstore import init_pinecone
from langchain_openai import OpenAIEmbeddings
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
        if '|' in line and ('-' in lines[i+1] if i+1 < len(lines) else False):
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

def ingest_paper_to_production(arxiv_url, force_reingest=False, enrich_content=True):
    # 1. Initialization
    engine = init_postgres()
    Session = sessionmaker(bind=engine)
    session = Session()
    index = init_pinecone()
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Extract arxiv_id from URL
    # URL format: https://arxiv.org/pdf/1706.03762.pdf or https://arxiv.org/pdf/1706.03762
    arxiv_id = arxiv_url.split("/")[-1].replace(".pdf", "")

    # 3. Check if paper already exists
    existing_paper = session.query(Paper).filter_by(arxiv_id=arxiv_id).first()
    if existing_paper:
        if not force_reingest:
            print(f"Paper {arxiv_id} already exists. Skipping ingestion.")
            # Count existing data
            fragments_count = session.query(DocumentFragment).filter_by(paper_id=arxiv_id).count()
            images_count = session.query(ImageData).filter_by(paper_id=arxiv_id).count()
            tables_count = session.query(TableData).filter_by(paper_id=arxiv_id).count()
            return {
                "arxiv_id": arxiv_id,
                "fragments_count": fragments_count,
                "images_count": images_count,
                "tables_count": tables_count,
                "message": "Paper already exists"
            }
        else:
            # Delete existing data for re-ingestion
            print(f"Re-ingesting paper {arxiv_id}. Deleting existing data...")
            session.query(DocumentFragment).filter_by(paper_id=arxiv_id).delete()
            session.query(ImageData).filter_by(paper_id=arxiv_id).delete()
            session.query(TableData).filter_by(paper_id=arxiv_id).delete()
            session.query(Paper).filter_by(arxiv_id=arxiv_id).delete()
            session.commit()

    # 4. Mistral OCR Step (The "Temporary" Data)
    ocr_data = process_arxiv_to_markdown(arxiv_url)

    # 5. Create Paper Record
    new_paper = Paper(arxiv_id=arxiv_id, title="Research Paper")
    session.add(new_paper)
    session.flush()
    
    # 5. Extract markdown content from OCR response pages
    # Mistral OCR returns pages with markdown attribute
    markdown_content = ""
    if hasattr(ocr_data, 'pages') and ocr_data.pages:
        for page in ocr_data.pages:
            if hasattr(page, 'markdown') and page.markdown:
                markdown_content += page.markdown + "\n"
    
    if not markdown_content:
        raise ValueError("No markdown content extracted from OCR")
    
    print(f"✅ Extracted {len(markdown_content)} chars of markdown from {len(ocr_data.pages)} pages")
    
    # 6. Process and Chunk Text
    chunks = generate_rag_chunks(markdown_content)
    print(f"✅ Generated {len(chunks)} semantic chunks")
    
    # 7. Store chunks in PostgreSQL and Pinecone
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

    # 8. Extract and Store Images with LLM Descriptions
    print(f"Processing images from {len(ocr_data.pages)} pages...")
    image_counter = 0
    for page_num, page in enumerate(ocr_data.pages, 1):
        # Get page markdown for context
        page_markdown = page.markdown if hasattr(page, 'markdown') else ""

        if hasattr(page, 'images') and page.images:
            for img_idx, image in enumerate(page.images):
                image_counter += 1
                # Get base64 image - could be in different attribute names
                image_base64 = None
                if hasattr(image, 'image_base64'):
                    image_base64 = image.image_base64
                elif hasattr(image, 'base64'):
                    image_base64 = image.base64
                elif hasattr(image, 'data'):
                    image_base64 = image.data

                if not image_base64:
                    print(f"   ⚠️  Could not extract image base64 from page {page_num}")
                    continue

                # Save image file
                image_path = f"output_images/{new_paper.arxiv_id}_p{page_num}_img{img_idx}.png"
                save_image_file(image_base64, image_path)

                # Generate LLM description for image (if enrichment enabled)
                if enrich_content:
                    print(f"   🖼️  Generating description for image {image_counter} (page {page_num})...")
                    description = summarize_image(image_base64, page_num, page_markdown)
                else:
                    description = f"Figure on page {page_num}"

                img_data = ImageData(
                    id=f"{new_paper.arxiv_id}_img_{image_counter}",
                    paper_id=new_paper.arxiv_id,
                    page_number=page_num,
                    file_path=image_path,
                    image_base64=image_base64[:500],  # Store truncated base64
                    description=description,
                    bbox=None,
                    created_at=datetime.now().isoformat()
                )
                session.add(img_data)

    print(f"✅ Stored {image_counter} images")

    # 9. Extract and Store Tables with LLM Descriptions
    print(f"Processing tables from {len(ocr_data.pages)} pages...")
    table_counter = 0
    for page_num, page in enumerate(ocr_data.pages, 1):
        # Get page markdown for context
        page_markdown = page.markdown if hasattr(page, 'markdown') else ""

        if hasattr(page, 'tables') and page.tables:
            for tbl_idx, table in enumerate(page.tables):
                table_counter += 1
                # Get table content - could be in different attribute names
                table_content = None
                if hasattr(table, 'content'):
                    table_content = table.content
                elif hasattr(table, 'markdown'):
                    table_content = table.markdown
                elif hasattr(table, 'data'):
                    table_content = table.data
                else:
                    table_content = str(table)

                # Generate LLM description for table (if enrichment enabled)
                if enrich_content:
                    print(f"   📊 Generating description for table {table_counter} (page {page_num})...")
                    description = summarize_table(table_content, page_num, page_markdown)
                else:
                    description = f"Table on page {page_num}"

                table_data = TableData(
                    id=f"{new_paper.arxiv_id}_table_{table_counter}",
                    paper_id=new_paper.arxiv_id,
                    page_number=page_num,
                    content=table_content,
                    description=description,
                    bbox=None,
                    created_at=datetime.now().isoformat()
                )
                session.add(table_data)

    print(f"✅ Stored {table_counter} tables")

    session.commit()
    print(f"\n✅ INGESTION COMPLETE for {arxiv_id}")
    print(f"   - {len(chunks)} text chunks")
    print(f"   - {image_counter} images")
    print(f"   - {table_counter} tables")
    
    # Return stats for API response
    return {
        "arxiv_id": arxiv_id,
        "fragments_count": len(chunks),
        "images_count": image_counter,
        "tables_count": table_counter
    }


def ingest_pdf_file(file_path: str, doc_id: str, force_reingest=False):
    """
    Ingest a local PDF file into the RAG system.

    Args:
        file_path: Path to the local PDF file
        doc_id: Unique document identifier
        force_reingest: If True, delete existing data and re-ingest

    Returns:
        Stats dictionary with ingestion results
    """
    from ocr_processor import process_pdf_file

    # 1. Initialization
    engine = init_postgres()
    Session = sessionmaker(bind=engine)
    session = Session()
    index = init_pinecone()
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Check if document already exists
    existing_paper = session.query(Paper).filter_by(arxiv_id=doc_id).first()
    if existing_paper:
        if not force_reingest:
            print(f"Document {doc_id} already exists. Skipping ingestion.")
            fragments_count = session.query(DocumentFragment).filter_by(paper_id=doc_id).count()
            images_count = session.query(ImageData).filter_by(paper_id=doc_id).count()
            tables_count = session.query(TableData).filter_by(paper_id=doc_id).count()
            return {
                "arxiv_id": doc_id,
                "fragments_count": fragments_count,
                "images_count": images_count,
                "tables_count": tables_count,
                "message": "Document already exists"
            }
        else:
            print(f"Re-ingesting document {doc_id}. Deleting existing data...")
            session.query(DocumentFragment).filter_by(paper_id=doc_id).delete()
            session.query(ImageData).filter_by(paper_id=doc_id).delete()
            session.query(TableData).filter_by(paper_id=doc_id).delete()
            session.query(Paper).filter_by(arxiv_id=doc_id).delete()
            session.commit()

    # 3. Process PDF with Mistral OCR
    ocr_data = process_pdf_file(file_path)

    # 4. Create Paper Record
    new_paper = Paper(arxiv_id=doc_id, title=os.path.basename(file_path))
    session.add(new_paper)
    session.flush()

    # 4. Extract markdown content from OCR response pages
    markdown_content = ""
    if hasattr(ocr_data, 'pages') and ocr_data.pages:
        for page in ocr_data.pages:
            if hasattr(page, 'markdown') and page.markdown:
                markdown_content += page.markdown + "\n"

    if not markdown_content:
        raise ValueError("No markdown content extracted from PDF")

    print(f"✅ Extracted {len(markdown_content)} chars of markdown from {len(ocr_data.pages)} pages")

    # 5. Process and Chunk Text
    chunks = generate_rag_chunks(markdown_content)
    print(f"✅ Generated {len(chunks)} semantic chunks")

    # 6. Store chunks in PostgreSQL and Pinecone
    for chunk in chunks:
        fragment = DocumentFragment(
            paper_id=new_paper.arxiv_id,
            content=chunk.page_content,
            page_number=chunk.metadata.get("page", 1),
            bbox=chunk.metadata.get("bbox")
        )
        session.add(fragment)
        session.flush()

        vector = embeddings_model.embed_query(chunk.page_content)
        index.upsert(vectors=[(
            fragment.id,
            vector,
            {"arxiv_id": new_paper.arxiv_id, "text": chunk.page_content[:100], "page": chunk.metadata.get("page", 1)}
        )])

    # 7. Extract and Store Images with LLM Descriptions
    print(f"Processing images from {len(ocr_data.pages)} pages...")
    image_counter = 0
    for page_num, page in enumerate(ocr_data.pages, 1):
        page_markdown = page.markdown if hasattr(page, 'markdown') else ""

        if hasattr(page, 'images') and page.images:
            for img_idx, image in enumerate(page.images):
                image_counter += 1
                image_base64 = None
                if hasattr(image, 'image_base64'):
                    image_base64 = image.image_base64
                elif hasattr(image, 'base64'):
                    image_base64 = image.base64
                elif hasattr(image, 'data'):
                    image_base64 = image.data

                if not image_base64:
                    continue

                image_path = f"output_images/{new_paper.arxiv_id}_p{page_num}_img{img_idx}.png"
                save_image_file(image_base64, image_path)

                # Generate LLM description for image
                print(f"   🖼️  Generating description for image {image_counter} (page {page_num})...")
                description = summarize_image(image_base64, page_num, page_markdown)

                img_data = ImageData(
                    id=f"{new_paper.arxiv_id}_img_{image_counter}",
                    paper_id=new_paper.arxiv_id,
                    page_number=page_num,
                    file_path=image_path,
                    image_base64=image_base64[:500],
                    description=description,
                    bbox=None,
                    created_at=datetime.now().isoformat()
                )
                session.add(img_data)

    print(f"✅ Stored {image_counter} images")

    # 8. Extract and Store Tables with LLM Descriptions
    print(f"Processing tables from {len(ocr_data.pages)} pages...")
    table_counter = 0
    for page_num, page in enumerate(ocr_data.pages, 1):
        page_markdown = page.markdown if hasattr(page, 'markdown') else ""

        if hasattr(page, 'tables') and page.tables:
            for tbl_idx, table in enumerate(page.tables):
                table_counter += 1
                table_content = None
                if hasattr(table, 'content'):
                    table_content = table.content
                elif hasattr(table, 'markdown'):
                    table_content = table.markdown
                elif hasattr(table, 'data'):
                    table_content = table.data
                else:
                    table_content = str(table)

                # Generate LLM description for table
                print(f"   📊 Generating description for table {table_counter} (page {page_num})...")
                description = summarize_table(table_content, page_num, page_markdown)

                table_data = TableData(
                    id=f"{new_paper.arxiv_id}_table_{table_counter}",
                    paper_id=new_paper.arxiv_id,
                    page_number=page_num,
                    content=table_content,
                    description=description,
                    bbox=None,
                    created_at=datetime.now().isoformat()
                )
                session.add(table_data)

    print(f"✅ Stored {table_counter} tables")

    session.commit()
    print(f"\n✅ INGESTION COMPLETE for {doc_id}")

    return {
        "arxiv_id": doc_id,
        "fragments_count": len(chunks),
        "images_count": image_counter,
        "tables_count": table_counter
    }