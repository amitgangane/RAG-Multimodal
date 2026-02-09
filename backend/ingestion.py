from ocr_processor import process_arxiv_to_markdown
from enricher import summarize_image, summarize_table, batch_summarize_images, batch_summarize_tables
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

    print(f"Extracted {len(markdown_content)} chars of markdown from {len(ocr_data.pages)} pages")

    # 6. Process and Chunk Text
    chunks = generate_rag_chunks(markdown_content)
    print(f"Generated {len(chunks)} semantic chunks")

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

    # 8. Collect all images (no LLM calls yet)
    print(f"Collecting images from {len(ocr_data.pages)} pages...")
    image_records = []  # metadata for DB records
    image_items = []    # (base64, page_num, context) for batch enrichment
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
                    print(f"   Could not extract image base64 from page {page_num}")
                    continue

                # Save image file (file I/O in main thread)
                image_path = f"output_images/{new_paper.arxiv_id}_p{page_num}_img{img_idx}.png"
                save_image_file(image_base64, image_path)

                image_records.append({
                    "id": f"{new_paper.arxiv_id}_img_{image_counter}",
                    "paper_id": new_paper.arxiv_id,
                    "page_number": page_num,
                    "file_path": image_path,
                    "image_base64_trunc": image_base64[:500],
                })
                image_items.append((image_base64, page_num, page_markdown))

    # Batch-enrich images in parallel
    if enrich_content and image_items:
        print(f"Enriching {len(image_items)} images in parallel...")
        image_descriptions = batch_summarize_images(image_items)
    else:
        image_descriptions = [f"Figure on page {rec['page_number']}" for rec in image_records]

    # Store image records in DB
    for rec, desc in zip(image_records, image_descriptions):
        img_data = ImageData(
            id=rec["id"],
            paper_id=rec["paper_id"],
            page_number=rec["page_number"],
            file_path=rec["file_path"],
            image_base64=rec["image_base64_trunc"],
            description=desc,
            bbox=None,
            created_at=datetime.now().isoformat()
        )
        session.add(img_data)

    print(f"Stored {len(image_records)} images")

    # 9. Collect all tables (no LLM calls yet)
    print(f"Collecting tables from {len(ocr_data.pages)} pages...")
    table_records = []  # metadata for DB records
    table_items = []    # (content, page_num, context) for batch enrichment
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

                table_records.append({
                    "id": f"{new_paper.arxiv_id}_table_{table_counter}",
                    "paper_id": new_paper.arxiv_id,
                    "page_number": page_num,
                    "content": table_content,
                })
                table_items.append((table_content, page_num, page_markdown))

    # Batch-enrich tables in parallel
    if enrich_content and table_items:
        print(f"Enriching {len(table_items)} tables in parallel...")
        table_descriptions = batch_summarize_tables(table_items)
    else:
        table_descriptions = [f"Table on page {rec['page_number']}" for rec in table_records]

    # Store table records in DB
    for rec, desc in zip(table_records, table_descriptions):
        table_data = TableData(
            id=rec["id"],
            paper_id=rec["paper_id"],
            page_number=rec["page_number"],
            content=rec["content"],
            description=desc,
            bbox=None,
            created_at=datetime.now().isoformat()
        )
        session.add(table_data)

    print(f"Stored {len(table_records)} tables")

    session.commit()
    print(f"\nINGESTION COMPLETE for {arxiv_id}")
    print(f"   - {len(chunks)} text chunks")
    print(f"   - {len(image_records)} images")
    print(f"   - {len(table_records)} tables")

    # Return stats for API response
    return {
        "arxiv_id": arxiv_id,
        "fragments_count": len(chunks),
        "images_count": len(image_records),
        "tables_count": len(table_records)
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

    print(f"Extracted {len(markdown_content)} chars of markdown from {len(ocr_data.pages)} pages")

    # 5. Process and Chunk Text
    chunks = generate_rag_chunks(markdown_content)
    print(f"Generated {len(chunks)} semantic chunks")

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

    # 7. Collect all images (no LLM calls yet)
    print(f"Collecting images from {len(ocr_data.pages)} pages...")
    image_records = []
    image_items = []
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

                image_records.append({
                    "id": f"{new_paper.arxiv_id}_img_{image_counter}",
                    "paper_id": new_paper.arxiv_id,
                    "page_number": page_num,
                    "file_path": image_path,
                    "image_base64_trunc": image_base64[:500],
                })
                image_items.append((image_base64, page_num, page_markdown))

    # Batch-enrich images in parallel
    if image_items:
        print(f"Enriching {len(image_items)} images in parallel...")
        image_descriptions = batch_summarize_images(image_items)
    else:
        image_descriptions = []

    # Store image records in DB
    for rec, desc in zip(image_records, image_descriptions):
        img_data = ImageData(
            id=rec["id"],
            paper_id=rec["paper_id"],
            page_number=rec["page_number"],
            file_path=rec["file_path"],
            image_base64=rec["image_base64_trunc"],
            description=desc,
            bbox=None,
            created_at=datetime.now().isoformat()
        )
        session.add(img_data)

    print(f"Stored {len(image_records)} images")

    # 8. Collect all tables (no LLM calls yet)
    print(f"Collecting tables from {len(ocr_data.pages)} pages...")
    table_records = []
    table_items = []
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

                table_records.append({
                    "id": f"{new_paper.arxiv_id}_table_{table_counter}",
                    "paper_id": new_paper.arxiv_id,
                    "page_number": page_num,
                    "content": table_content,
                })
                table_items.append((table_content, page_num, page_markdown))

    # Batch-enrich tables in parallel
    if table_items:
        print(f"Enriching {len(table_items)} tables in parallel...")
        table_descriptions = batch_summarize_tables(table_items)
    else:
        table_descriptions = []

    # Store table records in DB
    for rec, desc in zip(table_records, table_descriptions):
        table_data = TableData(
            id=rec["id"],
            paper_id=rec["paper_id"],
            page_number=rec["page_number"],
            content=rec["content"],
            description=desc,
            bbox=None,
            created_at=datetime.now().isoformat()
        )
        session.add(table_data)

    print(f"Stored {len(table_records)} tables")

    session.commit()
    print(f"\nINGESTION COMPLETE for {doc_id}")

    return {
        "arxiv_id": doc_id,
        "fragments_count": len(chunks),
        "images_count": len(image_records),
        "tables_count": len(table_records)
    }
