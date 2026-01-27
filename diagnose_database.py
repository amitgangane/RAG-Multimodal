"""
Diagnostic: Check what's actually stored in the database for the ingested paper
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

def diagnose_database():
    try:
        print("=" * 80)
        print("DATABASE DIAGNOSTIC - CHECKING STORED CONTENT")
        print("=" * 80)
        
        from database_setup import init_postgres, Paper, DocumentFragment, ImageData, TableData
        from sqlalchemy.orm import sessionmaker
        
        engine = init_postgres()
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Check papers
        print("\n1. PAPERS IN DATABASE:")
        papers = session.query(Paper).all()
        print(f"   Total: {len(papers)}")
        for paper in papers:
            print(f"   - {paper.arxiv_id}: {paper.title}")
        
        if not papers:
            print("   ⚠️  No papers found!")
            session.close()
            return
        
        # Check fragments
        print("\n2. DOCUMENT FRAGMENTS:")
        paper_arxiv_id = papers[0].arxiv_id
        fragments = session.query(DocumentFragment).filter(DocumentFragment.paper_id == paper_arxiv_id).all()
        print(f"   Total fragments: {len(fragments)}")
        for i, frag in enumerate(fragments[:3], 1):
            preview = frag.content[:80].replace('\n', ' ')
            print(f"   [{i}] Page {frag.page_number}: {preview}...")
        if len(fragments) > 3:
            print(f"   ... and {len(fragments) - 3} more")
        
        # Check images
        print("\n3. IMAGES:")
        images = session.query(ImageData).filter(ImageData.paper_id == paper_arxiv_id).all()
        print(f"   Total images: {len(images)}")
        for i, img in enumerate(images[:3], 1):
            print(f"   [{i}] Page {img.page_number}: {img.file_path}")
            if img.description:
                desc_preview = img.description[:60]
            else:
                desc_preview = "(no description)"
            print(f"        Description: {desc_preview}...")
        if len(images) > 3:
            print(f"   ... and {len(images) - 3} more")
        
        # Check tables
        print("\n4. TABLES:")
        tables = session.query(TableData).filter(TableData.paper_id == paper_arxiv_id).all()
        print(f"   Total tables: {len(tables)}")
        for i, tbl in enumerate(tables[:3], 1):
            content_preview = tbl.content[:60].replace('\n', ' ')
            print(f"   [{i}] Page {tbl.page_number}: {content_preview}...")
            if tbl.description:
                desc_preview = tbl.description[:60]
            else:
                desc_preview = "(no description)"
            print(f"        Summary: {desc_preview}...")
        if len(tables) > 3:
            print(f"   ... and {len(tables) - 3} more")
        
        session.close()
        
        # Summary
        print("\n" + "=" * 80)
        print("DIAGNOSIS SUMMARY:")
        print("=" * 80)
        
        if len(images) == 0:
            print("\n⚠️  NO IMAGES FOUND")
            print("   Possible causes:")
            print("   1. OCR didn't detect images in the PDF")
            print("   2. Images weren't saved to disk")
            print("   3. Images weren't recorded in the database")
            print("\n   Check: Is the extract_images from markdown working?")
        
        if len(tables) == 0:
            print("\n⚠️  NO TABLES FOUND")
            print("   Possible causes:")
            print("   1. OCR didn't extract tables from the PDF")
            print("   2. Tables weren't parsed correctly")
            print("   3. Tables weren't recorded in the database")
            print("\n   Check: Is the extract_tables_from_markdown working?")
        
        print(f"\n✅ Fragments stored: {len(fragments)}")
        print(f"✅ Images stored: {len(images)}")
        print(f"✅ Tables stored: {len(tables)}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_database()
