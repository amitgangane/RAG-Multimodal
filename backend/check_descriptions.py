"""Quick script to check if image/table descriptions are stored properly"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

from database_setup import init_postgres, ImageData, TableData
from sqlalchemy.orm import sessionmaker

def check_descriptions(arxiv_id="2308.08155"):
    engine = init_postgres()
    Session = sessionmaker(bind=engine)
    session = Session()

    print(f"\n{'='*60}")
    print(f"Checking descriptions for paper: {arxiv_id}")
    print(f"{'='*60}")

    # Check images
    images = session.query(ImageData).filter_by(paper_id=arxiv_id).all()
    print(f"\n📸 IMAGES ({len(images)} found):")
    for img in images:
        print(f"\n  ID: {img.id}")
        print(f"  Page: {img.page_number}")
        print(f"  Description: {img.description[:200]}..." if len(img.description) > 200 else f"  Description: {img.description}")

    # Check tables
    tables = session.query(TableData).filter_by(paper_id=arxiv_id).all()
    print(f"\n📊 TABLES ({len(tables)} found):")
    for tbl in tables:
        print(f"\n  ID: {tbl.id}")
        print(f"  Page: {tbl.page_number}")
        print(f"  Description: {tbl.description[:200]}..." if len(tbl.description) > 200 else f"  Description: {tbl.description}")

    session.close()

if __name__ == "__main__":
    arxiv_id = sys.argv[1] if len(sys.argv) > 1 else "2308.08155"
    check_descriptions(arxiv_id)
