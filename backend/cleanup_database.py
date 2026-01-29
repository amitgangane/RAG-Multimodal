"""
Cleanup: Delete mock test data from database
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

def cleanup_database():
    try:
        print("=" * 80)
        print("CLEANING UP DATABASE - REMOVING MOCK TEST DATA")
        print("=" * 80)
        
        from database_setup import init_postgres, Paper, DocumentFragment, ImageData, TableData
        from sqlalchemy.orm import sessionmaker
        
        print("\n1. Connecting to database...")
        engine = init_postgres()
        Session = sessionmaker(bind=engine)
        session = Session()
        print("   ✅ Connected")
        
        # Show what we're about to delete
        print("\n2. Scanning for mock data...")
        papers = session.query(Paper).all()
        print(f"   📊 Total papers in DB: {len(papers)}")
        
        for paper in papers:
            print(f"      - {paper.arxiv_id}: {paper.title}")
        
        # Delete all data
        print("\n3. Deleting all data...")
        
        # Count before deletion
        image_count = session.query(ImageData).count()
        table_count = session.query(TableData).count()
        fragment_count = session.query(DocumentFragment).count()
        paper_count = session.query(Paper).count()
        
        print(f"   Deleting ImageData entries: {image_count}")
        session.query(ImageData).delete()
        
        print(f"   Deleting TableData entries: {table_count}")
        session.query(TableData).delete()
        
        print(f"   Deleting DocumentFragment entries: {fragment_count}")
        session.query(DocumentFragment).delete()
        
        print(f"   Deleting Paper entries: {paper_count}")
        session.query(Paper).delete()
        
        session.commit()
        print("\n   ✅ All data deleted")
        
        # Verify deletion
        print("\n4. Verifying deletion...")
        remaining_papers = session.query(Paper).count()
        remaining_fragments = session.query(DocumentFragment).count()
        remaining_images = session.query(ImageData).count()
        remaining_tables = session.query(TableData).count()
        
        print(f"   Papers remaining: {remaining_papers}")
        print(f"   Fragments remaining: {remaining_fragments}")
        print(f"   Images remaining: {remaining_images}")
        print(f"   Tables remaining: {remaining_tables}")
        
        session.close()

        # 5. Clean up Pinecone index
        print("\n5. Cleaning up Pinecone index...")
        try:
            from pinecone import Pinecone
            PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
            INDEX_NAME = "sagemind-research-index"

            pc = Pinecone(api_key=PINECONE_API_KEY)

            if INDEX_NAME in pc.list_indexes().names():
                print(f"   Deleting Pinecone index '{INDEX_NAME}'...")
                pc.delete_index(INDEX_NAME)
                print(f"   ✅ Pinecone index deleted")
            else:
                print(f"   ℹ️  Pinecone index '{INDEX_NAME}' does not exist")
        except Exception as e:
            print(f"   ⚠️  Could not clean Pinecone: {e}")

        if remaining_papers == 0:
            print("\n" + "=" * 80)
            print("✅ DATABASE CLEANUP COMPLETE!")
            print("=" * 80)
            print("\n📋 SUMMARY:")
            print(f"   ✅ Deleted {paper_count} papers")
            print(f"   ✅ Deleted {fragment_count} fragments")
            print(f"   ✅ Deleted {image_count} images")
            print(f"   ✅ Deleted {table_count} tables")
            print(f"   ✅ Database is now CLEAN and ready for fresh data")
            
            print("\n💡 NEXT STEPS:")
            print("   1. Run test_ingestion_mock.py to test with fresh mock data")
            print("   2. Or use ingest_paper_to_production() to add real arXiv papers")
            
            return True
        else:
            print("\n❌ Cleanup incomplete - data still exists")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    confirm = input("⚠️  WARNING: This will DELETE ALL data from the database!\n   Type 'yes' to confirm: ")
    
    if confirm.lower() == 'yes':
        success = cleanup_database()
        sys.exit(0 if success else 1)
    else:
        print("❌ Cleanup cancelled")
        sys.exit(1)
