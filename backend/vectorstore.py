import os
from pinecone import Pinecone, ServerlessSpec

# Configuration - Replace with your actual Pinecone API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "sagemind-research-index"

def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,  # Google text-embedding-004 outputs 768 dimensions
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"Pinecone index '{INDEX_NAME}' created.")
    else:
        print(f"Pinecone index '{INDEX_NAME}' already exists.")
    
    return pc.Index(INDEX_NAME)

# index = init_pinecone()