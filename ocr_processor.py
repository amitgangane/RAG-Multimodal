import os
import requests
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()

def process_arxiv_to_markdown(arxiv_url):
    """
    Downloads arXiv PDF and processes it via Mistral OCR.
    """
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    
    # Mistral OCR process call
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": arxiv_url
        },
        include_image_base64=True,  # Crucial for enrichment
        table_format="markdown"    # Best format for RAG reasoning
    )
    
    # Returns the structured OCR response containing markdown and images
    return ocr_response

if __name__ == "__main__":
    ocr_data = process_arxiv_to_markdown("https://arxiv.org/pdf/1706.03762")
    print(ocr_data.pages.markdown)  # Example: print markdown of first page