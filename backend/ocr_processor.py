import os
import base64
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


def process_pdf_file(file_path: str):
    """
    Process a local PDF file via Mistral OCR.

    Args:
        file_path: Path to the local PDF file

    Returns:
        Structured OCR response containing markdown and images
    """
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    # Read and encode the PDF file as base64
    with open(file_path, "rb") as f:
        pdf_content = f.read()
    pdf_base64 = base64.b64encode(pdf_content).decode("utf-8")

    # Mistral OCR process call with base64 document
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "base64",
            "base64": pdf_base64,
            "name": os.path.basename(file_path)
        },
        include_image_base64=True,
        table_format="markdown"
    )

    return ocr_response


if __name__ == "__main__":
    ocr_data = process_arxiv_to_markdown("https://arxiv.org/pdf/1706.03762")
    print(ocr_data.pages.markdown)  # Example: print markdown of first page