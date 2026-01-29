"""Debug script to inspect OCR response structure"""
from ocr_processor import process_arxiv_to_markdown
import json

arxiv_url = "https://arxiv.org/pdf/1706.03762.pdf"

print("Testing OCR response structure...")
ocr_data = process_arxiv_to_markdown(arxiv_url)

print("\n=== OCR Response Type ===")
print(f"Type: {type(ocr_data)}")

print("\n=== Available Attributes ===")
print(f"Dir: {[attr for attr in dir(ocr_data) if not attr.startswith('_')]}")

print("\n=== Direct Attributes ===")
for attr in dir(ocr_data):
    if not attr.startswith('_'):
        try:
            value = getattr(ocr_data, attr)
            if not callable(value):
                print(f"{attr}: {type(value).__name__}")
                if attr == 'pages' and value:
                    print(f"  - Pages count: {len(value)}")
                    if len(value) > 0:
                        page = value[0]
                        print(f"  - First page type: {type(page).__name__}")
                        print(f"  - First page attributes: {[a for a in dir(page) if not a.startswith('_')]}")
        except Exception as e:
            print(f"{attr}: Error - {e}")

print("\n=== Full Response (first 1000 chars) ===")
print(str(ocr_data)[:1000])
