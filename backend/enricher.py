import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Reusable LLM instance
_llm = None

def get_llm():
    """Get or create LLM instance."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini")
    return _llm


def summarize_image(image_base64: str, page_num: int, surrounding_text: str = "") -> str:
    """
    Generate a description for an image using GPT-4o-mini vision.

    Args:
        image_base64: Base64 encoded image data
        page_num: Page number where image appears
        surrounding_text: Optional context from surrounding markdown

    Returns:
        Generated description of the image
    """
    # Use GPT-4o-mini for vision (faster and cheaper)
    vision_llm = ChatOpenAI(model="gpt-4o-mini")

    # Clean up base64 string - remove data URL prefix if present
    if image_base64.startswith("data:"):
        # Extract just the base64 part
        image_base64 = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64

    # Build prompt with context if available
    if surrounding_text:
        prompt = f"""Analyze this image from page {page_num} of a research paper.

Context from the paper:
"{surrounding_text[:1000]}"

Provide a concise description (2-3 sentences) that:
1. Describes what the image shows (diagram, chart, figure, etc.)
2. Explains its relevance to the paper's content
3. Highlights key data points or concepts illustrated"""
    else:
        prompt = f"""Analyze this image from page {page_num} of a research paper.

Provide a concise description (2-3 sentences) that:
1. Describes what the image shows (diagram, chart, figure, etc.)
2. Identifies key elements, labels, or data points
3. Explains what concept or result it likely illustrates"""

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
    ])

    try:
        response = vision_llm.invoke([message])
        print(f"   ✅ Image description generated successfully")
        return response.content
    except Exception as e:
        print(f"   ❌ Image description failed: {type(e).__name__}: {e}")
        return f"Figure on page {page_num}"


def summarize_table(table_content: str, page_num: int, surrounding_text: str = "") -> str:
    """
    Generate a description for a table using GPT-4o-mini.

    Args:
        table_content: Markdown formatted table content
        page_num: Page number where table appears
        surrounding_text: Optional context from surrounding markdown

    Returns:
        Generated description of the table
    """
    llm = get_llm()

    # Build prompt with context if available
    if surrounding_text:
        prompt = f"""Analyze this table from page {page_num} of a research paper.

Context from the paper:
"{surrounding_text[:500]}"

Table content:
{table_content[:2000]}

Provide a concise description (2-3 sentences) that:
1. Describes what data the table presents
2. Highlights key findings or comparisons shown
3. Explains its significance to the paper's results"""
    else:
        prompt = f"""Analyze this table from page {page_num} of a research paper.

Table content:
{table_content[:2000]}

Provide a concise description (2-3 sentences) that:
1. Describes what data the table presents
2. Identifies column headers and what they measure
3. Highlights any notable patterns or key values"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        print(f"   ⚠️  Table description failed: {e}")
        return f"Table on page {page_num}"


# Legacy function for backwards compatibility
def summarize_image_with_surroundings(image_base64, markdown_text, image_id):
    """Legacy function - use summarize_image instead."""
    return summarize_image(image_base64, page_num=1, surrounding_text=markdown_text)