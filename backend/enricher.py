import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

MAX_RETRIES = 6
MIN_WAIT = 2.0  # minimum seconds to wait on rate limit


def _create_llm():
    """Create a new ChatOpenAI instance (thread-safe: one per thread)."""
    return ChatOpenAI(model="gpt-4o-mini")


def _parse_retry_after(error_str):
    """Extract wait time in seconds from OpenAI rate limit error message."""
    match = re.search(r"try again in (\d+(?:\.\d+)?)\s*(ms|s)", error_str, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if match.group(2).lower() == "ms":
            val /= 1000
        return val
    return None


def _retry_on_rate_limit(fn, max_retries=MAX_RETRIES):
    """Call fn(), retrying on 429 RateLimitError with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                if attempt == max_retries:
                    raise
                # Use parsed wait time or exponential backoff, whichever is larger
                parsed = _parse_retry_after(str(e))
                backoff = MIN_WAIT * (2 ** attempt)
                wait = max(parsed + 0.5, backoff) if parsed else backoff
                print(f"   Rate limited (attempt {attempt + 1}/{max_retries}), waiting {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise


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
    vision_llm = _create_llm()

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
        response = _retry_on_rate_limit(lambda: vision_llm.invoke([message]))
        return response.content
    except Exception as e:
        print(f"   Image description failed: {type(e).__name__}: {e}")
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
    llm = _create_llm()

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
        response = _retry_on_rate_limit(lambda: llm.invoke([HumanMessage(content=prompt)]))
        return response.content
    except Exception as e:
        print(f"   Table description failed: {e}")
        return f"Table on page {page_num}"


def batch_summarize_images(image_items, max_workers=2):
    """
    Generate descriptions for multiple images in parallel using ThreadPoolExecutor.

    Args:
        image_items: List of (image_base64, page_num, surrounding_text) tuples
        max_workers: Max concurrent API calls (default=5 to respect rate limits)

    Returns:
        List of description strings in the same order as image_items
    """
    if not image_items:
        return []

    results = [None] * len(image_items)
    start_time = time.time()
    print(f"   Starting parallel image enrichment: {len(image_items)} images, {max_workers} workers")

    def _process_image(idx, item):
        image_base64, page_num, context = item
        desc = summarize_image(image_base64, page_num, context)
        return idx, desc

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_image, i, item): i
            for i, item in enumerate(image_items)
        }
        completed = 0
        for future in as_completed(futures):
            idx, desc = future.result()
            results[idx] = desc
            completed += 1
            print(f"   Image {completed}/{len(image_items)} done (index {idx})")

    elapsed = time.time() - start_time
    print(f"   Parallel image enrichment complete: {len(image_items)} images in {elapsed:.1f}s")
    return results


def batch_summarize_tables(table_items, max_workers=3):
    """
    Generate descriptions for multiple tables in parallel using ThreadPoolExecutor.

    Args:
        table_items: List of (table_content, page_num, surrounding_text) tuples
        max_workers: Max concurrent API calls (default=5 to respect rate limits)

    Returns:
        List of description strings in the same order as table_items
    """
    if not table_items:
        return []

    results = [None] * len(table_items)
    start_time = time.time()
    print(f"   Starting parallel table enrichment: {len(table_items)} tables, {max_workers} workers")

    def _process_table(idx, item):
        table_content, page_num, context = item
        desc = summarize_table(table_content, page_num, context)
        return idx, desc

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_table, i, item): i
            for i, item in enumerate(table_items)
        }
        completed = 0
        for future in as_completed(futures):
            idx, desc = future.result()
            results[idx] = desc
            completed += 1
            print(f"   Table {completed}/{len(table_items)} done (index {idx})")

    elapsed = time.time() - start_time
    print(f"   Parallel table enrichment complete: {len(table_items)} tables in {elapsed:.1f}s")
    return results


# Legacy function for backwards compatibility
def summarize_image_with_surroundings(image_base64, markdown_text, image_id):
    """Legacy function - use summarize_image instead."""
    return summarize_image(image_base64, page_num=1, surrounding_text=markdown_text)
