import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

def summarize_image_with_surroundings(image_base64, markdown_text, image_id):
    """
    Finds the image tag in markdown and sends it + surrounding context to Gemini.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

    # Logic to find the text block (e.g., 500 chars) near the image placeholder
    placeholder = f"![{image_id}]"
    idx = markdown_text.find(placeholder)
    context_text = markdown_text[max(0, idx-500):min(len(markdown_text), idx+500)]

    prompt = f"""You are a technical expert. Analyze this image using the following 
    context from the research paper: "{context_text}". 
    Explain the data shown and its specific relevance to the paper's findings."""

    message = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
    ])

    return llm.invoke([message]).content