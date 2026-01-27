from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

def generate_rag_chunks(markdown_content):
    # Split by headers to preserve semantic hierarchy
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    sections = header_splitter.split_text(markdown_content)
    
    # Final sub-chunking for vector database compatibility
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents(sections)