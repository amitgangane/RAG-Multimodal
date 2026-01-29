from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from retriever import retrieve_hybrid_context
import json

# 1. Define State
class AgentState(TypedDict):
    # add_messages ensures conversation history is appended, not overwritten
    messages: Annotated[List[BaseMessage], add_messages]
    context: Dict[str, Any]  # Now contains text_chunks, images, tables, pages_referenced
    arxiv_id: str

# 2. Nodes
def retrieve_node(state: AgentState):
    """Fetches text chunks, images, and tables from Pinecone + Postgres."""
    last_query = state["messages"][-1].content
    
    # Use hybrid retriever that returns text, images, tables, and page references
    results = retrieve_hybrid_context(last_query, state["arxiv_id"])
    
    return {"context": results}

def format_context_for_llm(context: Dict[str, Any]) -> str:
    """Format retrieved context into a readable prompt for the LLM."""
    formatted = "📚 RETRIEVED CONTEXT:\n\n"
    
    # Format text chunks
    if context.get("text_chunks"):
        formatted += "📄 RELEVANT TEXT:\n"
        for i, chunk in enumerate(context["text_chunks"], 1):
            formatted += f"\n[Chunk {i} - Page {chunk.get('page', '?')}]\n"
            formatted += f"{chunk['content'][:500]}...\n"
            formatted += f"(Similarity: {chunk.get('score', 0):.2%})\n"
    
    # Format tables
    if context.get("tables"):
        formatted += "\n📊 RELEVANT TABLES:\n"
        for i, table in enumerate(context["tables"], 1):
            formatted += f"\n[Table {i} - Page {table.get('page', '?')}]\n"
            formatted += f"Description: {table.get('description', 'N/A')}\n"
            formatted += f"Content:\n{table.get('content', 'N/A')[:300]}\n"
    
    # Format images
    if context.get("images"):
        formatted += "\n🖼️  RELEVANT IMAGES:\n"
        for i, image in enumerate(context["images"], 1):
            formatted += f"\n[Image {i} - Page {image.get('page', '?')}]\n"
            formatted += f"Description: {image.get('description', 'N/A')}\n"
            formatted += f"Location: {image.get('file_path', 'N/A')}\n"
    
    # Add page references for PDF highlighting
    if context.get("pages_referenced"):
        formatted += f"\n📍 Referenced Pages: {', '.join(map(str, context['pages_referenced']))}\n"
    
    return formatted

def generate_node(state: AgentState):
    """Generates a comprehensive answer using retrieved context and message history."""
    llm = ChatOpenAI(model="gpt-4o-mini")  # Using cost-effective OpenAI model
    
    # Format the retrieved context
    formatted_context = format_context_for_llm(state["context"])
    
    # Create a detailed system prompt
    system_prompt = f"""You are an expert research assistant analyzing academic papers.

Your task is to provide detailed, accurate answers using the provided context.

Instructions:
1. Base your answer ONLY on the provided context
2. Cite the pages where information comes from
3. If context includes tables or images, explain their relevance
4. Provide specific examples and data from the text
5. Be comprehensive but concise

{formatted_context}"""
    
    # Get the user's question
    user_query = state["messages"][-1].content
    
    # Invoke LLM with system prompt and conversation history
    response = llm.invoke([
        HumanMessage(content=system_prompt),
        *state["messages"]  # Include full conversation history
    ])
    
    return {"messages": [response]}

# 3. Build Graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# 4. Compile with optional Persistence
try:
    from database_setup import get_graph_checkpointer
    checkpointer = get_graph_checkpointer()
    research_agent = workflow.compile(checkpointer=checkpointer)
except ImportError:
    # If LangGraph checkpointer not available, compile without persistence
    research_agent = workflow.compile()