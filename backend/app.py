import uuid
from langchain_core.messages import HumanMessage
from chat_engine import research_agent

def chat_with_agent(user_query, arxiv_id, thread_id=None):
    # If no thread_id is passed, it's a "New Chat"
    if not thread_id:
        thread_id = str(uuid.uuid4())
        
    config = {"configurable": {"thread_id": thread_id}}
    
    inputs = {
        "messages": [HumanMessage(content=user_query)],
        "arxiv_id": arxiv_id
    }
    
    # Stream the graph execution
    final_state = research_agent.invoke(inputs, config=config)
    
    return {
        "answer": final_state["messages"][-1].content,
        "thread_id": thread_id # Return this to the UI to continue the conversation
    }