"""
LangSmith Tracing Configuration

To enable tracing, set these environment variables in your .env file:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=your_langsmith_api_key
    LANGCHAIN_PROJECT=rag-multimodal
"""

import os
from dotenv import load_dotenv

load_dotenv()


def setup_langsmith():
    """
    Initialize LangSmith tracing.

    LangSmith automatically traces all LangChain operations when enabled.
    Get your API key from: https://smith.langchain.com/
    """
    api_key = os.getenv("LANGCHAIN_API_KEY")
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    project_name = os.getenv("LANGCHAIN_PROJECT", "rag-multimodal")

    if tracing_enabled and api_key:
        # These are already set via environment, but let's ensure they're configured
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project_name
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

        print(f"✅ LangSmith tracing enabled for project: {project_name}")
        print(f"   View traces at: https://smith.langchain.com/")
        return True
    else:
        if not api_key:
            print("ℹ️  LangSmith tracing disabled (no LANGCHAIN_API_KEY set)")
        elif not tracing_enabled:
            print("ℹ️  LangSmith tracing disabled (LANGCHAIN_TRACING_V2 != true)")
        return False


def get_tracing_status():
    """Check if LangSmith tracing is currently enabled."""
    return os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
