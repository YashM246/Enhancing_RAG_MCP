"""
LLM Integration Module

This module provides LLM-based tool selection for the RAG-MCP system.
The LLM acts as a final selector that chooses the best tool from the
top-k candidates retrieved by the semantic search system.

Main components:
- LLMToolSelector: Main class for tool selection via LLM API
- prompt_templates: Functions for formatting prompts
- response_parser: Utilities for parsing LLM responses

Usage:
    from src.llm import LLMToolSelector

    selector = LLMToolSelector(server_url="http://localhost:8000")
    result = selector.select_tool(query, candidate_tools)
    print(result["selected_tool"])
"""

# Import the main class to make it available when importing this module
from .llm_selector import LLMToolSelector

# Define what gets exported when someone does "from src.llm import *"
__all__ = ['LLMToolSelector']
