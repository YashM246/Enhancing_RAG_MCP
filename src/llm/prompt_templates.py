"""
Prompt Templates for LLM Tool Selection

This module provides functions to format prompts for the LLM.
The prompt includes the user's query and a list of candidate tools
retrieved by the semantic search system.

The LLM's job is to analyze the query and select 1-3 tools based on
the query's requirements. The LLM should prefer fewer tools when possible.
"""

from typing import List, Dict


def format_tool_selection_prompt(query: str, tools: List[Dict]) -> List[Dict]:
    """
    Format a tool selection prompt for the LLM (multi-tool support).

    This function creates a structured prompt that:
    1. Explains the LLM's role (tool selector)
    2. Presents the user's query
    3. Lists all candidate tools with descriptions
    4. Requests a JSON response with 1-3 selected tools

    Args:
        query (str): The user's query/question
        tools (List[Dict]): List of candidate tool dictionaries.
                           Each tool should have:
                           - tool_name (str): Unique identifier
                           - description (str): What the tool does
                           - usage_example (str, optional): Example use case

    Returns:
        List[Dict]: A list of message dictionaries in OpenAI chat format:
                    [
                        {"role": "system", "content": "..."},
                        {"role": "user", "content": "..."}
                    ]

    Example:
        >>> tools = [
        ...     {"tool_name": "brave_search", "description": "Web search"},
        ...     {"tool_name": "arxiv_search", "description": "Academic papers"}
        ... ]
        >>> messages = format_tool_selection_prompt("Find papers on AI", tools)
        >>> len(messages)
        2
    """
    # System message: Defines the LLM's role and output format
    # Emphasizes selecting minimum tools needed (1-3 max)
    system_msg = """You are a tool selection assistant.
Analyze the query and select the MINIMUM number of tools needed (1-3 maximum).

Rules:
- Use 1 tool if the query has a single, clear objective
- Use 2-3 tools only if the query explicitly requires multiple distinct actions
- Return tools in priority order (most important first)
- YOU MUST use the EXACT tool_name as listed (case-sensitive, including spaces and capitalization)
- DO NOT modify, abbreviate, or reformat tool names

Return ONLY JSON: {"selected_tools": ["Exact Tool Name", "Another Tool Name", ...]}

CRITICAL: Copy tool names EXACTLY as shown in the tool list. Do not lowercase, add underscores, or change formatting."""

    # Format the tool list as numbered items
    # Example output:
    # 1. brave_search: Search the web using Brave API
    # 2. arxiv_search: Search academic papers on arXiv
    tool_list = "\n".join([
        f"{i+1}. {tool['tool_name']}: {tool['description']}"
        for i, tool in enumerate(tools)
    ])

    # User message: Contains the actual query and tool options
    # The LLM will analyze this to make its selection
    user_msg = f"""Query: {query}

Available Tools (use EXACT names as shown):
{tool_list}

Return JSON with EXACT tool names:"""

    # Return in OpenAI chat completion format
    # This format is compatible with vLLM and most LLM APIs
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]


def format_tool_selection_prompt_verbose(query: str, tools: List[Dict]) -> List[Dict]:
    """
    Alternative verbose prompt template with more detailed instructions.

    Use this if the LLM produces verbose or incorrect responses with the
    simple template. This version includes usage examples and more explicit
    instructions for multi-tool selection.

    Args:
        query (str): The user's query/question
        tools (List[Dict]): List of candidate tool dictionaries

    Returns:
        List[Dict]: Messages in OpenAI chat format
    """
    system_msg = """You are a tool selection assistant. Your job is to analyze
a user's query and select the MINIMUM number of tools needed (1-3 maximum).

Instructions:
1. Read the user's query carefully
2. Review all available tools and their descriptions
3. Determine if the query requires 1, 2, or 3 tools:
   - Use 1 tool if the query has ONE clear objective
   - Use 2-3 tools only if the query explicitly requires MULTIPLE distinct actions
4. Return ONLY a JSON object with this exact format: {"selected_tools": ["tool1", "tool2", ...]}

CRITICAL REQUIREMENTS:
- YOU MUST use the EXACT tool_name as listed (case-sensitive, including all spaces and capitalization)
- DO NOT modify, lowercase, abbreviate, add underscores, or reformat tool names in ANY way
- COPY tool names EXACTLY character-for-character from the provided list
- Prefer fewer tools when possible (minimize, don't maximize)
- Return tools in priority order (most important first)
- Do not add any explanation or additional text
- Maximum 3 tools allowed

WRONG: {"selected_tools": ["weather_api"]}  <- modified name
RIGHT: {"selected_tools": ["Weather API"]}  <- exact name from list"""

    # Format tools with more detail including usage examples
    tool_list = []
    for i, tool in enumerate(tools, 1):
        tool_name = tool.get('tool_name', 'unknown')
        description = tool.get('description', 'No description')
        usage_example = tool.get('usage_example', '')

        # Include usage example if available
        if usage_example:
            tool_entry = f"{i}. **{tool_name}**\n   Description: {description}\n   Example: {usage_example}"
        else:
            tool_entry = f"{i}. **{tool_name}**\n   Description: {description}"

        tool_list.append(tool_entry)

    tools_formatted = "\n\n".join(tool_list)

    user_msg = f"""User Query: "{query}"

Available Tools (copy names EXACTLY as shown):
{tools_formatted}

Select the minimum tools needed (1-3 max) and respond with JSON using EXACT tool names: {{"selected_tools": ["Exact Name", "Another Exact Name", ...]}}"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
