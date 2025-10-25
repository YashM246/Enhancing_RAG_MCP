"""
Prompt Templates for LLM Tool Selection

This module provides functions to format prompts for the LLM.
The prompt includes the user's query and a list of candidate tools
retrieved by the semantic search system.

The LLM's job is to analyze the query and select the most appropriate
tool from the candidates.
"""

from typing import List, Dict


def format_tool_selection_prompt(query: str, tools: List[Dict]) -> List[Dict]:
    """
    Format a tool selection prompt for the LLM.

    This function creates a structured prompt that:
    1. Explains the LLM's role (tool selector)
    2. Presents the user's query
    3. Lists all candidate tools with descriptions
    4. Requests a JSON response with the selected tool

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
    # This stays the same for all queries
    system_msg = """You are a tool selection assistant.
Select the most appropriate tool from the list.
Return ONLY JSON: {"selected_tool": "tool_name"}"""

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

Tools:
{tool_list}

JSON response:"""

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
    instructions.

    Args:
        query (str): The user's query/question
        tools (List[Dict]): List of candidate tool dictionaries

    Returns:
        List[Dict]: Messages in OpenAI chat format
    """
    system_msg = """You are a tool selection assistant. Your job is to analyze
a user's query and select the most appropriate tool from a list of available tools.

Instructions:
1. Read the user's query carefully
2. Review all available tools and their descriptions
3. Select the ONE tool that best matches the query's intent
4. Return ONLY a JSON object with this exact format: {"selected_tool": "exact_tool_name"}

Important:
- Use the exact tool name as provided
- Do not add any explanation or additional text
- If unsure, choose the closest match"""

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

Available Tools:
{tools_formatted}

Select the best tool and respond with JSON: {{"selected_tool": "exact_tool_name"}}"""

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
