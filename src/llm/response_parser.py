"""
Response Parser for LLM Outputs

This module provides utilities to parse tool selections from LLM responses.
Since open-source LLMs (like Mistral 7B) sometimes produce inconsistent
output formats, we implement multiple parsing strategies with fallbacks.

Supports multi-tool selection (1-3 tools per query).

Parsing strategies (tried in order):
1. Direct JSON parse - if response is clean JSON
2. Markdown code block extraction - if JSON is wrapped in ```json ... ```
3. Regex search - find JSON object anywhere in text
4. Fallback error - if all strategies fail

Example responses handled:
- '{"selected_tools": ["brave_search"]}'
- '{"selected_tools": ["arxiv_search", "file_writer"]}'
- '```json\n{"selected_tools": ["arxiv", "email"]}\n```'
- 'I select {"selected_tools": ["file_reader"]} for this task.'
"""

import json
import re
from typing import Dict, List


def parse_tool_selection_response(response_text: str) -> Dict:
    """
    Parse tool selection from LLM response using multiple strategies.

    This function tries multiple parsing approaches to handle various
    response formats from different LLMs. Supports multi-tool selection (1-3 tools).

    Args:
        response_text (str): Raw text response from the LLM

    Returns:
        Dict: A dictionary with the selected tools:
              {"selected_tools": ["tool_name1", "tool_name2", ...]}
              Array contains 1-3 tool names in priority order.

    Raises:
        ValueError: If the response cannot be parsed by any strategy,
                   or if more than 3 tools are selected

    Example:
        >>> parse_tool_selection_response('{"selected_tools": ["brave_search"]}')
        {'selected_tools': ['brave_search']}

        >>> parse_tool_selection_response('{"selected_tools": ["arxiv", "file_writer"]}')
        {'selected_tools': ['arxiv', 'file_writer']}

        >>> parse_tool_selection_response('```json\\n{"selected_tools": ["arxiv"]}\\n```')
        {'selected_tools': ['arxiv']}
    """
    # Validate input
    if not response_text or not response_text.strip():
        raise ValueError("Empty response from LLM")

    # Pre-processing: Fix escaped underscores (Ollama sometimes does this)
    # Changes: selected\_tool → selected_tool
    response_text = response_text.replace('\\_', '_')

    # Strategy 1: Direct JSON parse
    # Try to parse the response as pure JSON
    # Works for: {"selected_tools": ["brave_search", "arxiv"]}
    try:
        parsed = json.loads(response_text.strip())
        if "selected_tools" in parsed:
            tools = parsed["selected_tools"]
            # Validate: must be list with 1-3 items
            if not isinstance(tools, list):
                raise ValueError(f"selected_tools must be a list, got {type(tools)}")
            if len(tools) < 1:
                raise ValueError("selected_tools array is empty (need at least 1 tool)")
            if len(tools) > 3:
                raise ValueError(f"Too many tools selected ({len(tools)}). Maximum is 3.")
            # Ensure all items are strings
            if not all(isinstance(t, str) for t in tools):
                raise ValueError("All tools in selected_tools must be strings")
            return {"selected_tools": tools}
    except json.JSONDecodeError:
        # Not valid JSON, try next strategy
        pass

    # Strategy 2: Extract from markdown code block
    # Many LLMs wrap JSON in markdown code blocks
    # Works for: ```json\n{"selected_tools": ["..."]}\n```
    # Also works for: ```\n{"selected_tools": ["..."]}\n```
    markdown_match = re.search(
        r'```(?:json)?\s*(\{[^}]*"selected_tools"[^}]*\})\s*```',
        response_text,
        re.DOTALL | re.IGNORECASE
    )
    if markdown_match:
        try:
            # Extract the JSON string from the code block
            json_str = markdown_match.group(1)
            parsed = json.loads(json_str)
            if "selected_tools" in parsed:
                tools = parsed["selected_tools"]
                if isinstance(tools, list) and 1 <= len(tools) <= 3:
                    if all(isinstance(t, str) for t in tools):
                        return {"selected_tools": tools}
        except (json.JSONDecodeError, KeyError, ValueError):
            # Code block didn't contain valid JSON, continue to next strategy
            pass

    # Strategy 3: Regex search for JSON array
    # Find any JSON-like object with "selected_tools" array field
    # Works for: "I select {"selected_tools": ["brave_search", "arxiv"]} for this"
    json_match = re.search(
        r'"selected_tools"\s*:\s*\[(.*?)\]',
        response_text,
        re.DOTALL
    )
    if json_match:
        try:
            # Extract and parse the array contents
            array_content = json_match.group(1)
            # Parse as JSON array
            tools = json.loads(f'[{array_content}]')
            if isinstance(tools, list) and 1 <= len(tools) <= 3:
                if all(isinstance(t, str) for t in tools):
                    return {"selected_tools": tools}
        except (json.JSONDecodeError, ValueError):
            # Couldn't parse array, continue to fallback
            pass

    # All strategies failed - response is unparseable
    # Provide a helpful error message with response preview
    raise ValueError(
        f"Cannot parse tool selection from response. "
        f"Expected format: {{\"selected_tools\": [\"tool1\", \"tool2\", ...]}}. "
        f"Response (first 200 chars): {response_text[:200]}..."
    )


def validate_tool_selection(
    selected_tools: List[str],
    candidate_tools: list
) -> bool:
    """
    Validate that all selected tools are in the candidate list.

    This is a safety check to ensure the LLM selected only tools
    from the list we provided, not hallucinated tool names.

    Args:
        selected_tools (List[str]): Tool names selected by LLM (1-3 tools)
        candidate_tools (list): List of tool dicts that were provided to LLM

    Returns:
        bool: True if ALL selected_tools are valid, False otherwise

    Example:
        >>> tools = [{"tool_name": "brave_search"}, {"tool_name": "arxiv"}]
        >>> validate_tool_selection(["brave_search"], tools)
        True
        >>> validate_tool_selection(["brave_search", "arxiv"], tools)
        True
        >>> validate_tool_selection(["nonexistent_tool"], tools)
        False
        >>> validate_tool_selection(["brave_search", "nonexistent"], tools)
        False
    """
    # Extract tool names from candidate list
    tool_names = [t.get('tool_name', '') for t in candidate_tools]

    # Check if ALL selected tools are in the list
    return all(tool in tool_names for tool in selected_tools)


def extract_reasoning(response_text: str) -> str:
    """
    Extract reasoning or explanation from LLM response (optional).

    Some LLMs provide reasoning before the JSON output.
    This function attempts to extract that reasoning.

    Args:
        response_text (str): Raw text response from LLM

    Returns:
        str: Extracted reasoning text, or empty string if none found

    Example:
        >>> response = "I choose arxiv because it's for papers. {\\"selected_tool\\": \\"arxiv\\"}"
        >>> extract_reasoning(response)
        'I choose arxiv because it's for papers.'
    """
    # Find text before the JSON object
    json_match = re.search(r'\{[^}]*"selected_tool"[^}]*\}', response_text)

    if json_match:
        # Get everything before the JSON
        reasoning = response_text[:json_match.start()].strip()
        return reasoning

    return ""
