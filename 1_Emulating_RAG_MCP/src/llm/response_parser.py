"""
Response Parser for LLM Outputs

This module provides utilities to parse tool selections from LLM responses.
Since open-source LLMs (like Mistral 7B) sometimes produce inconsistent
output formats, we implement multiple parsing strategies with fallbacks.

Parsing strategies (tried in order):
1. Direct JSON parse - if response is clean JSON
2. Markdown code block extraction - if JSON is wrapped in ```json ... ```
3. Regex search - find JSON object anywhere in text
4. Fallback error - if all strategies fail

Example responses handled:
- '{"selected_tool": "brave_search"}'
- '```json\n{"selected_tool": "arxiv_search"}\n```'
- 'I select {"selected_tool": "file_reader"} for this task.'
"""

import json
import re
from typing import Dict


def parse_tool_selection_response(response_text: str) -> Dict:
    """
    Parse tool selection from LLM response using multiple strategies.

    This function tries multiple parsing approaches to handle various
    response formats from different LLMs. It's designed to be robust
    against formatting inconsistencies.

    Args:
        response_text (str): Raw text response from the LLM

    Returns:
        Dict: A dictionary with the selected tool:
              {"selected_tool": "tool_name"}

    Raises:
        ValueError: If the response cannot be parsed by any strategy

    Example:
        >>> parse_tool_selection_response('{"selected_tool": "brave_search"}')
        {'selected_tool': 'brave_search'}

        >>> parse_tool_selection_response('```json\\n{"selected_tool": "arxiv"}\\n```')
        {'selected_tool': 'arxiv'}
    """
    # Validate input
    if not response_text or not response_text.strip():
        raise ValueError("Empty response from LLM")

    # Pre-processing: Fix escaped underscores (Ollama sometimes does this)
    # Changes: selected\_tool → selected_tool
    response_text = response_text.replace('\\_', '_')

    # Strategy 1: Direct JSON parse
    # Try to parse the response as pure JSON
    # Works for: {"selected_tool": "brave_search"}
    try:
        parsed = json.loads(response_text.strip())
        if "selected_tool" in parsed:
            return {"selected_tool": parsed["selected_tool"]}
    except json.JSONDecodeError:
        # Not valid JSON, try next strategy
        pass

    # Strategy 2: Extract from markdown code block
    # Many LLMs wrap JSON in markdown code blocks
    # Works for: ```json\n{"selected_tool": "..."}\n```
    # Also works for: ```\n{"selected_tool": "..."}\n```
    markdown_match = re.search(
        r'```(?:json)?\s*(\{[^}]*"selected_tool"[^}]*\})\s*```',
        response_text,
        re.DOTALL | re.IGNORECASE
    )
    if markdown_match:
        try:
            # Extract the JSON string from the code block
            json_str = markdown_match.group(1)
            parsed = json.loads(json_str)
            return {"selected_tool": parsed["selected_tool"]}
        except (json.JSONDecodeError, KeyError):
            # Code block didn't contain valid JSON, continue to next strategy
            pass

    # Strategy 3: Regex search for JSON object
    # Find any JSON-like object with "selected_tool" field
    # Works for: "I think {"selected_tool": "brave_search"} is best"
    json_match = re.search(
        r'\{[^}]*"selected_tool"\s*:\s*"([^"]+)"[^}]*\}',
        response_text,
        re.DOTALL
    )
    if json_match:
        # Extract just the tool name from the regex match
        tool_name = json_match.group(1)
        return {"selected_tool": tool_name}

    # All strategies failed - response is unparseable
    # Provide a helpful error message with response preview
    raise ValueError(
        f"Cannot parse tool selection from response. "
        f"Response (first 200 chars): {response_text[:200]}..."
    )


def validate_tool_selection(
    selected_tool: str,
    candidate_tools: list
) -> bool:
    """
    Validate that the selected tool is in the candidate list.

    This is a safety check to ensure the LLM selected one of the
    tools we actually provided, not a hallucinated tool name.

    Args:
        selected_tool (str): Tool name selected by LLM
        candidate_tools (list): List of tool dicts that were provided to LLM

    Returns:
        bool: True if selected_tool is valid, False otherwise

    Example:
        >>> tools = [{"tool_name": "brave_search"}, {"tool_name": "arxiv"}]
        >>> validate_tool_selection("brave_search", tools)
        True
        >>> validate_tool_selection("nonexistent_tool", tools)
        False
    """
    # Extract tool names from candidate list
    tool_names = [t.get('tool_name', '') for t in candidate_tools]

    # Check if selected tool is in the list
    return selected_tool in tool_names


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
