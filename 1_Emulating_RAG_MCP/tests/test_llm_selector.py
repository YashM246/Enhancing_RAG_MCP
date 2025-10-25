"""
Unit Tests for LLM Integration Module

These tests verify the LLM integration functionality WITHOUT requiring
a running vLLM server. We use Python's unittest.mock to simulate HTTP
responses from the server.

Test Coverage:
- Response parsing (various formats)
- Prompt formatting
- LLM tool selection with mocked responses
- Token tracking
- Error handling

Run tests with:
    pytest tests/test_llm_selector.py -v
"""

import pytest
from unittest.mock import Mock, patch
import requests

# Import the modules we're testing
from src.llm import LLMToolSelector
from src.llm.response_parser import parse_tool_selection_response, validate_tool_selection
from src.llm.prompt_templates import format_tool_selection_prompt


class TestResponseParser:
    """
    Test the response parsing module.

    This tests our ability to extract tool selections from various
    LLM response formats.
    """

    def test_parse_direct_json(self):
        """
        Test parsing a clean JSON response.

        This is the ideal case where the LLM returns perfect JSON.
        """
        response = '{"selected_tool": "brave_search"}'
        result = parse_tool_selection_response(response)

        assert result["selected_tool"] == "brave_search"
        assert isinstance(result, dict)

    def test_parse_markdown_json(self):
        """
        Test parsing JSON wrapped in markdown code block.

        Many LLMs wrap JSON in ```json ... ``` blocks.
        """
        response = '```json\n{"selected_tool": "arxiv_search"}\n```'
        result = parse_tool_selection_response(response)

        assert result["selected_tool"] == "arxiv_search"

    def test_parse_markdown_without_language(self):
        """
        Test parsing JSON in code block without 'json' specifier.

        Some LLMs use ``` without the language tag.
        """
        response = '```\n{"selected_tool": "file_reader"}\n```'
        result = parse_tool_selection_response(response)

        assert result["selected_tool"] == "file_reader"

    def test_parse_with_extra_text(self):
        """
        Test parsing when JSON is embedded in surrounding text.

        Some LLMs add explanation before/after the JSON.
        """
        response = 'I select {"selected_tool": "postgres_query"} for this task.'
        result = parse_tool_selection_response(response)

        assert result["selected_tool"] == "postgres_query"

    def test_parse_empty_response(self):
        """
        Test that empty response raises ValueError.

        The LLM should always return something.
        """
        with pytest.raises(ValueError, match="Empty response"):
            parse_tool_selection_response("")

    def test_parse_unparseable_response(self):
        """
        Test that unparseable response raises ValueError.

        If we can't extract a tool, we should fail explicitly.
        """
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_tool_selection_response("This has no tool selection at all")

    def test_validate_tool_selection_valid(self):
        """
        Test validation with a valid tool selection.

        The LLM selected a tool from our candidate list.
        """
        tools = [
            {"tool_name": "brave_search"},
            {"tool_name": "arxiv_search"}
        ]

        assert validate_tool_selection("brave_search", tools) is True

    def test_validate_tool_selection_invalid(self):
        """
        Test validation with an invalid tool selection.

        The LLM hallucinated a tool name not in our list.
        """
        tools = [
            {"tool_name": "brave_search"},
            {"tool_name": "arxiv_search"}
        ]

        assert validate_tool_selection("nonexistent_tool", tools) is False


class TestPromptTemplates:
    """
    Test the prompt formatting module.

    This verifies that we correctly format queries and tools into
    prompts for the LLM.
    """

    def test_format_tool_selection_prompt_structure(self):
        """
        Test that prompt has correct structure.

        Should return a list with system and user messages.
        """
        query = "Find papers on AI"
        tools = [
            {"tool_name": "arxiv_search", "description": "Search papers"},
            {"tool_name": "brave_search", "description": "Web search"}
        ]

        messages = format_tool_selection_prompt(query, tools)

        # Should have 2 messages: system and user
        assert len(messages) == 2

        # First message should be system role
        assert messages[0]["role"] == "system"
        assert "tool selection" in messages[0]["content"].lower()

        # Second message should be user role
        assert messages[1]["role"] == "user"
        assert "Find papers on AI" in messages[1]["content"]

    def test_format_includes_all_tools(self):
        """
        Test that all tools appear in the formatted prompt.

        The LLM needs to see all candidate tools to choose from.
        """
        query = "Search for something"
        tools = [
            {"tool_name": "tool1", "description": "First tool"},
            {"tool_name": "tool2", "description": "Second tool"},
            {"tool_name": "tool3", "description": "Third tool"}
        ]

        messages = format_tool_selection_prompt(query, tools)
        user_content = messages[1]["content"]

        # All tool names should appear in the prompt
        assert "tool1" in user_content
        assert "tool2" in user_content
        assert "tool3" in user_content

        # All descriptions should appear
        assert "First tool" in user_content
        assert "Second tool" in user_content
        assert "Third tool" in user_content

    def test_format_single_tool(self):
        """
        Test formatting with just one tool.

        Edge case: What if only 1 tool was retrieved?
        """
        query = "Do something"
        tools = [{"tool_name": "only_tool", "description": "The only option"}]

        messages = format_tool_selection_prompt(query, tools)

        # Should still work with single tool
        assert len(messages) == 2
        assert "only_tool" in messages[1]["content"]


class TestLLMToolSelector:
    """
    Test the main LLMToolSelector class with mocked HTTP responses.

    These tests simulate server responses without needing a real server.
    """

    @pytest.fixture
    def sample_tools(self):
        """
        Fixture providing sample tools for testing.

        This reusable test data represents typical retriever output.
        """
        return [
            {
                "tool_name": "brave_search",
                "description": "Web search using Brave API"
            },
            {
                "tool_name": "arxiv_search",
                "description": "Search academic papers on arXiv"
            },
            {
                "tool_name": "file_reader",
                "description": "Read files from local filesystem"
            }
        ]

    @patch('src.llm.llm_selector.requests.post')
    def test_select_tool_success(self, mock_post, sample_tools):
        """
        Test successful tool selection with mocked HTTP response.

        This simulates a perfect response from the vLLM server.

        The @patch decorator replaces requests.post with a mock object,
        so we don't actually make HTTP requests.
        """
        # Create a mock response object
        mock_response = Mock()

        # Define what the mock should return when .json() is called
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": '{"selected_tool": "arxiv_search"}'}}
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120
            }
        }

        # Define what the mock should do when .raise_for_status() is called
        mock_response.raise_for_status = Mock()

        # Make mock_post return our mock_response
        mock_post.return_value = mock_response

        # Now test the actual functionality
        selector = LLMToolSelector()
        result = selector.select_tool("Find papers on AI", sample_tools)

        # Verify the result
        assert result["selected_tool"] == "arxiv_search"
        assert result["usage"]["prompt_tokens"] == 100
        assert result["usage"]["completion_tokens"] == 20
        assert "raw_response" in result

    @patch('src.llm.llm_selector.requests.post')
    def test_select_tool_with_markdown_response(self, mock_post, sample_tools):
        """
        Test tool selection when LLM returns markdown-wrapped JSON.

        Some LLMs wrap JSON in code blocks - our parser should handle this.
        """
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {"message": {"content": '```json\n{"selected_tool": "brave_search"}\n```'}}
            ],
            "usage": {"prompt_tokens": 95, "completion_tokens": 18}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        selector = LLMToolSelector()
        result = selector.select_tool("Search the web", sample_tools)

        # Should successfully parse despite markdown wrapper
        assert result["selected_tool"] == "brave_search"

    @patch('src.llm.llm_selector.requests.post')
    def test_token_tracking(self, mock_post, sample_tools):
        """
        Test that token usage is tracked correctly across multiple queries.

        This is important for evaluation metrics.
        """
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"selected_tool": "arxiv_search"}'}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        selector = LLMToolSelector()

        # First query
        selector.select_tool("Query 1", sample_tools)
        assert selector.total_prompt_tokens == 100
        assert selector.total_completion_tokens == 20

        # Second query (tokens should accumulate)
        selector.select_tool("Query 2", sample_tools)
        assert selector.total_prompt_tokens == 200  # 100 + 100
        assert selector.total_completion_tokens == 40  # 20 + 20

        # Get stats
        stats = selector.get_token_stats()
        assert stats["total_tokens"] == 240  # 200 + 40

    def test_no_tools_error(self):
        """
        Test that providing no tools raises ValueError.

        We need at least one tool to select from.
        """
        selector = LLMToolSelector()

        with pytest.raises(ValueError, match="No candidate tools"):
            selector.select_tool("some query", [])

    @patch('src.llm.llm_selector.requests.post')
    def test_http_timeout(self, mock_post, sample_tools):
        """
        Test handling of HTTP timeout errors.

        If the server is slow or overloaded, requests will timeout.
        """
        # Make the mock raise a Timeout exception
        mock_post.side_effect = requests.exceptions.Timeout()

        selector = LLMToolSelector(timeout=5)

        with pytest.raises(TimeoutError, match="timed out"):
            selector.select_tool("Query", sample_tools)

    @patch('src.llm.llm_selector.requests.post')
    def test_http_connection_error(self, mock_post, sample_tools):
        """
        Test handling of connection errors.

        If the server is not running, we'll get a connection error.
        """
        # Make the mock raise a ConnectionError
        mock_post.side_effect = requests.exceptions.ConnectionError()

        selector = LLMToolSelector()

        with pytest.raises(RuntimeError, match="failed"):
            selector.select_tool("Query", sample_tools)

    @patch('src.llm.llm_selector.requests.post')
    def test_invalid_response_format(self, mock_post, sample_tools):
        """
        Test handling of unexpected response format.

        If the server returns something we don't expect, handle gracefully.
        """
        mock_response = Mock()
        # Return malformed response (missing expected keys)
        mock_response.json.return_value = {"unexpected": "format"}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        selector = LLMToolSelector()

        with pytest.raises(ValueError, match="Unexpected"):
            selector.select_tool("Query", sample_tools)

    def test_reset_token_stats(self):
        """
        Test resetting token counters.

        Useful when starting a new experiment.
        """
        selector = LLMToolSelector()

        # Manually set some values
        selector.total_prompt_tokens = 1000
        selector.total_completion_tokens = 500

        # Reset
        selector.reset_token_stats()

        # Should be zero
        assert selector.total_prompt_tokens == 0
        assert selector.total_completion_tokens == 0


# Run tests with: pytest tests/test_llm_selector.py -v
