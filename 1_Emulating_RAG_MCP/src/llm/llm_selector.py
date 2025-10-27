"""
LLM Tool Selector Module

This is the main module for LLM-based tool selection in the RAG-MCP system.
It provides the LLMToolSelector class which:
1. Connects to a vLLM/Ollama server (or compatible API) via HTTP
2. Formats prompts with query + candidate tools
3. Sends requests to the LLM for tool selection (1-3 tools)
4. Parses responses and tracks metrics

The LLM acts as a "final selector" that chooses 1-3 tools from
the top-k candidates retrieved by semantic search. The LLM selects
the MINIMUM number of tools needed for the query.

Supports both vLLM and Ollama backends automatically.
"""

import requests
from typing import List, Dict, Optional
from .prompt_templates import format_tool_selection_prompt
from .response_parser import parse_tool_selection_response


class LLMToolSelector:
    """
    Selects 1-3 appropriate tools from candidates using an LLM.

    This class connects to a vLLM or Ollama server (or any OpenAI-compatible API)
    and uses an LLM to make the final tool selection decision from a
    set of candidate tools retrieved by semantic search.

    The LLM selects the MINIMUM number of tools needed (1-3 max) for the query.

    Attributes:
        server_url (str): URL of the LLM server
        model_name (str): Name of the model being served
        backend (str): Backend type ('vllm' or 'ollama')
        timeout (int): Request timeout in seconds
        temperature (float): Sampling temperature (0.0-1.0)
        max_tokens (int): Maximum tokens in response
        total_prompt_tokens (int): Cumulative prompt tokens used
        total_completion_tokens (int): Cumulative completion tokens used

    Example (vLLM):
        >>> selector = LLMToolSelector(
        ...     server_url="http://localhost:8000",
        ...     backend="vllm"
        ... )

    Example (Ollama):
        >>> selector = LLMToolSelector(
        ...     server_url="http://localhost:11434",
        ...     model_name="mistral:7b-instruct-q4_0",
        ...     backend="ollama"
        ... )
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        backend: str = "vllm",
        timeout: int = 30,
        temperature: float = 0.1,
        max_tokens: int = 100
    ):
        """
        Initialize the LLM Tool Selector.

        Args:
            server_url: URL of the LLM server
                       vLLM: http://localhost:8000
                       Ollama: http://localhost:11434
            model_name: Name of the model
                       vLLM: mistralai/Mistral-7B-Instruct-v0.3
                       Ollama: mistral:7b-instruct-q4_0
            backend: Backend type - 'vllm' or 'ollama' (default: 'vllm')
            timeout: Request timeout in seconds (default: 30)
            temperature: Sampling temperature for LLM
                        Lower = more deterministic (default: 0.1)
                        0.0 = completely deterministic
                        1.0 = maximum randomness
            max_tokens: Maximum tokens in LLM response (default: 100)
                       Tool selection only needs ~20-50 tokens
        """
        # Store configuration
        self.server_url = server_url.rstrip('/')  # Remove trailing slash
        self.model_name = model_name
        self.backend = backend.lower()
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Validate backend
        if self.backend not in ["vllm", "ollama"]:
            raise ValueError(f"Unsupported backend: {backend}. Use 'vllm' or 'ollama'")

        # Initialize token tracking
        # These accumulate across all queries for evaluation metrics
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def select_tool(
        self,
        query: str,
        candidate_tools: List[Dict]
    ) -> Dict:
        """
        Select 1-3 tools from candidates using the LLM.

        The LLM will analyze the query and select the MINIMUM number of tools
        needed (1-3 maximum). Simple queries get 1 tool, complex queries requiring
        multiple distinct actions may get 2-3 tools.

        This is the main method that:
        1. Formats a prompt with the query and candidate tools
        2. Sends an HTTP request to the LLM server (vLLM or Ollama)
        3. Parses the LLM's response to extract 1-3 selected tools
        4. Tracks token usage for evaluation

        Args:
            query: The user's query/question
            candidate_tools: List of tool dicts from the retriever
                            Each tool should have:
                            - tool_name (str): Unique identifier
                            - description (str): What the tool does
                            - usage_example (str, optional): Example use

        Returns:
            Dict with keys:
                - selected_tools (List[str]): Names of 1-3 chosen tools (priority order)
                - num_tools_selected (int): How many tools were selected (1-3)
                - usage (dict): Token counts from LLM
                  - prompt_tokens (int)
                  - completion_tokens (int)
                  - total_tokens (int)
                - raw_response (str): Original LLM response

        Raises:
            ValueError: If no tools provided, response unparseable, or >3 tools selected
            TimeoutError: If LLM server request times out
            RuntimeError: If HTTP request fails

        Example (single tool):
            >>> selector = LLMToolSelector()
            >>> tools = [{"tool_name": "arxiv", "description": "Papers"}]
            >>> result = selector.select_tool("Find papers on AI", tools)
            >>> result["selected_tools"]
            ['arxiv']
            >>> result["num_tools_selected"]
            1

        Example (multiple tools):
            >>> result = selector.select_tool("Search papers and email results", tools)
            >>> result["selected_tools"]
            ['arxiv_search', 'email_sender']
            >>> result["num_tools_selected"]
            2
        """
        # Validation: Ensure we have tools to select from
        if not candidate_tools:
            raise ValueError("No candidate tools provided")

        # Step 1: Format the prompt
        # Convert query + tools into LLM-friendly format
        messages = format_tool_selection_prompt(query, candidate_tools)

        # Step 2: Call the LLM server via HTTP
        # This sends the prompt and receives the LLM's response
        response_text, usage = self._call_llm_server(messages)

        # Step 3: Parse the response
        # Extract 1-3 selected tool names from the response text
        result = parse_tool_selection_response(response_text)

        # Step 4: Track token usage
        # Accumulate tokens for evaluation metrics
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)

        # Step 5: Add metadata to result
        # Include usage stats, raw response, and count for convenience
        result["usage"] = usage
        result["raw_response"] = response_text
        result["num_tools_selected"] = len(result["selected_tools"])

        return result

    def _call_llm_server(self, messages: List[Dict]) -> tuple:
        """
        Make HTTP request to LLM server (vLLM or Ollama).

        This internal method handles the low-level HTTP communication
        with the LLM server. Both vLLM and Ollama support the
        OpenAI-compatible /v1/chat/completions endpoint.

        Args:
            messages: List of message dicts with 'role' and 'content'
                     Format: [
                         {"role": "system", "content": "..."},
                         {"role": "user", "content": "..."}
                     ]

        Returns:
            tuple: (response_text, usage_dict)
                  - response_text (str): The LLM's output text
                  - usage_dict (dict): Token usage statistics

        Raises:
            TimeoutError: If request exceeds timeout
            RuntimeError: If HTTP request fails
            ValueError: If response format is unexpected
        """
        # Prepare the request payload
        # This follows the OpenAI chat completion API format
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            # Make HTTP POST request to LLM server
            # Both vLLM and Ollama use the same endpoint
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )

            # Raise exception if HTTP status is not 200 OK
            response.raise_for_status()

            # Parse JSON response
            data = response.json()

            # Extract the LLM's text output
            # Response format: {"choices": [{"message": {"content": "..."}}]}
            content = data["choices"][0]["message"]["content"]

            # Extract token usage statistics
            # Format: {"usage": {"prompt_tokens": 100, "completion_tokens": 20}}
            usage = data.get("usage", {})

            return content, usage

        except requests.exceptions.Timeout:
            # Request took too long
            raise TimeoutError(
                f"{self.backend.upper()} server request timed out after {self.timeout}s. "
                f"Server may be overloaded or unreachable."
            )

        except requests.exceptions.RequestException as e:
            # Other HTTP errors (connection refused, network error, etc.)
            raise RuntimeError(
                f"{self.backend.upper()} server request failed: {e}. "
                f"Check if server is running at {self.server_url}"
            )

        except (KeyError, IndexError) as e:
            # Response JSON structure was unexpected
            raise ValueError(
                f"Unexpected {self.backend.upper()} response format: {e}. "
                f"Server may not be OpenAI-compatible."
            )

    def get_token_stats(self) -> Dict:
        """
        Get cumulative token usage statistics.

        This method returns the total tokens used across all queries
        since the selector was initialized. Used for evaluation metrics
        to compare RAG-MCP vs all-tools baseline.

        Returns:
            Dict with keys:
                - total_prompt_tokens (int): Sum of all prompt tokens
                - total_completion_tokens (int): Sum of all completion tokens
                - total_tokens (int): Sum of prompt + completion

        Example:
            >>> selector = LLMToolSelector()
            >>> # ... make several select_tool() calls ...
            >>> stats = selector.get_token_stats()
            >>> print(f"Used {stats['total_tokens']} tokens")
            Used 2547 tokens
        """
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens
        }

    def reset_token_stats(self):
        """
        Reset token usage counters to zero.

        Useful when starting a new experiment or evaluation run.

        Example:
            >>> selector = LLMToolSelector()
            >>> # Run experiment 1
            >>> selector.reset_token_stats()
            >>> # Run experiment 2 with fresh counters
        """
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
