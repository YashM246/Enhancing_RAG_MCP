# LLM Tool Selector Module

LLM-based tool selection for RAG-MCP system.

## Usage

```python
from src.llm import LLMToolSelector

# Initialize
selector = LLMToolSelector(server_url="http://localhost:8000")

# Select tool from candidates
tools = [
    {"tool_name": "arxiv_search", "description": "Search papers"},
    {"tool_name": "brave_search", "description": "Web search"}
]

result = selector.select_tool("Find AI papers", tools)
print(result["selected_tool"])  # "arxiv_search"
```

## Configuration

Update `.env` file:
```
VLLM_SERVER_URL=http://localhost:8000
VLLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
```

## Testing

```bash
pytest tests/test_llm_selector.py -v
```

All tests use mocks - no server required.

## Modules

- `llm_selector.py` - Main LLMToolSelector class
- `prompt_templates.py` - Prompt formatting
- `response_parser.py` - Response parsing
