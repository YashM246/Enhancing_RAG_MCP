# LLM Tool Selector Module

LLM-based tool selection for RAG-MCP system.
Supports both **vLLM** and **Ollama** backends.

## Usage

### Ollama (Recommended for Local Development)

```python
from src.llm import LLMToolSelector

# Initialize with Ollama
selector = LLMToolSelector(
    server_url="http://localhost:11434",
    model_name="mistral:7b-instruct-q4_0",
    backend="ollama"
)

# Select tool from candidates
tools = [
    {"tool_name": "arxiv_search", "description": "Search papers"},
    {"tool_name": "brave_search", "description": "Web search"}
]

result = selector.select_tool("Find AI papers", tools)
print(result["selected_tool"])  # "arxiv_search"
```

### vLLM (For Production/HPC)

```python
selector = LLMToolSelector(
    server_url="http://localhost:8000",
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    backend="vllm"
)
```

## Configuration

Update `.env` file:
```bash
# For Ollama (local development)
LLM_BACKEND=ollama
LLM_SERVER_URL=http://localhost:11434
LLM_MODEL_NAME=mistral:7b-instruct-q4_0

# For vLLM (production/HPC)
LLM_BACKEND=vllm
LLM_SERVER_URL=http://localhost:8000
LLM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
```

## Setup Instructions

### Option 1: Ollama (Easy Setup)

See [Ollama Setup Guide](../../Ollama_Setup_Guide.md)

### Option 2: vLLM (Advanced)

See [vLLM Setup Guide](../../Running_Mistral7B_on_CARC_Guide.md)

## Testing

```bash
pytest tests/test_llm_selector.py -v
```

All tests use mocks - no server required.

## Modules

- `llm_selector.py` - Main LLMToolSelector class (supports both backends)
- `prompt_templates.py` - Prompt formatting
- `response_parser.py` - Response parsing
