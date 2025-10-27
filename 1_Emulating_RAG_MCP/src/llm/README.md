# LLM Tool Selector Module

LLM-based tool selection for RAG-MCP system.
Supports both **vLLM** and **Ollama** backends.

**Multi-tool support:** Selects 1-3 tools per query based on requirements. LLM prefers fewer tools when possible.

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

# Example 1: Single tool query
tools = [
    {"tool_name": "arxiv_search", "description": "Search academic papers"},
    {"tool_name": "brave_search", "description": "Web search"}
]

result = selector.select_tool("Find AI papers", tools)
print(result["selected_tools"])      # ["arxiv_search"]
print(result["num_tools_selected"])  # 1

# Example 2: Multi-tool query
result = selector.select_tool("Search papers and save results to file", tools)
print(result["selected_tools"])      # ["arxiv_search", "file_writer"]
print(result["num_tools_selected"])  # 2
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

- `llm_selector.py` - Main LLMToolSelector class (supports both backends, multi-tool selection)
- `prompt_templates.py` - Prompt formatting (instructs LLM to select 1-3 tools)
- `response_parser.py` - Response parsing (handles tool arrays)

## Multi-Tool Selection

The LLM analyzes each query and selects the **minimum** number of tools needed:

- **1 tool**: Queries with a single clear objective
  - Example: "Find research papers on AI"
  - Result: `["arxiv_search"]`

- **2 tools**: Queries requiring two distinct actions
  - Example: "Search papers and email the results"
  - Result: `["arxiv_search", "email_sender"]`

- **3 tools** (maximum): Complex queries with three objectives
  - Example: "Query database, write to file, and send email"
  - Result: `["database_query", "file_writer", "email_sender"]`

Tools are returned in **priority order** (most important first).
