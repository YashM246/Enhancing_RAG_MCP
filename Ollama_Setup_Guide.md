# Ollama Setup Guide - Mistral 7B

Simple guide to run Mistral-7B locally with Ollama for the LLM Tool Selector.

## Prerequisites

- **GPU**: 6GB+ VRAM (recommended) OR CPU with 16GB+ RAM
- **OS**: Windows, macOS, or Linux

## Installation

### Step 1: Install Ollama

**Windows/macOS:**
- Download from: https://ollama.com/download
- Run installer

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 2: Pull Mistral 7B Model

```bash
# Quantized version (recommended - uses ~3.5GB VRAM)
ollama pull mistral:7b-instruct-q4_0

# OR full version (uses ~4GB VRAM)
ollama pull mistral:7b-instruct
```

### Step 3: Run the Model

```bash
# Start Ollama server (runs on port 11434 by default)
ollama serve
```

The server will automatically use GPU if available.

### Step 4: Test It

```bash
# In a new terminal
ollama run mistral:7b-instruct-q4_0
```

Type a message to test. Use `/bye` to exit.

## Configuration for RAG-MCP

Update your `.env` file:

```bash
LLM_BACKEND=ollama
LLM_SERVER_URL=http://localhost:11434
LLM_MODEL_NAME=mistral:7b-instruct-q4_0
```

## Usage in Python

```python
from src.llm import LLMToolSelector

selector = LLMToolSelector(
    server_url="http://localhost:11434",
    model_name="mistral:7b-instruct-q4_0",
    backend="ollama"
)

# Example usage (multi-tool support)
tools = [
    {"tool_name": "arxiv_search", "description": "Search papers"},
    {"tool_name": "file_writer", "description": "Write files"}
]

# Single tool query
result = selector.select_tool("Find AI papers", tools)
print(result["selected_tools"])      # ["arxiv_search"]
print(result["num_tools_selected"])  # 1

# Multi-tool query
result = selector.select_tool("Find papers and save them", tools)
print(result["selected_tools"])      # ["arxiv_search", "file_writer"]
print(result["num_tools_selected"])  # 2
```

**Note:** The LLM selects 1-3 tools per query, preferring fewer tools when possible.

## Performance Tips

### For GPU (Your Setup: 8GB VRAM)
- Use `mistral:7b-instruct-q4_0` (best balance)
- Expected speed: 1-2 seconds per query
- VRAM usage: ~3.5GB

### For CPU Only
```bash
# Force CPU usage
OLLAMA_NUM_GPU=0 ollama serve
```
- Expected speed: 10-30 seconds per query
- RAM usage: ~8GB

## Troubleshooting

**Server not starting?**
```bash
# Check if port 11434 is in use
netstat -ano | findstr :11434  # Windows
lsof -i :11434                # Mac/Linux
```

**Out of memory?**
```bash
# Use smaller quantized model
ollama pull mistral:7b-instruct-q2_K
```

**Slow on GPU?**
```bash
# Check GPU is being used
ollama ps  # Should show GPU:0
```

## Model Variants

| Model | Size | VRAM | Speed | Quality |
|-------|------|------|-------|---------|
| `mistral:7b-instruct-q2_K` | 2.7GB | ~3GB | Fastest | 85% |
| `mistral:7b-instruct-q4_0` | 3.8GB | ~4GB | Fast | 95% |
| `mistral:7b-instruct-q8_0` | 7.7GB | ~8GB | Medium | 99% |
| `mistral:7b-instruct` | 14GB | ~14GB | Slow | 100% |

**Recommendation:** Use `q4_0` for your 8GB VRAM system.

## Additional Resources

- Ollama Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
- Model Library: https://ollama.com/library/mistral
- API Docs: https://github.com/ollama/ollama/blob/main/docs/openai.md
