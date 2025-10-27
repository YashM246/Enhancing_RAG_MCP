# Benchmarking Module

## Quick Start

```python
from benchmarking import Benchmarker

benchmarker = Benchmarker()
results = benchmarker.benchmark_retrieval_only(
    tools_json_path="data/tools/sample_tools.json",
    queries_json_path="data/queries/test_queries.json",
    model_name='all-MiniLM-L6-v2',
    k_values=[1, 3, 5]
)
```

Or run from command line:
```bash
python benchmarking/benchmarker.py
```

## Data Format

**Tools JSON** (`data/tools/sample_tools.json`):
```json
[
  {
    "tool_id": "tool_001",
    "tool_name": "Tool Name",
    "description": "Description...",
    "category": "category_name"
  }
]
```

**Queries JSON** (`data/queries/test_queries.json`):
```json
[
  {
    "query": "Your query",
    "ground_truth_tool": "tool_001",
    "category": "category_name",
    "difficulty": "easy"
  }
]
```

## Key Metrics

- **Recall@k**: % of queries where correct tool is in top-k
- **MRR**: Mean reciprocal rank (higher = better)
- **Avg Time**: Average retrieval time per query (ms)

## Parameters

- `model_name`: 'all-MiniLM-L6-v2' (fast) or 'intfloat/e5-base-v2' (better quality)
- `k_values`: List of k values to evaluate, e.g., [1, 3, 5]
- `save_results`: Save results to JSON (default: True)

## Output

Results saved to `data/results/` with:
- Overall metrics (Recall@k, MRR, timing)
- Category breakdown
- Difficulty breakdown
- Individual query results with retrieval times
- Failure cases

