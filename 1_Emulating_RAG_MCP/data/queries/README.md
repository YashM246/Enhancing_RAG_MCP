# Queries Ingestion 

It describes the purpose, file structure, and execution steps for `ingest_mcp_bench_queries.py`.

---
## Purpose
1. Reads all query definitions from an **external MCP-Bench dataset** (e.g., `mcp-bench/tasks/*.json`).
2. Extracts relevant benchmark entries (query, category, servers, ground truth, etc.).
3. Merges them into a single consolidated file →  
   **`data/queries/queries.json`**
4. Validates schema consistency (field names, nested arrays, valid category tags).
5. Outputs a clean, ready-to-use JSON file for downstream RAG benchmarking.




---
##  Output Format
The final file (`queries.json`) is a single JSON object structured as:

```json
{
  "queries": [
    {
      "query_id": "google_maps_weather_data_national_parks_000",
      "query": "You are planning a 3-day camping expedition...",
      "category": "three_server_combinations",
      "ground_truth_tool": "Google Maps+Weather Data+National Parks",
      "servers": ["Google Maps", "Weather Data", "National Parks"],
      "distraction_servers": [
        "Bibliomantic", "BioMCP", "DEX Paprika", "Math MCP"
      ],
      "combination_name": "Travel Planning Suite"
    }
  ]
}