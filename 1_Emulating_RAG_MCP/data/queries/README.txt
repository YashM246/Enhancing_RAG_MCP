✅ TEST QUERIES - SAMPLE DATA AVAILABLE

Files:
✅ test_queries.json - Sample test queries with ground truth

Expected format: test_queries.json (JSON array)
Schema:
{
  "query_id": "q001",
  "query": "Find recent papers about climate change",
  "ground_truth_tool": "academic_search_api",
  "category": "academic",
  "difficulty": "easy"
}

Usage:
Load in benchmarking module with:
benchmarker.benchmark_retrieval_only(
    queries_json_path="data/queries/test_queries.json",
    ...
)

