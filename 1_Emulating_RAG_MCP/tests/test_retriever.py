"""
Test script for the Retrieval Module
Demonstrates basic usage of ToolRetriever and RetrievalMetrics.
"""

import sys
from pathlib import Path
import os
import json

# Add src directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from indexing import ToolIndexer
from retrieval import ToolRetriever, RetrievalMetrics


def test_retriever():
    """Test the ToolRetriever with sample tools."""
    print("=" * 70)
    print("Testing Tool Retrieval Module")
    print("=" * 70)

    # Load sample tools
    tools_path = project_root / "data" / "tools" / "sample_tools.json"
    with open(tools_path, 'r', encoding='utf-8') as f:
        tools = json.load(f)

    print(f"\nLoaded {len(tools)} sample tools")

    # Build index using ToolIndexer
    print("\n" + "-" * 70)
    print("Step 1: Building FAISS index")
    print("-" * 70)

    indexer = ToolIndexer(model_name='all-MiniLM-L6-v2')
    indexer.build_index(tools, batch_size=8)

    # Initialize retriever with the same model
    print("\n" + "-" * 70)
    print("Step 2: Initializing retriever")
    print("-" * 70)

    retriever = ToolRetriever(model_name='all-MiniLM-L6-v2')
    retriever.set_index(indexer.index, indexer.tool_metadata)

    # Test queries
    test_queries = [
        {
            "query": "Find recent papers about climate change",
            "ground_truth": "arxiv_search_002",
            "description": "Academic search query"
        },
        {
            "query": "Search for news about AI",
            "ground_truth": "brave_search_001",
            "description": "Web search query"
        },
        {
            "query": "Get all users from database",
            "ground_truth": "postgresql_query_003",
            "description": "Database query"
        },
        {
            "query": "What's the weather today?",
            "ground_truth": "weather_api_006",
            "description": "Weather information query"
        },
        {
            "query": "Read configuration file",
            "ground_truth": "file_read_004",
            "description": "File operations query"
        }
    ]

    # Test retrieval for each query
    print("\n" + "-" * 70)
    print("Step 3: Testing retrieval for sample queries")
    print("-" * 70)

    results_for_metrics = []

    for i, test_case in enumerate(test_queries, 1):
        query = test_case["query"]
        ground_truth_id = test_case["ground_truth"]

        print(f"\n[Query {i}] {test_case['description']}")
        print(f"Query: '{query}'")
        print(f"Ground Truth: {ground_truth_id}")

        # Retrieve top-3 tools
        retrieved_tools = retriever.retrieve(query, k=3, return_scores=True)

        print(f"\nTop-3 Retrieved Tools:")
        for tool in retrieved_tools:
            is_correct = "✓ CORRECT" if tool['tool_id'] == ground_truth_id else ""
            print(f"  [{tool['rank']}] {tool['tool_name']} ({tool['tool_id']}) "
                  f"- Score: {tool['similarity_score']:.4f} {is_correct}")

        # Store results for metrics calculation
        retrieved_ids = [tool['tool_id'] for tool in retrieved_tools]
        results_for_metrics.append({
            'query': query,
            'retrieved_ids': retrieved_ids,
            'ground_truth_id': ground_truth_id,
            'category': 'test',
            'difficulty': 'easy'
        })

        # Analyze retrieval quality
        analysis = retriever.analyze_retrieval_quality(
            query,
            ground_truth_id,
            k_values=[1, 3, 5]
        )
        print(f"\nRetrieval Analysis:")
        print(f"  Recall@1: {analysis['recall@1']}")
        print(f"  Recall@3: {analysis['recall@3']}")
        print(f"  Reciprocal Rank: {analysis['reciprocal_rank']:.4f}")
        if analysis['rank@3']:
            print(f"  Found at Rank: {analysis['rank@3']}")

    # Calculate overall metrics
    print("\n" + "=" * 70)
    print("Step 4: Calculating overall metrics")
    print("=" * 70)

    metrics = RetrievalMetrics.calculate_all_metrics(
        results_for_metrics,
        k_values=[1, 3, 5]
    )

    RetrievalMetrics.print_summary(metrics, "Overall Retrieval Performance")

    # Test LLM formatting
    print("\n" + "-" * 70)
    print("Step 5: Testing LLM-friendly formatting")
    print("-" * 70)

    sample_query = test_queries[0]["query"]
    retrieved_tools = retriever.retrieve(sample_query, k=3, return_scores=True)

    print(f"\nQuery: '{sample_query}'")
    print("\nFormatted for LLM (with scores):")
    print("-" * 70)
    formatted = retriever.format_tools_for_llm(
        retrieved_tools,
        include_rank=True,
        include_score=True
    )
    print(formatted)

    # Test threshold-based retrieval
    print("\n" + "-" * 70)
    print("Step 6: Testing threshold-based retrieval")
    print("-" * 70)

    threshold_results = retriever.retrieve_with_threshold(
        sample_query,
        similarity_threshold=0.3,
        max_k=5,
        return_scores=True
    )

    print(f"\nQuery: '{sample_query}'")
    print(f"Threshold: 0.3, Max K: 5")
    print(f"Found {len(threshold_results)} tools above threshold:")
    for tool in threshold_results:
        print(f"  - {tool['tool_name']}: {tool['similarity_score']:.4f}")

    # Test failure case analysis
    print("\n" + "-" * 70)
    print("Step 7: Analyzing failure cases")
    print("-" * 70)

    failures = RetrievalMetrics.get_failure_cases(results_for_metrics, k=3)
    if failures:
        print(f"\nFound {len(failures)} failure cases (ground truth not in top-3):")
        for failure in failures:
            print(f"\n  Query: '{failure['query']}'")
            print(f"  Expected: {failure['ground_truth_id']}")
            print(f"  Retrieved: {failure['retrieved_ids']}")
    else:
        print("\n✓ No failures! All ground truth tools were retrieved in top-3.")

    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)


def test_batch_retrieval():
    """Test batch retrieval functionality."""
    print("\n\n" + "=" * 70)
    print("Testing Batch Retrieval")
    print("=" * 70)

    # Load sample tools
    tools_path = project_root / "data" / "tools" / "sample_tools.json"
    with open(tools_path, 'r', encoding='utf-8') as f:
        tools = json.load(f)

    # Build index
    indexer = ToolIndexer(model_name='all-MiniLM-L6-v2')
    indexer.build_index(tools, batch_size=8)

    # Initialize retriever
    retriever = ToolRetriever(model_name='all-MiniLM-L6-v2')
    retriever.set_index(indexer.index, indexer.tool_metadata)

    # Batch queries
    queries = [
        "Search for AI news",
        "Query database",
        "Check weather"
    ]

    print(f"\nProcessing {len(queries)} queries in batch:")
    batch_results = retriever.batch_retrieve(queries, k=2, return_scores=True)

    for query, results in zip(queries, batch_results):
        print(f"\nQuery: '{query}'")
        print(f"Top-2 Tools:")
        for tool in results:
            print(f"  [{tool['rank']}] {tool['tool_name']} - Score: {tool['similarity_score']:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        test_retriever()
        test_batch_retrieval()
        print("\n✓ All tests passed successfully!")
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
