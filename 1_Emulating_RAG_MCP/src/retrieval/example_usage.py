"""
Example: End-to-End Retrieval Pipeline
Demonstrates the complete workflow from indexing to retrieval and evaluation.
"""

import sys
from pathlib import Path
import json
import os

# Add src directory to path
current_dir = Path(__file__).parent
src_dir = current_dir.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))

from indexing import ToolIndexer
from retrieval import ToolRetriever, RetrievalMetrics


def main():
    """
    Complete example of the retrieval pipeline:
    1. Load tools
    2. Build and save index
    3. Load index and retrieve
    4. Evaluate performance
    """
    
    print("=" * 80)
    print("RAG-MCP Retrieval Pipeline - Complete Example")
    print("=" * 80)
    
    # ===== Step 1: Load Tools =====
    print("\n[Step 1] Loading tools...")
    print("-" * 80)
    
    tools_path = project_root / "data" / "tools" / "sample_tools.json"
    with open(tools_path, 'r', encoding='utf-8') as f:
        tools = json.load(f)
    
    print(f"✓ Loaded {len(tools)} tools")
    print(f"  Categories: {set(tool['category'] for tool in tools)}")
    
    # ===== Step 2: Build Index =====
    print("\n[Step 2] Building FAISS index...")
    print("-" * 80)
    
    # Initialize indexer with embedding model
    indexer = ToolIndexer(model_name='all-MiniLM-L6-v2')
    
    # Build index
    indexer.build_index(tools, batch_size=8)
    
    # Save index
    index_dir = project_root / "data" / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = index_dir / "tools_sample.index"
    metadata_path = index_dir / "tools_sample.metadata.json"
    
    indexer.save_index(str(index_path), str(metadata_path))
    print(f"✓ Index saved to: {index_path}")
    
    # ===== Step 3: Initialize Retriever =====
    print("\n[Step 3] Initializing retriever...")
    print("-" * 80)
    
    retriever = ToolRetriever(model_name='all-MiniLM-L6-v2')
    retriever.load_index(str(index_path), str(metadata_path))
    print(f"✓ Retriever ready with {len(retriever.tool_metadata)} tools")
    
    # ===== Step 4: Create Test Queries =====
    print("\n[Step 4] Creating test queries with ground truth...")
    print("-" * 80)
    
    test_queries = [
        {
            "query_id": "q001",
            "query": "Find recent research papers about climate change",
            "ground_truth_tool": "arxiv_search_002",
            "category": "academic",
            "difficulty": "easy"
        },
        {
            "query_id": "q002",
            "query": "Search the web for latest AI news",
            "ground_truth_tool": "brave_search_001",
            "category": "web_search",
            "difficulty": "easy"
        },
        {
            "query_id": "q003",
            "query": "Retrieve user data from the database",
            "ground_truth_tool": "postgresql_query_003",
            "category": "database",
            "difficulty": "easy"
        },
        {
            "query_id": "q004",
            "query": "Read the contents of my config file",
            "ground_truth_tool": "file_read_004",
            "category": "file_operations",
            "difficulty": "easy"
        },
        {
            "query_id": "q005",
            "query": "Send a message to the team channel",
            "ground_truth_tool": "slack_send_005",
            "category": "communication",
            "difficulty": "easy"
        },
        {
            "query_id": "q006",
            "query": "What's the temperature outside?",
            "ground_truth_tool": "weather_api_006",
            "category": "data_api",
            "difficulty": "easy"
        }
    ]
    
    print(f"✓ Created {len(test_queries)} test queries")
    
    # ===== Step 5: Run Retrieval =====
    print("\n[Step 5] Running retrieval for all queries (k=3)...")
    print("-" * 80)
    
    retrieval_results = []
    
    for query_data in test_queries:
        query = query_data['query']
        ground_truth = query_data['ground_truth_tool']
        
        # Retrieve top-3 tools
        retrieved_tools = retriever.retrieve(query, k=3, return_scores=True)
        retrieved_ids = [tool['tool_id'] for tool in retrieved_tools]
        
        # Store results
        result = {
            'query': query,
            'retrieved_ids': retrieved_ids,
            'ground_truth_id': ground_truth,
            'category': query_data['category'],
            'difficulty': query_data['difficulty']
        }
        retrieval_results.append(result)
        
        # Display results
        is_correct = ground_truth in retrieved_ids
        status = "✓" if is_correct else "✗"
        
        print(f"\n{status} Query: '{query}'")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Retrieved (Top-3):")
        for tool in retrieved_tools:
            marker = "→" if tool['tool_id'] == ground_truth else " "
            print(f"    {marker} [{tool['rank']}] {tool['tool_id']} "
                  f"(score: {tool['similarity_score']:.4f})")
    
    # ===== Step 6: Calculate Metrics =====
    print("\n[Step 6] Calculating retrieval metrics...")
    print("-" * 80)
    
    # Overall metrics
    metrics = RetrievalMetrics.calculate_all_metrics(
        retrieval_results,
        k_values=[1, 3, 5]
    )
    
    RetrievalMetrics.print_summary(metrics, "Overall Retrieval Performance")
    
    # By category
    print("\n" + "=" * 80)
    print("Metrics by Category")
    print("=" * 80)
    
    category_metrics = RetrievalMetrics.analyze_by_category(
        retrieval_results,
        k_values=[1, 3, 5]
    )
    
    for category, cat_metrics in category_metrics.items():
        print(f"\nCategory: {category}")
        print(f"  Recall@1: {cat_metrics['recall@1']:.4f}")
        print(f"  Recall@3: {cat_metrics['recall@3']:.4f}")
        print(f"  MRR: {cat_metrics['mrr']:.4f}")
    
    # ===== Step 7: Analyze Failures =====
    print("\n[Step 7] Analyzing failure cases...")
    print("-" * 80)
    
    failures = RetrievalMetrics.get_failure_cases(retrieval_results, k=3)
    
    if failures:
        print(f"\nFound {len(failures)} failure cases:")
        for i, failure in enumerate(failures, 1):
            print(f"\n  [{i}] Query: '{failure['query']}'")
            print(f"      Expected: {failure['ground_truth_id']}")
            print(f"      Got: {failure['retrieved_ids']}")
    else:
        print("\n✓ No failures! All ground truth tools were retrieved in top-3.")
    
    # ===== Step 8: Example LLM Integration =====
    print("\n[Step 8] Example: Formatting for LLM consumption...")
    print("-" * 80)
    
    example_query = test_queries[0]['query']
    retrieved = retriever.retrieve(example_query, k=3, return_scores=False)
    
    print(f"\nQuery: '{example_query}'")
    print("\nFormatted tools for LLM prompt:")
    print("-" * 80)
    
    formatted = retriever.format_tools_for_llm(
        retrieved,
        include_rank=True,
        include_score=False
    )
    print(formatted)
    
    # ===== Step 9: Save Results =====
    print("\n[Step 9] Saving results...")
    print("-" * 80)
    
    results_dir = project_root / "data" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = results_dir / "retrieval_example_results.json"
    
    output_data = {
        'queries': test_queries,
        'retrieval_results': retrieval_results,
        'metrics': metrics,
        'category_metrics': category_metrics
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to: {results_path}")
    
    # ===== Summary =====
    print("\n" + "=" * 80)
    print("Pipeline Complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Tools indexed: {len(tools)}")
    print(f"  - Queries processed: {len(test_queries)}")
    print(f"  - Overall Recall@3: {metrics['recall@3']:.2%}")
    print(f"  - Overall MRR: {metrics['mrr']:.4f}")
    print(f"  - Success rate: {(len(test_queries) - len(failures)) / len(test_queries):.2%}")
    print(f"\nNext steps:")
    print(f"  1. Add more tools and queries")
    print(f"  2. Experiment with different k values")
    print(f"  3. Try different embedding models (e.g., intfloat/e5-base-v2)")
    print(f"  4. Integrate with LLM for final tool selection")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
