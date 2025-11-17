from typing import Dict, Any
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.retrieval.bm25_retriever import BM25Retriever

class BM25OnlyApproach:

    """
    Approach 2: BM25 Only tool selection.
    
    Uses pure BM25 lexical search to select the top-1 tool.
    No LLM involved - fastest approach but limited by keyword matching.
    
    """

    def __init__(self, index_path:str):

        self.retriever = BM25Retriever(index_path)
        self.approach_name = "BM25 Only (Top-1)"

    
    def select_tool(self, query: str, k: int = 7)-> Dict[str, Any]:

        start_time = time.time()

        # Get top-k tools using BM25 (for metrics calculation)
        top_k_tools = self.retriever.retrieve(query, top_k=k)

        # Top-1 is the selected tool
        top_tool = top_k_tools[0] if top_k_tools else None

        if not top_tool:
            raise ValueError("No tools retrieved")

        latency = time.time() - start_time

        # Extract servers from top-k for recall@k metrics
        retrieved_servers = [tool.get("server", "Unknown") for tool in top_k_tools]

        result = {
            "query": query,
            "selected_tool_id": top_tool["tool_id"],
            "selected_tool_name": top_tool["tool_name"],
            "selected_server": top_tool.get("server", "Unknown"),
            "retrieved_servers": retrieved_servers,  # For recall@k metrics
            "bm25_score": top_tool["bm25_score"],
            "latency_seconds": latency,
            "approach": self.approach_name,
            "full_tool_info": top_tool
        }

        return result
    

    def evaluate_query(self, query:str, ground_truth_server: str)-> Dict[str, Any]:

        result = self.select_tool(query)

        # Check if selection is correct (compare server names)
        is_correct = result["selected_server"] == ground_truth_server

        evaluation = {
            **result,
            "ground_truth_server": ground_truth_server,
            "is_correct": is_correct,
            "accuracy": 1.0 if is_correct else 0.0
        }

        return evaluation
    
if __name__ == "__main__":
    print("Testing Approach 2: BM25 Only (top-1)")
    print("=" * 60)

    # First, build a BM25 index with sample tools
    print("\nStep 1: Building BM25 index with sample tools...")
    print("-" * 60)

    from src.indexing.bm25_indexer import BM25Indexer

    sample_tools = [
        {
            "tool_id": "tool_001",
            "tool_name": "Weather API",
            "description": "Get current weather data for any location",
            "usage_example": "Get weather for New York",
            "server": "Weather API"
        },
        {
            "tool_id": "tool_002",
            "tool_name": "Database Query",
            "description": "Execute SQL queries on the database",
            "usage_example": "Query user table",
            "server": "Database Tools"
        },
        {
            "tool_id": "tool_003",
            "tool_name": "File Reader",
            "description": "Read contents from files",
            "usage_example": "Read config.json",
            "server": "File Operations"
        }
    ]

    # Build index
    indexer = BM25Indexer()
    indexer.build_index(sample_tools)

    # Save index
    index_path = "data/indexes/bm25_index.pkl"
    indexer.save_index(index_path)
    print(f"✓ Index saved to {index_path}")

    # Initialize approach
    print("\nStep 2: Initializing Approach 2...")
    print("-" * 60)
    approach = BM25OnlyApproach(index_path=index_path)

    # Test queries with ground truth
    print("\nStep 3: Testing queries...")
    print("-" * 60)

    test_cases = [
        {
            "query": "get current weather data",
            "ground_truth": "Weather API",
            "description": "Should select Weather API"
        },
        {
            "query": "run SQL query on database",
            "ground_truth": "Database Tools",
            "description": "Should select Database Query"
        },
        {
            "query": "read file from disk",
            "ground_truth": "File Operations",
            "description": "Should select File Reader"
        },
        {
            "query": "what is the temperature",
            "ground_truth": "Weather API",
            "description": "Semantic query - might fail (no 'weather' keyword)"
        }
    ]
    
    results = []
    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Query: '{test_case['query']}'")
        print(f"Expected: {test_case['ground_truth']} ({test_case['description']})")
        print(f"{'='*60}")
        
        evaluation = approach.evaluate_query(
            test_case["query"],
            test_case["ground_truth"]
        )
        
        results.append(evaluation)

        print(f"Selected: {evaluation['selected_server']} (Tool: {evaluation['selected_tool_name']})")
        print(f"BM25 Score: {evaluation['bm25_score']:.4f}")
        print(f"Correct: {'✓' if evaluation['is_correct'] else '✗'}")
        print(f"Latency: {evaluation['latency_seconds']*1000:.2f}ms")
    
    # Summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_correct = sum(r['is_correct'] for r in results)
    accuracy = total_correct / len(results) * 100
    avg_latency = sum(r['latency_seconds'] for r in results) / len(results) * 1000
    
    print(f"Total queries: {len(results)}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Average latency: {avg_latency:.2f}ms")
    print(f"Token usage: 0 (no LLM)")
