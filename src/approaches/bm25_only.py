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

    
    def select_tool(self, query: str)-> Dict[str, Any]:

        start_time = time.time()

        # Get top-1 tool using BM25
        top_tool = self.retriever.retrieve_top1(query)

        latency = time.time() - start_time

        result = {
            "query": query,
            "selected_tool_id": top_tool["tool_id"],
            "selected_tool_name": top_tool["tool_name"],
            "bm25_score": top_tool["bm25_score"],
            "latency_seconds": latency,
            "approach": self.approach_name,
            "full_tool_info": top_tool
        }

        return result
    

    def evaluate_query(self, query:str, ground_truth_tool_id: str)-> Dict[str, Any]:

        result = self.select_tool(query)

        # Check if selection is correct
        is_correct = result["selected_tool_id"] == ground_truth_tool_id

        evaluation = {
            **result,
            "ground_truth_tool_id": ground_truth_tool_id,
            "is_correct": is_correct,
            "accuracy": 1.0 if is_correct else 0.0
        }

        return evaluation
    
if __name__ == "__main__":
    print("Testing Approach 2: BM25 Only (top-1)")
    print("=" * 60)
    
    # Initialize approach
    approach = BM25OnlyApproach(index_path="data/indexes/bm25_index.pkl")
    
    # Test queries with ground truth
    test_cases = [
        {
            "query": "get current weather data",
            "ground_truth": "tool_001",
            "description": "Should select Weather API"
        },
        {
            "query": "run SQL query on database",
            "ground_truth": "tool_002",
            "description": "Should select Database Query"
        },
        {
            "query": "read file from disk",
            "ground_truth": "tool_003",
            "description": "Should select File Reader"
        },
        {
            "query": "what is the temperature",
            "ground_truth": "tool_001",
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
        
        print(f"Selected: {evaluation['selected_tool_id']} - {evaluation['selected_tool_name']}")
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
