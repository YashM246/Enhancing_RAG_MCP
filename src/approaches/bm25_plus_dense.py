from typing import Dict, Any
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.retrieval.bm25_plus_dense_retriever import HybridRetriever


class BM25PlusDenseApproach:
    """
    Approach 3: BM25 + Dense Retrieval using Reciprocal Rank Fusion (RRF).
    
    Combines BM25 lexical search and dense semantic search using RRF.
    No LLM involved - combines strengths of both retrieval methods.
    """

    def __init__(
        self,
        bm25_index_path: str,
        dense_index_path: str,
        dense_metadata_path: str,
        model_name: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60
    ):
        """
        Initialize BM25 + Dense Approach.
        
        Args:
            bm25_index_path: Path to BM25 index file
            dense_index_path: Path to FAISS dense index file
            dense_metadata_path: Path to dense index metadata JSON
            model_name: Name of the sentence-transformer model
            rrf_k: Constant k for RRF formula (default: 60)
        """
        self.retriever = HybridRetriever(
            bm25_index_path=bm25_index_path,
            dense_index_path=dense_index_path,
            dense_metadata_path=dense_metadata_path,
            model_name=model_name,
            rrf_k=rrf_k
        )
        self.approach_name = f"BM25 + Dense with RRF (k={rrf_k})"
        self.model_name = model_name
        self.rrf_k = rrf_k

    def select_tool(self, query: str, k: int = 7) -> Dict[str, Any]:
        """
        Select tool using BM25 + Dense RRF approach.

        Args:
            query: User query string
            k: Number of top tools to retrieve for metrics

        Returns:
            Dictionary with selected tool and metadata
        """
        start_time = time.perf_counter()

        # Get top-k tools using RRF (for metrics calculation)
        top_k_tools = self.retriever.retrieve(query, top_k=k)

        if not top_k_tools:
            raise ValueError("No tools retrieved")

        # Top-1 is the selected tool
        top_tool = top_k_tools[0]

        latency = time.perf_counter() - start_time

        # Extract servers from top-k for recall@k metrics
        retrieved_servers = [tool.get("server", "Unknown") for tool in top_k_tools]

        result = {
            "query": query,
            "selected_tool_id": top_tool["tool_id"],
            "selected_tool_name": top_tool["tool_name"],
            "selected_server": top_tool.get("server", "Unknown"),
            "retrieved_servers": retrieved_servers,  # For recall@k metrics
            "rrf_score": top_tool["rrf_score"],
            "bm25_rank": top_tool.get("bm25_rank"),
            "bm25_score": top_tool.get("bm25_score", 0.0),
            "dense_rank": top_tool.get("dense_rank"),
            "dense_score": top_tool.get("dense_score", 0.0),
            "latency_seconds": latency,
            "approach": self.approach_name,
            "model_name": self.model_name,
            "rrf_k": self.rrf_k,
            "full_tool_info": top_tool
        }

        return result

    def evaluate_query(self, query: str, ground_truth_server: str) -> Dict[str, Any]:
        """
        Evaluate tool selection for single query.

        Args:
            query: User query string
            ground_truth_server: Correct server name

        Returns:
            Evaluation dictionary with correctness info
        """
        result = self.select_tool(query)

        # Check if selection is correct (server-level comparison)
        is_correct = result["selected_server"] == ground_truth_server

        evaluation = {
            **result,
            "ground_truth_server": ground_truth_server,
            "is_correct": is_correct,
            "accuracy": 1.0 if is_correct else 0.0
        }

        return evaluation

    def evaluate_with_comparison(
        self,
        query: str,
        ground_truth_tool_id: str,
        show_separate: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate and show comparison with separate BM25 and Dense results.
        
        Args:
            query: User query string
            ground_truth_tool_id: Correct tool ID
            show_separate: Whether to retrieve separate BM25/Dense results
            
        Returns:
            Evaluation dictionary with comparison data
        """
        evaluation = self.evaluate_query(query, ground_truth_tool_id)

        if show_separate:
            # Get separate retrieval results for comparison
            separate_results = self.retriever.retrieve_separate(query, top_k=3)
            evaluation["bm25_top3"] = separate_results["bm25"]
            evaluation["dense_top3"] = separate_results["dense"]
            evaluation["hybrid_top3"] = separate_results["hybrid"]

        return evaluation


if __name__ == "__main__":
    print("Testing Approach 3: BM25 + Dense with RRF")
    print("=" * 60)
    
    # Initialize approach
    print("\nInitializing BM25 + Dense Approach...")
    print("-" * 60)
    
    approach = BM25PlusDenseApproach(
        bm25_index_path="data/indexes/bm25_index.pkl",
        dense_index_path="data/indexes/tools_all-MiniLM-L6-v2.index",
        dense_metadata_path="data/indexes/tools_all-MiniLM-L6-v2.metadata.json",
        model_name="all-MiniLM-L6-v2",
        rrf_k=60
    )
    
    # Test queries with ground truth
    print("\nTesting queries...")
    print("-" * 60)
    
    test_cases = [
        {
            "query": "get current weather data",
            "ground_truth": "tool_001",
            "description": "Direct keyword match - both should work"
        },
        {
            "query": "run SQL query on database",
            "ground_truth": "tool_002",
            "description": "SQL query - both should work"
        },
        {
            "query": "read file from disk",
            "ground_truth": "tool_003",
            "description": "File reading - both should work"
        },
        {
            "query": "what is the temperature outside",
            "ground_truth": "tool_001",
            "description": "Semantic query - Dense should help"
        },
        {
            "query": "execute database command",
            "ground_truth": "tool_002",
            "description": "Semantic variation - tests fusion benefit"
        }
    ]
    
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: '{test_case['query']}'")
        print(f"Expected: {test_case['ground_truth']} ({test_case['description']})")
        print(f"{'='*60}")
        
        evaluation = approach.evaluate_query(
            test_case["query"],
            test_case["ground_truth"]
        )
        
        results.append(evaluation)
        
        # Print results
        status = "✓" if evaluation['is_correct'] else "✗"
        print(f"\n{status} Selected: {evaluation['selected_tool_id']} - {evaluation['selected_tool_name']}")
        print(f"  RRF Score: {evaluation['rrf_score']:.4f}")
        print(f"  BM25: Rank={evaluation['bm25_rank']}, Score={evaluation['bm25_score']:.4f}")
        print(f"  Dense: Rank={evaluation['dense_rank']}, Score={evaluation['dense_score']:.4f}")
        print(f"  Latency: {evaluation['latency_seconds']*1000:.2f}ms")
    
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
    print(f"Model: {approach.model_name}")
    print(f"RRF k: {approach.rrf_k}")
    
    # Detailed comparison for one query
    if results:
        print(f"\n{'='*60}")
        print("DETAILED COMPARISON (First Query)")
        print(f"{'='*60}")
        
        detailed = approach.evaluate_with_comparison(
            test_cases[0]["query"],
            test_cases[0]["ground_truth"],
            show_separate=True
        )
        
        print(f"\nQuery: '{detailed['query']}'")
        print(f"\nBM25 Top-3:")
        for i, tool in enumerate(detailed["bm25_top3"][:3], 1):
            marker = "→" if tool["tool_id"] == test_cases[0]["ground_truth"] else " "
            print(f"  {marker}[{i}] {tool['tool_name']} (Score: {tool.get('bm25_score', 0):.4f})")
        
        print(f"\nDense Top-3:")
        for i, tool in enumerate(detailed["dense_top3"][:3], 1):
            marker = "→" if tool["tool_id"] == test_cases[0]["ground_truth"] else " "
            print(f"  {marker}[{i}] {tool['tool_name']} (Score: {tool.get('similarity_score', 0):.4f})")
        
        print(f"\nBM25 + Dense RRF Top-3:")
        for i, tool in enumerate(detailed["hybrid_top3"][:3], 1):
            marker = "→" if tool["tool_id"] == test_cases[0]["ground_truth"] else " "
            print(f"  {marker}[{i}] {tool['tool_name']} (RRF: {tool.get('rrf_score', 0):.4f})")
            print(f"      BM25 Rank: {tool.get('bm25_rank', 'N/A')}, Dense Rank: {tool.get('dense_rank', 'N/A')}")
    
    print(f"\n{'='*60}")
    print("✓ BM25 + Dense approach testing complete!")
    print(f"{'='*60}")
