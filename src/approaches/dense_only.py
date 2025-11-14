from typing import Dict, Any
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.retrieval.dense_retriever import ToolRetriever


class DenseOnlyApproach:
    # Uses Pure semantic search (embedding + FAISS) to select top-1 tool
    # No LLM Involved

    def __init__(self, index_path: str, metadata_path:str, model_name:str= "ll-MiniLM-L6-v2"):
        # Initialize Dense Only Approach

        self.retriever = ToolRetriever(model_name=model_name)
        self.retriever.load_index(index_path, metadata_path)
        self.approach_name = "Dense Retrieval Only (Top-1)"
        self.model_name = model_name

    def select_tool(self, query:str)-> Dict[str, Any]:
        # Select tool using dense retrieval only

        import time
        start_time = time.perf_counter()

        # Get top 1
        top_tools = self.retriever.retrieve(query, k=1, return_scores=True)

        if not top_tools:
            raise ValueError("No tools retrieved")
        
        top_tool = top_tools[0]

        latency = time.perf_counter() - start_time

        result = {
            "query": query,
            "selected_tool_id": top_tool["tool_id"],
            "selected_tool_name": top_tool["tool_name"],
            "selected_server": top_tool.get("server", "Unknown"),
            "similarity_score": top_tool["similarity_score"],
            "latency_seconds": latency,
            "approach": self.approach_name,
            "model_name": self.model_name,
            "full_tool_info": top_tool
        }

        return result
    
    def evaluate_query(self, query:str, ground_truth_server: str)-> Dict[str, Any]:
        # Evaluate tool selection for single query

        result = self.select_tool(query)

        # Check if correct
        is_correct = result["selected_server"] == ground_truth_server

        evaluation = {
            **result,
            "ground_truth_server": ground_truth_server,
            "is_correct": is_correct,
            "accuracy": 1.0 if is_correct else 0.0
        }
        
        return evaluation
    

if __name__ == "__main__":
    print("Testing Approach 1: Dense Retrieval Only (top-1)")
    print("=" * 60)
    
    # First, build an index with sample tools
    print("\nStep 1: Building FAISS index with sample tools...")
    print("-" * 60)
    
    from src.indexing.tool_indexer import ToolIndexer
    
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
    indexer = ToolIndexer(model_name="all-MiniLM-L6-v2")
    indexer.build_index(sample_tools, batch_size=8)
    
    # Save index
    index_path = "data/indexes/dense_index.faiss"
    metadata_path = "data/indexes/dense_index.metadata.json"
    indexer.save_index(index_path, metadata_path)
    print(f"✓ Index saved to {index_path}")
    
    # Initialize approach
    print("\nStep 2: Initializing Approach 1...")
    print("-" * 60)
    approach = DenseOnlyApproach(
        index_path=index_path,
        metadata_path=metadata_path,
        model_name="all-MiniLM-L6-v2"
    )
    
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
            "query": "what is the temperature outside",
            "ground_truth": "Weather API",
            "description": "Semantic query - should work better than BM25"
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
        print(f"Similarity Score: {evaluation['similarity_score']:.4f}")
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
    print(f"Model: {approach.model_name}")
