"""
Approach 6: Hybrid (BM25 + Dense) Retrieval + LLM 

This approach uses hybrid retrieval (BM25 + Dense with RRF) for retrieval.
Two-stage pipeline:
1. Hybrid retrieval (RRF fusion) narrows down to top-k candidates
2. LLM selects final tool(s) from candidates with reasoning

This combines the best of all worlds:
- BM25 lexical matching
- Dense semantic search  
- RRF fusion for robust ranking
- LLM reasoning for final selection
"""

from typing import Dict, Any
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.retrieval.bm25_plus_dense_retriever import HybridRetriever
from src.llm.llm_selector import LLMToolSelector


class LLMHybridApproach:
    """
    LLM + Hybrid Retrieval approach.
    Uses RRF fusion of BM25 and Dense retrieval, then LLM selection.
    """

    def __init__(
        self,
        bm25_index_path: str,
        dense_index_path: str,
        dense_metadata_path: str,
        server_url: str = "http://localhost:11434",
        model_name: str = "mistral:7b-instruct-q4_0",
        backend: str = "ollama",
        embedding_model: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
        retrieval_k: int = 5
    ):
        """
        Initialize Hybrid Retrieval + LLM approach.

        Args:
            bm25_index_path: Path to BM25 index file
            dense_index_path: Path to FAISS dense index file
            dense_metadata_path: Path to dense index metadata JSON
            server_url: LLM server URL (Ollama: http://localhost:11434, vLLM: http://localhost:8000)
            model_name: LLM model name (Ollama: mistral:7b-instruct-q4_0, vLLM: mistralai/Mistral-7B-Instruct-v0.3)
            backend: Backend type - 'ollama' or 'vllm'
            embedding_model: Name of the sentence-transformer model for dense retrieval
            rrf_k: Constant k for RRF formula (default: 60)
            retrieval_k: Number of candidate tools to retrieve before LLM selection
        """
        # Initialize Hybrid retriever (Stage 1)
        self.retriever = HybridRetriever(
            bm25_index_path=bm25_index_path,
            dense_index_path=dense_index_path,
            dense_metadata_path=dense_metadata_path,
            model_name=embedding_model,
            rrf_k=rrf_k
        )

        # Initialize LLM selector (Stage 2)
        self.llm_selector = LLMToolSelector(
            server_url=server_url,
            model_name=model_name,
            backend=backend,
            temperature=0.1,
            max_tokens=500
        )

        # Store configuration
        self.retrieval_k = retrieval_k
        self.rrf_k = rrf_k
        self.embedding_model = embedding_model
        self.approach_name = f"Hybrid (BM25+Dense RRF) + LLM (k={retrieval_k})"

    def select_tool(self, query: str) -> Dict[str, Any]:
        """
        Select tool(s) using two-stage pipeline:
        1. Retrieve top-k candidates with Hybrid RRF
        2. LLM selects best tool(s) from candidates

        Args:
            query: User query string

        Returns:
            Dictionary containing:
                - query: Input query
                - selected_tools: List of selected tool names (from LLM)
                - num_tools_selected: Number of tools selected
                - candidate_tools: List of candidate tool names (from retrieval)
                - num_candidates: Number of candidates retrieved
                - prompt_tokens: Tokens used in LLM prompt
                - completion_tokens: Tokens in LLM response
                - total_tokens: Total tokens used
                - retrieval_latency_seconds: Time for retrieval stage
                - llm_latency_seconds: Time for LLM stage
                - latency_seconds: Total time
                - approach: Approach name
                - retrieval_k: Number of candidates retrieved
                - rrf_k: RRF constant used
        """
        # Start timing
        start_time = time.perf_counter()

        # Stage 1: Hybrid Retrieval (BM25 + Dense with RRF)
        retrieval_start = time.perf_counter()
        candidate_tools = self.retriever.retrieve(
            query=query,
            top_k=self.retrieval_k
        )
        retrieval_latency = time.perf_counter() - retrieval_start

        # Stage 2: LLM Selection
        llm_start = time.perf_counter()
        llm_result = self.llm_selector.select_tool(
            query=query,
            candidate_tools=candidate_tools
        )
        llm_latency = time.perf_counter() - llm_start

        # Calculate total latency
        total_latency = time.perf_counter() - start_time

        # Build result dictionary
        result = {
            "query": query,
            "selected_tools": llm_result["selected_tools"],
            "num_tools_selected": llm_result["num_tools_selected"],
            "candidate_tools": [t["tool_name"] for t in candidate_tools],
            "num_candidates": len(candidate_tools),
            "prompt_tokens": llm_result["usage"]["prompt_tokens"],
            "completion_tokens": llm_result["usage"]["completion_tokens"],
            "total_tokens": llm_result["usage"]["total_tokens"],
            "retrieval_latency_seconds": retrieval_latency,
            "llm_latency_seconds": llm_latency,
            "latency_seconds": total_latency,
            "approach": self.approach_name,
            "retrieval_k": self.retrieval_k,
            "rrf_k": self.rrf_k,
            "embedding_model": self.embedding_model,
            "raw_response": llm_result.get("raw_response", "")
        }

        return result

    def evaluate_query(self, query: str, ground_truth_server: str) -> Dict[str, Any]:
        """
        Evaluate tool selection for a single query.

        Args:
            query: User query string
            ground_truth_server: Expected server name (e.g., "Weather API")

        Returns:
            Dictionary with selection results + accuracy evaluation
        """
        # Get selection result
        result = self.select_tool(query)

        # Extract server names from selected tools with strict validation
        selected_servers = []
        mismatched_tools = []
        
        # Get valid tool names from the hybrid retriever's metadata
        valid_tool_names = [t["tool_name"] for t in self.retriever.dense_retriever.tool_metadata]

        for tool_name in result["selected_tools"]:
            # Find the tool in metadata by EXACT matching (case-sensitive)
            tool = next((t for t in self.retriever.dense_retriever.tool_metadata if t["tool_name"] == tool_name), None)
            if tool:
                selected_servers.append(tool.get("server", "Unknown"))
            else:
                # LLM returned a tool name that doesn't match exactly - STRICT VALIDATION FAILURE
                selected_servers.append("Unknown")
                mismatched_tools.append({
                    "llm_returned": tool_name,
                    "valid_options": valid_tool_names
                })
                print(f"⚠️  WARNING: LLM returned invalid tool name: '{tool_name}'")
                print(f"   Valid options: {valid_tool_names}")
                print(f"   The LLM must return EXACT tool names (case-sensitive)!")

        # Check if selection is correct (server-level comparison)
        is_correct = ground_truth_server in selected_servers

        # Build evaluation dictionary
        evaluation = {
            **result,
            "selected_servers": selected_servers,
            "ground_truth_server": ground_truth_server,
            "is_correct": is_correct,
            "accuracy": 1.0 if is_correct else 0.0,
            "mismatched_tools": mismatched_tools  # Track LLM tool name errors
        }

        return evaluation


if __name__ == "__main__":
    print("Testing Approach 6: Hybrid (BM25+Dense) Retrieval + LLM")
    print("=" * 60)

    # First, build indexes with sample tools
    print("\nStep 1: Building indexes with sample tools...")
    print("-" * 60)

    from src.indexing.bm25_indexer import BM25Indexer
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

    # Build BM25 index
    bm25_indexer = BM25Indexer()
    bm25_indexer.build_index(sample_tools)
    bm25_index_path = "data/indexes/bm25_index.pkl"
    bm25_indexer.save_index(bm25_index_path)
    print(f"✓ BM25 index saved to {bm25_index_path}")

    # Build Dense index
    dense_indexer = ToolIndexer(model_name="all-MiniLM-L6-v2")
    dense_indexer.build_index(sample_tools, batch_size=8)
    dense_index_path = "data/indexes/dense_index.faiss"
    dense_metadata_path = "data/indexes/dense_index.metadata.json"
    dense_indexer.save_index(dense_index_path, dense_metadata_path)
    print(f"✓ Dense index saved to {dense_index_path}")

    # Initialize approach
    print("\nStep 2: Initializing Approach 6...")
    print("-" * 60)

    try:
        approach = LLMHybridApproach(
            bm25_index_path=bm25_index_path,
            dense_index_path=dense_index_path,
            dense_metadata_path=dense_metadata_path,
            server_url="http://localhost:11434",
            model_name="mistral:7b-instruct-q4_0",
            backend="ollama",
            embedding_model="all-MiniLM-L6-v2",
            rrf_k=60,
            retrieval_k=3
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
                "description": "Should select Database Tools"
            },
            {
                "query": "read file from disk",
                "ground_truth": "File Operations",
                "description": "Should select File Operations"
            },
            {
                "query": "what is the temperature outside",
                "ground_truth": "Weather API",
                "description": "Semantic query - Hybrid retrieval should handle well"
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

            # Display results
            selected_server = evaluation['selected_servers'][0] if evaluation['selected_servers'] else 'None'
            selected_tool = evaluation['selected_tools'][0] if evaluation['selected_tools'] else 'None'
            print(f"Selected: {selected_server} (Tool: {selected_tool})")
            print(f"Candidates (k={evaluation['retrieval_k']}): {evaluation['candidate_tools']}")
            print(f"Tokens: {evaluation['prompt_tokens']} prompt + {evaluation['completion_tokens']} completion = {evaluation['total_tokens']} total")
            print(f"Correct: {'✓' if evaluation['is_correct'] else '✗'}")
            print(f"Retrieval: {evaluation['retrieval_latency_seconds']*1000:.2f}ms | LLM: {evaluation['llm_latency_seconds']*1000:.2f}ms | Total: {evaluation['latency_seconds']*1000:.2f}ms")

        # Summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        total_correct = sum(r['is_correct'] for r in results)
        accuracy = total_correct / len(results) * 100
        avg_latency = sum(r['latency_seconds'] for r in results) / len(results) * 1000
        avg_retrieval_latency = sum(r['retrieval_latency_seconds'] for r in results) / len(results) * 1000
        avg_llm_latency = sum(r['llm_latency_seconds'] for r in results) / len(results) * 1000
        avg_prompt_tokens = sum(r['prompt_tokens'] for r in results) / len(results)
        avg_total_tokens = sum(r['total_tokens'] for r in results) / len(results)

        print(f"Total queries: {len(results)}")
        print(f"Correct: {total_correct}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Average total latency: {avg_latency:.2f}ms")
        print(f"  - Retrieval: {avg_retrieval_latency:.2f}ms")
        print(f"  - LLM: {avg_llm_latency:.2f}ms")
        print(f"Average prompt tokens: {avg_prompt_tokens:.0f}")
        print(f"Average total tokens: {avg_total_tokens:.0f}")
        print(f"Retrieval k: {approach.retrieval_k}")
        print(f"RRF k: {approach.rrf_k}")
        print(f"Embedding model: {approach.embedding_model}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure:")
        print("1. Ollama is installed and running")
        print("2. Run: ollama run mistral:7b-instruct-q4_0")
        print("3. Ollama is accessible at http://localhost:11434")
        import traceback
        traceback.print_exc()
