"""
Approach 4: Dense Retrieval + LLM (RAG-MCP)

This is the RAG-MCP approach from Gan & Sun (2025).
Two-stage pipeline:
1. Dense retrieval narrows down to top-k candidates (semantic search via FAISS)
2. LLM selects final tool(s) from candidates with reasoning

Expected performance:
- Accuracy: ~43% (from paper)
- Token reduction: >50% vs LLM-only
- Latency: Fast (retrieval) + Medium (LLM on small set)
"""

from typing import Dict, Any
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.retrieval.dense_retriever import ToolRetriever
from src.llm.llm_selector import LLMToolSelector


class DenseLLMApproach:
    """
    Approach 4: Dense Retrieval + LLM (RAG-MCP from paper)

    Two-stage tool selection:
    1. Dense retriever gets top-k candidates using semantic search (FAISS + embeddings)
    2. LLM selects the best tool(s) from those candidates

    This approach reduces prompt bloat by only sending k tools to the LLM
    instead of all tools, while maintaining high accuracy through LLM reasoning.
    """

    def __init__(self,
                 index_path: str,
                 metadata_path: str,
                 server_url: str = "http://localhost:11434",
                 model_name: str = "mistral:7b-instruct-q4_0",
                 backend: str = "ollama",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 k: int = 5):
        """
        Initialize Dense Retrieval + LLM approach.

        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
            server_url: LLM server URL (Ollama: http://localhost:11434, vLLM: http://localhost:8000)
            model_name: LLM model name (Ollama: mistral:7b-instruct-q4_0, vLLM: mistralai/Mistral-7B-Instruct-v0.3)
            backend: Backend type - 'ollama' or 'vllm'
            embedding_model: Sentence transformer model for dense retrieval
            k: Number of candidate tools to retrieve before LLM selection
        """
        # Initialize dense retriever (Stage 1)
        self.retriever = ToolRetriever(model_name=embedding_model)
        self.retriever.load_index(index_path, metadata_path)

        # Initialize LLM selector (Stage 2)
        self.llm_selector = LLMToolSelector(
            server_url=server_url,
            model_name=model_name,
            backend=backend,
            temperature=0.1,
            max_tokens=500
        )

        # Store configuration
        self.k = k
        self.approach_name = f"Dense Retrieval + LLM (k={k})"
        self.embedding_model = embedding_model

    def select_tool(self, query: str) -> Dict[str, Any]:
        """
        Select tool(s) using two-stage pipeline:
        1. Retrieve top-k candidates with dense retrieval
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
                - k: Number of candidates retrieved
        """
        # Start timing
        start_time = time.perf_counter()

        # Stage 1: Dense Retrieval
        retrieval_start = time.perf_counter()
        candidate_tools = self.retriever.retrieve(
            query=query,
            k=self.k,
            return_scores=True
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
            "k": self.k,
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
        valid_tool_names = [t["tool_name"] for t in self.retriever.tool_metadata]

        for tool_name in result["selected_tools"]:
            # Find the tool in metadata by EXACT matching (case-sensitive)
            tool = next((t for t in self.retriever.tool_metadata if t["tool_name"] == tool_name), None)
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
    print("Testing Approach 4: Dense Retrieval + LLM (RAG-MCP)")
    print("=" * 60)

    # First, build FAISS index with sample tools
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
    print("\nStep 2: Initializing Approach 4...")
    print("-" * 60)

    try:
        approach = DenseLLMApproach(
            index_path=index_path,
            metadata_path=metadata_path,
            server_url="http://localhost:11434",
            model_name="mistral:7b-instruct-q4_0",
            backend="ollama",
            embedding_model="all-MiniLM-L6-v2",
            k=3
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
                "description": "Semantic query - LLM should understand"
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

            # Display results similar to dense_only.py
            selected_server = evaluation['selected_servers'][0] if evaluation['selected_servers'] else 'None'
            selected_tool = evaluation['selected_tools'][0] if evaluation['selected_tools'] else 'None'
            print(f"Selected: {selected_server} (Tool: {selected_tool})")
            print(f"Candidates (k={evaluation['k']}): {evaluation['candidate_tools']}")
            print(f"Tokens: {evaluation['prompt_tokens']} prompt + {evaluation['completion_tokens']} completion = {evaluation['total_tokens']} total")
            print(f"Correct: {'✓' if evaluation['is_correct'] else '✗'}")
            print(f"Latency: {evaluation['latency_seconds']*1000:.2f}ms")

        # Summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        total_correct = sum(r['is_correct'] for r in results)
        accuracy = total_correct / len(results) * 100
        avg_latency = sum(r['latency_seconds'] for r in results) / len(results) * 1000
        avg_prompt_tokens = sum(r['prompt_tokens'] for r in results) / len(results)
        avg_total_tokens = sum(r['total_tokens'] for r in results) / len(results)

        print(f"Total queries: {len(results)}")
        print(f"Correct: {total_correct}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Average latency: {avg_latency:.2f}ms")
        print(f"Average prompt tokens: {avg_prompt_tokens:.0f}")
        print(f"Average total tokens: {avg_total_tokens:.0f}")
        print(f"k value: {approach.k}")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure:")
        print("1. Ollama is installed and running")
        print("2. Run: ollama run mistral:7b-instruct-q4_0")
        print("3. Ollama is accessible at http://localhost:11434")
        import traceback
        traceback.print_exc()
