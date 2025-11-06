"""
Benchmarker Module
Provides benchmarking tools for evaluating retrieval performance in the RAG-MCP system.
"""

import sys
from pathlib import Path
import json
import time
from typing import List, Dict, Any, Optional

# Add src directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from src.indexing import ToolIndexer
from src.retrieval.dense_retriever import ToolRetriever
from src.retrieval import RetrievalMetrics


class Benchmarker:
    """
    Benchmarking class for evaluating RAG-MCP retrieval performance.
    
    This class provides methods to benchmark different components of the RAG-MCP
    system, including retrieval-only evaluation.
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the Benchmarker.
        
        Args:
            project_root: Path to the project root directory. 
                         If None, will auto-detect from current file location.
        """
        if project_root is None:
            self.project_root = Path(__file__).parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.src_dir = self.project_root / "src"
        self.data_dir = self.project_root / "data"
        self.results_dir = self.data_dir / "results"
        
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Benchmarker initialized")
        print(f"  Project root: {self.project_root}")
        print(f"  Data directory: {self.data_dir}")
        print(f"  Results directory: {self.results_dir}")
    
    def benchmark_retrieval_only(
        self,
        tools_json_path: str,
        queries_json_path: str,
        model_name: str = 'all-MiniLM-L6-v2',
        k_values: List[int] = [1, 3, 5],
        batch_size: int = 8,
        save_results: bool = True,
        results_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Benchmark retrieval-only performance (no LLM selection).
        
        This method:
        1. Loads tools from JSON
        2. Builds FAISS index
        3. Loads test queries from JSON
        4. Runs retrieval for all queries
        5. Calculates metrics
        6. Analyzes failures
        7. Saves results
        
        Args:
            tools_json_path: Path to JSON file containing tool definitions
            queries_json_path: Path to JSON file containing test queries with ground truth
            model_name: Name of the sentence-transformer model to use
            k_values: List of k values to evaluate (e.g., [1, 3, 5])
            batch_size: Batch size for embedding generation
            save_results: Whether to save results to JSON file
            results_filename: Custom filename for results (default: auto-generated)
        
        Returns:
            Dictionary containing all benchmark results and metrics
        """
        print("=" * 80)
        print("RAG-MCP Retrieval Benchmarking")
        print("=" * 80)
        
        # ===== Step 1: Load Tools =====
        print("\n[Step 1] Loading tools...")
        print("-" * 80)
        
        tools_path = Path(tools_json_path)
        if not tools_path.is_absolute():
            tools_path = self.project_root / tools_json_path
        
        with open(tools_path, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        
        print(f"✓ Loaded {len(tools)} tools from: {tools_path}")
        print(f"  Categories: {set(tool.get('category', 'unknown') for tool in tools)}")
        
        # ===== Step 2: Build Index =====
        print("\n[Step 2] Building FAISS index...")
        print("-" * 80)
        
        # Initialize indexer with embedding model
        indexer = ToolIndexer(model_name=model_name)
        
        # Build index
        indexer.build_index(tools, batch_size=batch_size)
        
        # Save index
        index_dir = self.data_dir / "indexes"
        index_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = index_dir / f"tools_{model_name.replace('/', '_')}.index"
        metadata_path = index_dir / f"tools_{model_name.replace('/', '_')}.metadata.json"
        
        indexer.save_index(str(index_path), str(metadata_path))
        print(f"✓ Index saved to: {index_path}")
        
        # ===== Step 3: Initialize Retriever =====
        print("\n[Step 3] Initializing retriever...")
        print("-" * 80)
        
        retriever = ToolRetriever(model_name=model_name)
        retriever.load_index(str(index_path), str(metadata_path))
        print(f"✓ Retriever ready with {len(retriever.tool_metadata)} tools")
        
        # ===== Step 4: Load Test Queries =====
        print("\n[Step 4] Loading test queries...")
        print("-" * 80)
        
        queries_path = Path(queries_json_path)
        if not queries_path.is_absolute():
            queries_path = self.project_root / queries_json_path
        
        with open(queries_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Parse the schema and extract test queries
        test_queries = []
        
        if isinstance(raw_data, dict) and 'server_tasks' in raw_data:
            # New mcp_task_description.json schema
            print(f"✓ Detected mcp_task_description.json schema")
            for server_task in raw_data.get('server_tasks', []):
                server_name = server_task.get('server_name', 'Unknown')
                for task in server_task.get('tasks', []):
                    query_entry = {
                        'query': task.get('fuzzy_description', ''),
                        'fuzzy_query': task.get('fuzzy_description', ''),
                        'ground_truth_tool': server_name,
                        'query_id': task.get('task_id', ''),
                        'category': server_task.get('combination_type', 'single_server'),
                        'difficulty': 'unknown',
                        'distraction_servers': task.get('distraction_servers', [])
                    }
                    test_queries.append(query_entry)
            print(f"✓ Extracted {len(test_queries)} tasks from {len(raw_data.get('server_tasks', []))} servers")
        elif isinstance(raw_data, list):
            # Legacy schema: direct list of queries
            print(f"✓ Detected legacy query list schema")
            test_queries = raw_data
        else:
            raise ValueError(f"Unknown query JSON schema format")
        
        print(f"✓ Loaded {len(test_queries)} test queries from: {queries_path}")
        
        # Validate query format
        required_fields = ['query', 'ground_truth_tool']
        for i, query_data in enumerate(test_queries):
            for field in required_fields:
                if field not in query_data:
                    raise ValueError(
                        f"Query {i} missing required field '{field}'. "
                        f"Required fields: {required_fields}"
                    )
        
        # ===== Step 5: Run Retrieval =====
        print(f"\n[Step 5] Running retrieval for all queries (k={max(k_values)})...")
        print("-" * 80)
        
        retrieval_results = []
        retrieval_times = []
        
        for query_data in test_queries:
            query = query_data['query']
            ground_truth_server = query_data['ground_truth_tool']  # This is server_name
            
            # Retrieve top-k tools (use max k_value to get all needed) and measure time
            start_time = time.time()
            retrieved_tools = retriever.retrieve(query, k=max(k_values), return_scores=True)
            retrieval_time = time.time() - start_time
            
            # Extract server names from retrieved tools
            retrieved_servers = [tool.get('server', 'Unknown') for tool in retrieved_tools]
            retrieval_times.append(retrieval_time)
            
            # Check if first (highest scoring) tool matches ground truth
            is_correct = len(retrieved_servers) > 0 and retrieved_servers[0] == ground_truth_server
            
            # Store results (using server names for comparison)
            result = {
                'query': query,
                'retrieved_servers': retrieved_servers,  # Changed from retrieved_ids
                'ground_truth_server': ground_truth_server,  # Changed from ground_truth_id
                'query_id': query_data.get('query_id', f'q{len(retrieval_results)+1:03d}'),
                'retrieval_time_ms': retrieval_time * 1000,  # Convert to milliseconds
                'is_correct': is_correct  # True if first retrieved server matches ground truth
            }
            retrieval_results.append(result)
            
            # Display results for top-3
            status = "✓" if is_correct else "✗"
            
            print(f"\n{status} Query: '{query[:100]}...' ({retrieval_time*1000:.2f}ms)")
            print(f"  Ground Truth Server: {ground_truth_server}")
            print(f"  Retrieved (Top-3):")
            for i, tool in enumerate(retrieved_tools[:3]):
                server = tool.get('server', 'Unknown')
                marker = "→" if server == ground_truth_server else " "
                print(f"    {marker} [{tool['rank']}] {tool['tool_id']} (Server: {server}) "
                      f"(score: {tool['similarity_score']:.4f})")
        
        # ===== Step 6: Calculate Metrics =====
        print("\n[Step 6] Calculating retrieval metrics...")
        print("-" * 80)
        
        # Overall metrics
        metrics = RetrievalMetrics.calculate_all_metrics(
            retrieval_results,
            k_values=k_values
        )
        
        # Add performance metrics
        metrics['avg_retrieval_time_ms'] = sum(retrieval_times) / len(retrieval_times) * 1000
        metrics['min_retrieval_time_ms'] = min(retrieval_times) * 1000
        metrics['max_retrieval_time_ms'] = max(retrieval_times) * 1000
        
        RetrievalMetrics.print_summary(metrics, "Overall Retrieval Performance")
        
        # ===== Step 7: Analyze Failures =====
        print("\n[Step 7] Analyzing failure cases...")
        print("-" * 80)
        
        failures = RetrievalMetrics.get_failure_cases(retrieval_results, k=1)
        
        if failures:
            print(f"\nFound {len(failures)} failure cases (first retrieved server doesn't match ground truth):")
            for i, failure in enumerate(failures, 1):
                query_preview = failure['query'][:100] + '...' if len(failure['query']) > 100 else failure['query']
                print(f"\n  [{i}] Query: '{query_preview}'")
                print(f"      Expected Server: {failure['ground_truth_server']}")
                print(f"      Retrieved Servers (Top-3): {failure['retrieved_servers'][:3]}")
                print(f"      First Retrieved: {failure['retrieved_servers'][0] if failure['retrieved_servers'] else 'None'}")
        else:
            print("\n✓ No failures! All first retrieved servers match ground truth.")
        
        # ===== Step 8: Example LLM Integration =====
        # print("\n[Step 8] Example: Formatting for LLM consumption...")
        # print("-" * 80)
        
        # if test_queries:
        #     example_query = test_queries[0]['query']
        #     retrieved = retriever.retrieve(example_query, k=3, return_scores=False)
            
        #     print(f"\nQuery: '{example_query}'")
        #     print("\nFormatted tools for LLM prompt:")
        #     print("-" * 80)
            
        #     formatted = retriever.format_tools_for_llm(
        #         retrieved,
        #         include_rank=True,
        #         include_score=False
        #     )
        #     print(formatted)
        
        # ===== Step 9: Save Results =====
        if save_results:
            print("\n[Step 9] Saving results...")
            print("-" * 80)
            
            if results_filename is None:
                results_filename = f"retrieval_benchmark_{model_name.replace('/', '_')}.json"
            
            results_path = self.results_dir / results_filename
            
            output_data = {
                'benchmark_config': {
                    'model_name': model_name,
                    'k_values': k_values,
                    'batch_size': batch_size,
                    'tools_path': str(tools_path),
                    'queries_path': str(queries_path),
                    'num_tools': len(tools),
                    'num_queries': len(test_queries)
                },
                'queries': test_queries,
                'retrieval_results': retrieval_results,
                'metrics': {
                    'overall': metrics
                },
                'failures': failures
            }
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Results saved to: {results_path}")
        
        # ===== Summary =====
        print("\n" + "=" * 80)
        print("Benchmark Complete!")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  - Model: {model_name}")
        print(f"  - Tools indexed: {len(tools)}")
        print(f"  - Queries processed: {len(test_queries)}")
        print(f"  - Accuracy@1 (first result correct): {metrics['accuracy@1']:.2%}")
        print(f"  - Overall Recall@3: {metrics['recall@3']:.2%}")
        print(f"  - Overall MRR: {metrics['mrr']:.4f}")
        print(f"  - Success rate (accuracy@1): {(len(test_queries) - len(failures)) / len(test_queries):.2%}")
        print(f"  - Avg retrieval time: {metrics['avg_retrieval_time_ms']:.2f}ms")
        print(f"  - Time range: {metrics['min_retrieval_time_ms']:.2f}ms - {metrics['max_retrieval_time_ms']:.2f}ms")
        # print(f"\nNext steps:")
        # print(f"  1. Add more tools and queries")
        # print(f"  2. Experiment with different k values")
        # print(f"  3. Try different embedding models (e.g., intfloat/e5-base-v2)")
        # print(f"  4. Integrate with LLM for final tool selection")
        print("=" * 80)
        
        # Return complete results
        return {
            'config': {
                'model_name': model_name,
                'k_values': k_values,
                'num_tools': len(tools),
                'num_queries': len(test_queries)
            },
            'metrics': metrics,
            'failures': failures,
            'retrieval_results': retrieval_results
        }


def main():
    """
    Example usage of the Benchmarker class.
    """
    # Initialize benchmarker
    benchmarker = Benchmarker()
    
    # Run retrieval-only benchmark
    results = benchmarker.benchmark_retrieval_only(
        tools_json_path="data/tools/all_tools.json",
        queries_json_path="data/queries/mcp_task_description.json",
        model_name='all-MiniLM-L6-v2',
        k_values=[1, 3, 5],
        batch_size=8,
        save_results=True
    )
    
    print(f"\n✓ Benchmark completed successfully!")
    print(f"  Accuracy@1 (first result correct): {results['metrics']['accuracy@1']:.2%}")
    print(f"  Overall Recall@3: {results['metrics']['recall@3']:.2%}")
    print(f"  Overall MRR: {results['metrics']['mrr']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
