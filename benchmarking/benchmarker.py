"""
Benchmarker Module
Provides benchmarking tools for evaluating retrieval performance in the RAG-MCP system.
"""

import sys
from pathlib import Path
import json
import time
import argparse
from typing import List, Dict, Any, Optional

# Add src directory to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

from src.indexing import ToolIndexer
from src.indexing.bm25_indexer import BM25Indexer
from src.retrieval.dense_retriever import ToolRetriever
from src.retrieval import RetrievalMetrics
from src.approaches.bm25_only import BM25OnlyApproach
from src.approaches.bm25_plus_dense import BM25PlusDenseApproach
from src.approaches.llm_only import LLMOnlyApproach
from src.approaches.dense_llm import DenseLLMApproach
from src.approaches.bm25_llm import BM25LLMApproach
from src.approaches.llm_hybrid import LLMHybridApproach

# Default configuration (can be overridden by CLI arguments)
TOOLS_PATH = "data/tools/tools_list.json"
QUERIES_PATH = "data/queries/mcp_task_description.json"
K_VALUES = [3]
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
BATCH_SIZE = 8
RRF_K = 60
RETRIEVAL_K = 3
LIMIT_QUERIES = -1  # -1 means all queries
LLM_SERVER_URL = "http://localhost:8000"  # vLLM default port
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"  # vLLM model format
LLM_BACKEND = "vllm"  # vLLM backend type


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

        # Limit to first N queries if LIMIT_QUERIES is set
        if LIMIT_QUERIES is not None and LIMIT_QUERIES > 0:
            test_queries = test_queries[:LIMIT_QUERIES]
            print(f"  ⚠️  Limited to first {LIMIT_QUERIES} queries for local testing")

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
        print(f"\n[Step 5] Running retrieval for {len(test_queries)} queries (k={max(k_values)})...")
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

            # Show progress every 10 queries
            if (len(retrieval_results)) % 10 == 0:
                correct_so_far = sum(1 for r in retrieval_results if r['is_correct'])
                accuracy_so_far = correct_so_far / len(retrieval_results) * 100
                print(f"  Progress: {len(retrieval_results)}/{len(test_queries)} queries "
                      f"(Accuracy so far: {accuracy_so_far:.1f}%)")
        
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
            # for i, failure in enumerate(failures, 1):
            #     query_preview = failure['query'][:100] + '...' if len(failure['query']) > 100 else failure['query']
            #     print(f"\n  [{i}] Query: '{query_preview}'")
            #     print(f"      Expected Server: {failure['ground_truth_server']}")
            #     print(f"      Retrieved Servers (Top-3): {failure['retrieved_servers'][:3]}")
            #     print(f"      First Retrieved: {failure['retrieved_servers'][0] if failure['retrieved_servers'] else 'None'}")
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
                'approach': 'dense_only',
                'config': {
                    'model_name': model_name,
                    'k_values': k_values,
                    'batch_size': batch_size,
                    'num_tools': len(tools),
                    'num_queries': len(test_queries)
                },
                'metrics': metrics
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
    
    def benchmark_bm25_only(
        self,
        tools_json_path: str,
        queries_json_path: str,
        save_results: bool = True,
        results_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Benchmark BM25-only approach.
        
        Args:
            tools_json_path: Path to JSON file containing tool definitions
            queries_json_path: Path to JSON file containing test queries with ground truth
            save_results: Whether to save results to JSON file
            results_filename: Custom filename for results
        
        Returns:
            Dictionary containing all benchmark results and metrics
        """
        print("=" * 80)
        print("BM25-Only Approach Benchmarking")
        print("=" * 80)
        
        # Load tools and queries
        tools_path = Path(tools_json_path) if Path(tools_json_path).is_absolute() else self.project_root / tools_json_path
        queries_path = Path(queries_json_path) if Path(queries_json_path).is_absolute() else self.project_root / queries_json_path
        
        with open(tools_path, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        
        with open(queries_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Parse test queries
        test_queries = self._parse_queries(raw_data)
        
        print(f"\n✓ Loaded {len(tools)} tools and {len(test_queries)} queries")
        
        # Build BM25 index
        print("\nBuilding BM25 index...")
        indexer = BM25Indexer()
        indexer.build_index(tools)
        
        index_dir = self.data_dir / "indexes"
        index_dir.mkdir(parents=True, exist_ok=True)
        bm25_index_path = index_dir / "bm25_index.pkl"
        indexer.save_index(str(bm25_index_path))
        
        # Initialize approach
        approach = BM25OnlyApproach(index_path=str(bm25_index_path))
        
        # Run evaluations
        print("\nRunning evaluations...")
        results = []
        for query_data in test_queries:
            evaluation = approach.evaluate_query(
                query_data['query'],
                query_data['ground_truth_tool']
            )
            results.append(evaluation)
        
        # Calculate metrics
        metrics = self._calculate_approach_metrics(results)
        
        # Save and return
        return self._save_and_return_results(
            approach_name="bm25_only",
            results=results,
            metrics=metrics,
            config={'num_tools': len(tools), 'num_queries': len(test_queries)},
            save_results=save_results,
            results_filename=results_filename
        )
    
    def benchmark_bm25_plus_dense(
        self,
        tools_json_path: str,
        queries_json_path: str,
        model_name: str = 'all-MiniLM-L6-v2',
        rrf_k: int = 60,
        batch_size: int = 8,
        save_results: bool = True,
        results_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Benchmark BM25 + Dense hybrid approach.
        
        Args:
            tools_json_path: Path to JSON file containing tool definitions
            queries_json_path: Path to JSON file containing test queries
            model_name: Embedding model name
            rrf_k: RRF constant
            batch_size: Batch size for embedding
            save_results: Whether to save results
            results_filename: Custom filename for results
        
        Returns:
            Dictionary containing benchmark results
        """
        print("=" * 80)
        print("BM25 + Dense Hybrid Approach Benchmarking")
        print("=" * 80)
        
        # Load data
        tools_path = Path(tools_json_path) if Path(tools_json_path).is_absolute() else self.project_root / tools_json_path
        queries_path = Path(queries_json_path) if Path(queries_json_path).is_absolute() else self.project_root / queries_json_path
        
        with open(tools_path, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        
        with open(queries_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        test_queries = self._parse_queries(raw_data)
        print(f"\n✓ Loaded {len(tools)} tools and {len(test_queries)} queries")
        
        # Build indexes
        print("\nBuilding indexes...")
        index_dir = self.data_dir / "indexes"
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # BM25 index
        bm25_indexer = BM25Indexer()
        bm25_indexer.build_index(tools)
        bm25_index_path = index_dir / "bm25_index.pkl"
        bm25_indexer.save_index(str(bm25_index_path))
        
        # Dense index
        dense_indexer = ToolIndexer(model_name=model_name)
        dense_indexer.build_index(tools, batch_size=batch_size)
        dense_index_path = index_dir / f"tools_{model_name.replace('/', '_')}.index"
        dense_metadata_path = index_dir / f"tools_{model_name.replace('/', '_')}.metadata.json"
        dense_indexer.save_index(str(dense_index_path), str(dense_metadata_path))
        
        # Initialize approach
        approach = BM25PlusDenseApproach(
            bm25_index_path=str(bm25_index_path),
            dense_index_path=str(dense_index_path),
            dense_metadata_path=str(dense_metadata_path),
            model_name=model_name,
            rrf_k=rrf_k
        )
        
        # Run evaluations
        print("\nRunning evaluations...")
        results = []
        for query_data in test_queries:
            evaluation = approach.evaluate_query(
                query_data['query'],
                query_data['ground_truth_tool']
            )
            results.append(evaluation)
        
        metrics = self._calculate_approach_metrics(results)
        
        return self._save_and_return_results(
            approach_name="bm25_plus_dense",
            results=results,
            metrics=metrics,
            config={'model_name': model_name, 'rrf_k': rrf_k, 'num_tools': len(tools), 'num_queries': len(test_queries)},
            save_results=save_results,
            results_filename=results_filename
        )
    
    def benchmark_llm_only(
        self,
        tools_json_path: str,
        queries_json_path: str,
        server_url: str = "http://localhost:11434",
        model_name: str = "mistral:7b-instruct-q4_0",
        backend: str = "ollama",
        save_results: bool = True,
        results_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Benchmark LLM-only approach.
        
        Args:
            tools_json_path: Path to tools JSON
            queries_json_path: Path to queries JSON
            server_url: LLM server URL
            model_name: LLM model name
            backend: 'ollama' or 'vllm'
            save_results: Whether to save results
            results_filename: Custom filename
        
        Returns:
            Benchmark results dictionary
        """
        print("=" * 80)
        print("LLM-Only Approach Benchmarking")
        print("=" * 80)
        
        # Load data
        tools_path = Path(tools_json_path) if Path(tools_json_path).is_absolute() else self.project_root / tools_json_path
        queries_path = Path(queries_json_path) if Path(queries_json_path).is_absolute() else self.project_root / queries_json_path
        
        with open(tools_path, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        
        with open(queries_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        test_queries = self._parse_queries(raw_data)
        print(f"\n✓ Loaded {len(tools)} tools and {len(test_queries)} queries")
        
        # Initialize approach
        approach = LLMOnlyApproach(
            tools=tools,
            server_url=server_url,
            model_name=model_name,
            backend=backend
        )
        
        # Run evaluations
        print("\nRunning evaluations...")
        results = []
        for i, query_data in enumerate(test_queries, 1):
            evaluation = approach.evaluate_query(
                query_data['query'],
                query_data['ground_truth_tool']
            )
            results.append(evaluation)

            # Progress every 10 queries
            if i % 10 == 0:
                print(f"  {i}/{len(test_queries)} queries")

        metrics = self._calculate_llm_approach_metrics(results)
        
        return self._save_and_return_results(
            approach_name="llm_only",
            results=results,
            metrics=metrics,
            config={'server_url': server_url, 'model_name': model_name, 'backend': backend, 'num_tools': len(tools), 'num_queries': len(test_queries)},
            save_results=save_results,
            results_filename=results_filename
        )
    
    def benchmark_dense_llm(
        self,
        tools_json_path: str,
        queries_json_path: str,
        server_url: str = "http://localhost:11434",
        llm_model_name: str = "mistral:7b-instruct-q4_0",
        backend: str = "ollama",
        embedding_model: str = "all-MiniLM-L6-v2",
        k: int = 5,
        batch_size: int = 8,
        save_results: bool = True,
        results_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Benchmark Dense + LLM approach.
        
        Args:
            tools_json_path: Path to tools JSON
            queries_json_path: Path to queries JSON
            server_url: LLM server URL
            llm_model_name: LLM model name
            backend: 'ollama' or 'vllm'
            embedding_model: Embedding model name
            k: Number of candidates to retrieve
            batch_size: Batch size for embedding
            save_results: Whether to save results
            results_filename: Custom filename
        
        Returns:
            Benchmark results dictionary
        """
        print("=" * 80)
        print("Dense + LLM Approach Benchmarking")
        print("=" * 80)
        
        # Load data
        tools_path = Path(tools_json_path) if Path(tools_json_path).is_absolute() else self.project_root / tools_json_path
        queries_path = Path(queries_json_path) if Path(queries_json_path).is_absolute() else self.project_root / queries_json_path
        
        with open(tools_path, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        
        with open(queries_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        test_queries = self._parse_queries(raw_data)
        print(f"\n✓ Loaded {len(tools)} tools and {len(test_queries)} queries")
        
        # Build dense index
        print("\nBuilding dense index...")
        indexer = ToolIndexer(model_name=embedding_model)
        indexer.build_index(tools, batch_size=batch_size)
        
        index_dir = self.data_dir / "indexes"
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / f"tools_{embedding_model.replace('/', '_')}.index"
        metadata_path = index_dir / f"tools_{embedding_model.replace('/', '_')}.metadata.json"
        indexer.save_index(str(index_path), str(metadata_path))
        
        # Initialize approach
        approach = DenseLLMApproach(
            index_path=str(index_path),
            metadata_path=str(metadata_path),
            server_url=server_url,
            model_name=llm_model_name,
            backend=backend,
            embedding_model=embedding_model,
            k=k
        )
        
        # Run evaluations
        print("\nRunning evaluations...")
        results = []
        for i, query_data in enumerate(test_queries, 1):
            evaluation = approach.evaluate_query(
                query_data['query'],
                query_data['ground_truth_tool']
            )
            results.append(evaluation)

            # Progress every 10 queries
            if i % 10 == 0:
                print(f"  {i}/{len(test_queries)} queries")

        metrics = self._calculate_llm_approach_metrics(results)
        
        return self._save_and_return_results(
            approach_name="dense_llm",
            results=results,
            metrics=metrics,
            config={'embedding_model': embedding_model, 'llm_model': llm_model_name, 'k': k, 'num_tools': len(tools), 'num_queries': len(test_queries)},
            save_results=save_results,
            results_filename=results_filename
        )

    def benchmark_bm25_llm(
        self,
        tools_json_path: str,
        queries_json_path: str,
        server_url: str = "http://localhost:11434",
        llm_model_name: str = "mistral:7b-instruct-q4_0",
        backend: str = "ollama",
        k: int = 5,
        save_results: bool = True,
        results_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Benchmark BM25 + LLM approach.

        Args:
            tools_json_path: Path to tools JSON
            queries_json_path: Path to queries JSON
            server_url: LLM server URL
            llm_model_name: LLM model name
            backend: 'ollama' or 'vllm'
            k: Number of candidates to retrieve
            save_results: Whether to save results
            results_filename: Custom filename

        Returns:
            Benchmark results dictionary
        """
        print("=" * 80)
        print("BM25 + LLM Approach Benchmarking")
        print("=" * 80)

        # Load data
        tools_path = Path(tools_json_path) if Path(tools_json_path).is_absolute() else self.project_root / tools_json_path
        queries_path = Path(queries_json_path) if Path(queries_json_path).is_absolute() else self.project_root / queries_json_path

        with open(tools_path, 'r', encoding='utf-8') as f:
            tools = json.load(f)

        with open(queries_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        test_queries = self._parse_queries(raw_data)
        print(f"\n✓ Loaded {len(tools)} tools and {len(test_queries)} queries")

        # Build BM25 index
        print("\nBuilding BM25 index...")
        indexer = BM25Indexer()
        indexer.build_index(tools)

        index_dir = self.data_dir / "indexes"
        index_dir.mkdir(parents=True, exist_ok=True)
        index_path = index_dir / "tools_bm25.index"
        indexer.save_index(str(index_path))

        # Initialize approach
        approach = BM25LLMApproach(
            index_path=str(index_path),
            server_url=server_url,
            model_name=llm_model_name,
            backend=backend,
            k=k
        )

        # Run evaluations
        print("\nRunning evaluations...")
        results = []
        for query_data in test_queries:
            try:
                evaluation = approach.evaluate_query(
                    query_data['query'],
                    query_data['ground_truth_tool']
                )
                results.append(evaluation)
            except Exception as e:
                print(f"Error on query '{query_data['query'][:50]}...': {e}")
                continue

        metrics = self._calculate_llm_approach_metrics(results)

        return self._save_and_return_results(
            approach_name="bm25_llm",
            results=results,
            metrics=metrics,
            config={'llm_model': llm_model_name, 'k': k, 'num_tools': len(tools), 'num_queries': len(test_queries)},
            save_results=save_results,
            results_filename=results_filename
        )

    def benchmark_llm_hybrid(
        self,
        tools_json_path: str,
        queries_json_path: str,
        server_url: str = "http://localhost:11434",
        llm_model_name: str = "mistral:7b-instruct-q4_0",
        backend: str = "ollama",
        embedding_model: str = "all-MiniLM-L6-v2",
        rrf_k: int = 60,
        retrieval_k: int = 5,
        batch_size: int = 8,
        save_results: bool = True,
        results_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Benchmark Hybrid (BM25+Dense) + LLM approach.
        
        Args:
            tools_json_path: Path to tools JSON
            queries_json_path: Path to queries JSON
            server_url: LLM server URL
            llm_model_name: LLM model name
            backend: 'ollama' or 'vllm'
            embedding_model: Embedding model name
            rrf_k: RRF constant
            retrieval_k: Number of candidates to retrieve
            batch_size: Batch size for embedding
            save_results: Whether to save results
            results_filename: Custom filename
        
        Returns:
            Benchmark results dictionary
        """
        print("=" * 80)
        print("Hybrid (BM25+Dense) + LLM Approach Benchmarking")
        print("=" * 80)
        
        # Load data
        tools_path = Path(tools_json_path) if Path(tools_json_path).is_absolute() else self.project_root / tools_json_path
        queries_path = Path(queries_json_path) if Path(queries_json_path).is_absolute() else self.project_root / queries_json_path
        
        with open(tools_path, 'r', encoding='utf-8') as f:
            tools = json.load(f)
        
        with open(queries_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        test_queries = self._parse_queries(raw_data)
        print(f"\n✓ Loaded {len(tools)} tools and {len(test_queries)} queries")
        
        # Build indexes
        print("\nBuilding indexes...")
        index_dir = self.data_dir / "indexes"
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # BM25 index
        bm25_indexer = BM25Indexer()
        bm25_indexer.build_index(tools)
        bm25_index_path = index_dir / "bm25_index.pkl"
        bm25_indexer.save_index(str(bm25_index_path))
        
        # Dense index
        dense_indexer = ToolIndexer(model_name=embedding_model)
        dense_indexer.build_index(tools, batch_size=batch_size)
        dense_index_path = index_dir / f"tools_{embedding_model.replace('/', '_')}.index"
        dense_metadata_path = index_dir / f"tools_{embedding_model.replace('/', '_')}.metadata.json"
        dense_indexer.save_index(str(dense_index_path), str(dense_metadata_path))
        
        # Initialize approach
        approach = LLMHybridApproach(
            bm25_index_path=str(bm25_index_path),
            dense_index_path=str(dense_index_path),
            dense_metadata_path=str(dense_metadata_path),
            server_url=server_url,
            model_name=llm_model_name,
            backend=backend,
            embedding_model=embedding_model,
            rrf_k=rrf_k,
            retrieval_k=retrieval_k
        )
        
        # Run evaluations
        print("\nRunning evaluations...")
        results = []
        for i, query_data in enumerate(test_queries, 1):
            evaluation = approach.evaluate_query(
                query_data['query'],
                query_data['ground_truth_tool']
            )
            results.append(evaluation)

            # Progress every 10 queries
            if i % 10 == 0:
                print(f"  {i}/{len(test_queries)} queries")

        metrics = self._calculate_llm_approach_metrics(results)
        
        return self._save_and_return_results(
            approach_name="llm_hybrid",
            results=results,
            metrics=metrics,
            config={'embedding_model': embedding_model, 'llm_model': llm_model_name, 'rrf_k': rrf_k, 'retrieval_k': retrieval_k, 'num_tools': len(tools), 'num_queries': len(test_queries)},
            save_results=save_results,
            results_filename=results_filename
        )
    
    def _parse_queries(self, raw_data: Any) -> List[Dict[str, Any]]:
        """Parse queries from JSON data."""
        test_queries = []

        if isinstance(raw_data, dict) and 'server_tasks' in raw_data:
            for server_task in raw_data.get('server_tasks', []):
                server_name = server_task.get('server_name', 'Unknown')
                for task in server_task.get('tasks', []):
                    query_entry = {
                        'query': task.get('fuzzy_description', ''),
                        'ground_truth_tool': server_name,
                        'query_id': task.get('task_id', ''),
                        'category': server_task.get('combination_type', 'single_server'),
                    }
                    test_queries.append(query_entry)
        elif isinstance(raw_data, list):
            test_queries = raw_data
        else:
            raise ValueError("Unknown query JSON schema format")

        # Limit to first N queries if LIMIT_QUERIES is set
        if LIMIT_QUERIES is not None and LIMIT_QUERIES > 0:
            test_queries = test_queries[:LIMIT_QUERIES]
            print(f"  ⚠️  Limited to first {LIMIT_QUERIES} queries for local testing")

        return test_queries
    
    def _calculate_approach_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for non-LLM approaches including Recall@k and MRR."""
        if not results:
            return {}

        # Accuracy (top-1)
        total_correct = sum(1 for r in results if r.get('is_correct', False))
        accuracy = total_correct / len(results)

        # Latency
        avg_latency = sum(r.get('latency_seconds', 0) for r in results) / len(results)

        # Recall@k for k = 1, 3, 5, 7
        recall_at_1 = RetrievalMetrics.average_recall_at_k(results, k=1)
        recall_at_3 = RetrievalMetrics.average_recall_at_k(results, k=3)
        recall_at_5 = RetrievalMetrics.average_recall_at_k(results, k=5)
        recall_at_7 = RetrievalMetrics.average_recall_at_k(results, k=7)

        # MRR (Mean Reciprocal Rank)
        mrr = RetrievalMetrics.mean_reciprocal_rank(results)

        return {
            'accuracy': accuracy,
            'num_correct': total_correct,
            'num_queries': len(results),
            'avg_latency_ms': avg_latency * 1000,
            'recall@1': recall_at_1,
            'recall@3': recall_at_3,
            'recall@5': recall_at_5,
            'recall@7': recall_at_7,
            'mrr': mrr
        }
    
    def _calculate_llm_approach_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for LLM-based approaches including Recall@k and MRR."""
        if not results:
            return {}

        # Accuracy (top-1)
        total_correct = sum(1 for r in results if r.get('is_correct', False))
        accuracy = total_correct / len(results)

        # Latency
        avg_latency = sum(r.get('latency_seconds', 0) for r in results) / len(results)

        # Token usage
        avg_prompt_tokens = sum(r.get('prompt_tokens', 0) for r in results) / len(results)
        avg_completion_tokens = sum(r.get('completion_tokens', 0) for r in results) / len(results)
        avg_total_tokens = sum(r.get('total_tokens', 0) for r in results) / len(results)

        # Recall@k for k = 1, 3, 5, 7
        recall_at_1 = RetrievalMetrics.average_recall_at_k(results, k=1)
        recall_at_3 = RetrievalMetrics.average_recall_at_k(results, k=3)
        recall_at_5 = RetrievalMetrics.average_recall_at_k(results, k=5)
        recall_at_7 = RetrievalMetrics.average_recall_at_k(results, k=7)

        # MRR (Mean Reciprocal Rank)
        mrr = RetrievalMetrics.mean_reciprocal_rank(results)

        return {
            'accuracy': accuracy,
            'num_correct': total_correct,
            'num_queries': len(results),
            'avg_latency_ms': avg_latency * 1000,
            'avg_prompt_tokens': avg_prompt_tokens,
            'avg_completion_tokens': avg_completion_tokens,
            'avg_total_tokens': avg_total_tokens,
            'recall@1': recall_at_1,
            'recall@3': recall_at_3,
            'recall@5': recall_at_5,
            'recall@7': recall_at_7,
            'mrr': mrr
        }
    
    def _save_and_return_results(
        self,
        approach_name: str,
        results: List[Dict[str, Any]],
        metrics: Dict[str, float],
        config: Dict[str, Any],
        save_results: bool,
        results_filename: Optional[str]
    ) -> Dict[str, Any]:
        """Save results and return summary."""
        # Print summary
        print("\n" + "=" * 80)
        print(f"{approach_name.upper()} - Benchmark Complete!")
        print("=" * 80)
        print(f"\nSummary:")
        print(f"  - Approach: {approach_name}")
        print(f"  - Queries processed: {metrics.get('num_queries', 0)}")
        print(f"  - Correct: {metrics.get('num_correct', 0)}")
        print(f"  - Accuracy: {metrics.get('accuracy', 0):.2%}")
        print(f"  - Avg latency: {metrics.get('avg_latency_ms', 0):.2f}ms")
        
        if 'avg_total_tokens' in metrics:
            print(f"  - Avg tokens: {metrics['avg_total_tokens']:.0f}")
        
        print("=" * 80)
        
        # Save results
        if save_results:
            if results_filename is None:
                results_filename = f"{approach_name}_benchmark.json"
            
            results_path = self.results_dir / results_filename
            
            output_data = {
                'approach': approach_name,
                'config': config,
                'metrics': metrics
            }

            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Results saved to: {results_path}")
        
        return {
            'approach': approach_name,
            'config': config,
            'metrics': metrics,
            'results': results
        }


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run RAG-MCP benchmarking experiments")

    parser.add_argument(
        "--server-url",
        type=str,
        default=LLM_SERVER_URL,
        help="LLM server URL (default: http://localhost:8000 for vLLM)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=LLM_MODEL_NAME,
        help="LLM model name (default: mistralai/Mistral-7B-Instruct-v0.3)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=LLM_BACKEND,
        choices=["vllm", "ollama"],
        help="LLM backend type (default: vllm)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results",
        help="Output directory for results (default: data/results)"
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=LIMIT_QUERIES,
        help="Limit number of queries to process, -1 for all (default: -1)"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs='+',
        default=K_VALUES,
        help="K values for retrieval (default: 3)"
    )

    return parser.parse_args()


def main():
    """
    Run all benchmarks sequentially.
    """
    # Parse command-line arguments
    args = parse_args()

    # Update global configuration with CLI arguments
    global LLM_SERVER_URL, LLM_MODEL_NAME, LLM_BACKEND, LIMIT_QUERIES, K_VALUES, RETRIEVAL_K
    LLM_SERVER_URL = args.server_url
    LLM_MODEL_NAME = args.model_name
    LLM_BACKEND = args.backend
    LIMIT_QUERIES = args.limit_queries
    K_VALUES = args.k_values
    RETRIEVAL_K = args.k_values[0]  # Use first k value for retrieval approaches

    print("\n" + "=" * 80)
    print("RUNNING ALL BENCHMARKS")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Tools: {TOOLS_PATH}")
    print(f"  Queries: {QUERIES_PATH}")
    print(f"  K Values: {K_VALUES}")
    print(f"  Embedding Model: {EMBEDDING_MODEL}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  RRF K: {RRF_K}")
    print(f"  Retrieval K: {RETRIEVAL_K}")
    print(f"  Limit Queries: {LIMIT_QUERIES if LIMIT_QUERIES > 0 else 'All'}")
    print(f"  LLM Server: {LLM_SERVER_URL}")
    print(f"  LLM Model: {LLM_MODEL_NAME}")
    print(f"  LLM Backend: {LLM_BACKEND}")
    print(f"  Output Directory: {args.output_dir}")
    print("=" * 80)

    # Initialize benchmarker with custom output directory
    benchmarker = Benchmarker()
    if args.output_dir != "data/results":
        benchmarker.results_dir = Path(args.output_dir)
        benchmarker.results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # 1. Dense Only (baseline - retrieval only)
    print("\n\n" + "=" * 80)
    print("BENCHMARK 1/7: Dense Only (Retrieval)")
    print("=" * 80)
    try:
        results = benchmarker.benchmark_retrieval_only(
            tools_json_path=TOOLS_PATH,
            queries_json_path=QUERIES_PATH,
            model_name=EMBEDDING_MODEL,
            k_values=K_VALUES,
            batch_size=BATCH_SIZE,
            save_results=True,
            results_filename="dense_only_benchmark.json"
        )
        all_results['dense_only'] = results
        print(f"✓ Dense Only - Accuracy: {results['metrics']['accuracy@1']:.2%}")
    except Exception as e:
        print(f"✗ Dense Only failed: {e}")
        all_results['dense_only'] = {'error': str(e)}
    
    # 2. BM25 Only
    print("\n\n" + "=" * 80)
    print("BENCHMARK 2/7: BM25 Only")
    print("=" * 80)
    try:
        results = benchmarker.benchmark_bm25_only(
            tools_json_path=TOOLS_PATH,
            queries_json_path=QUERIES_PATH,
            save_results=True,
            results_filename="bm25_only_benchmark.json"
        )
        all_results['bm25_only'] = results
        print(f"✓ BM25 Only - Accuracy: {results['metrics']['accuracy']:.2%}")
    except Exception as e:
        print(f"✗ BM25 Only failed: {e}")
        all_results['bm25_only'] = {'error': str(e)}
    
    # 3. BM25 + Dense (Hybrid - no LLM)
    print("\n\n" + "=" * 80)
    print("BENCHMARK 3/7: BM25 + Dense (Hybrid - no LLM)")
    print("=" * 80)
    try:
        results = benchmarker.benchmark_bm25_plus_dense(
            tools_json_path=TOOLS_PATH,
            queries_json_path=QUERIES_PATH,
            model_name=EMBEDDING_MODEL,
            rrf_k=RRF_K,
            batch_size=BATCH_SIZE,
            save_results=True,
            results_filename="bm25_plus_dense_benchmark.json"
        )
        all_results['bm25_plus_dense'] = results
        print(f"✓ BM25 + Dense - Accuracy: {results['metrics']['accuracy']:.2%}")
    except Exception as e:
        print(f"✗ BM25 + Dense failed: {e}")
        all_results['bm25_plus_dense'] = {'error': str(e)}
    
    # 4. LLM Only
    print("\n\n" + "=" * 80)
    print("BENCHMARK 4/7: LLM Only")
    print("=" * 80)
    try:
        results = benchmarker.benchmark_llm_only(
            tools_json_path=TOOLS_PATH,
            queries_json_path=QUERIES_PATH,
            server_url=LLM_SERVER_URL,
            model_name=LLM_MODEL_NAME,
            backend=LLM_BACKEND,
            save_results=True,
            results_filename="llm_only_benchmark.json"
        )
        all_results['llm_only'] = results
        print(f"✓ LLM Only - Accuracy: {results['metrics']['accuracy']:.2%}")
    except Exception as e:
        print(f"✗ LLM Only failed: {e}")
        all_results['llm_only'] = {'error': str(e)}
    
    # 5. Dense + LLM
    print("\n\n" + "=" * 80)
    print("BENCHMARK 5/7: Dense + LLM")
    print("=" * 80)
    try:
        results = benchmarker.benchmark_dense_llm(
            tools_json_path=TOOLS_PATH,
            queries_json_path=QUERIES_PATH,
            server_url=LLM_SERVER_URL,
            llm_model_name=LLM_MODEL_NAME,
            backend=LLM_BACKEND,
            embedding_model=EMBEDDING_MODEL,
            k=RETRIEVAL_K,
            batch_size=BATCH_SIZE,
            save_results=True,
            results_filename="dense_llm_benchmark.json"
        )
        all_results['dense_llm'] = results
        print(f"✓ Dense + LLM - Accuracy: {results['metrics']['accuracy']:.2%}")
    except Exception as e:
        print(f"✗ Dense + LLM failed: {e}")
        all_results['dense_llm'] = {'error': str(e)}

    # 6. BM25 + LLM
    print("\n\n" + "=" * 80)
    print("BENCHMARK 6/7: BM25 + LLM")
    print("=" * 80)
    try:
        results = benchmarker.benchmark_bm25_llm(
            tools_json_path=TOOLS_PATH,
            queries_json_path=QUERIES_PATH,
            server_url=LLM_SERVER_URL,
            llm_model_name=LLM_MODEL_NAME,
            backend=LLM_BACKEND,
            k=RETRIEVAL_K,
            save_results=True,
            results_filename="bm25_llm_benchmark.json"
        )
        all_results['bm25_llm'] = results
        print(f"✓ BM25 + LLM - Accuracy: {results['metrics']['accuracy']:.2%}")
    except Exception as e:
        print(f"✗ BM25 + LLM failed: {e}")
        all_results['bm25_llm'] = {'error': str(e)}

    # 7. LLM Hybrid (BM25 + Dense + LLM)
    print("\n\n" + "=" * 80)
    print("BENCHMARK 7/7: LLM Hybrid (BM25 + Dense + LLM)")
    print("=" * 80)
    try:
        results = benchmarker.benchmark_llm_hybrid(
            tools_json_path=TOOLS_PATH,
            queries_json_path=QUERIES_PATH,
            server_url=LLM_SERVER_URL,
            llm_model_name=LLM_MODEL_NAME,
            backend=LLM_BACKEND,
            embedding_model=EMBEDDING_MODEL,
            rrf_k=RRF_K,
            retrieval_k=RETRIEVAL_K,
            batch_size=BATCH_SIZE,
            save_results=True,
            results_filename="llm_hybrid_benchmark.json"
        )
        all_results['llm_hybrid'] = results
        print(f"✓ LLM Hybrid - Accuracy: {results['metrics']['accuracy']:.2%}")
    except Exception as e:
        print(f"✗ LLM Hybrid failed: {e}")
        all_results['llm_hybrid'] = {'error': str(e)}
    
    # Print final summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY - ALL BENCHMARKS")
    print("=" * 80)
    print(f"\n{'Approach':<25} {'Accuracy':<12} {'Avg Latency':<15} {'Status'}")
    print("-" * 80)
    
    for approach_name, result in all_results.items():
        if 'error' in result:
            print(f"{approach_name:<25} {'N/A':<12} {'N/A':<15} ✗ Failed")
        else:
            metrics = result.get('metrics', {})
            accuracy = metrics.get('accuracy') or metrics.get('accuracy@1', 0)
            latency = metrics.get('avg_latency_ms', 0)
            print(f"{approach_name:<25} {accuracy:<12.2%} {latency:<15.2f} ✓ Success")
    
    print("=" * 80)
    print("\n✓ All benchmarks completed!")
    print(f"Results saved to: {benchmarker.results_dir}/")
    
    return all_results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
