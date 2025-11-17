"""
Hybrid Retriever Module
Implements Reciprocal Rank Fusion (RRF) combining BM25 and Dense Retrieval.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import ToolRetriever


class HybridRetriever:
    """
    Hybrid retriever that combines BM25 and Dense retrieval using Reciprocal Rank Fusion (RRF).
    
    RRF Formula: score(d) = Σ 1 / (k + rank_i(d))
    where:
    - d is a document/tool
    - rank_i(d) is the rank of document d in retrieval system i
    - k is a constant (typically 60)
    """
    
    def __init__(
        self,
        bm25_index_path: Optional[str] = None,
        dense_index_path: Optional[str] = None,
        dense_metadata_path: Optional[str] = None,
        model_name: str = 'all-MiniLM-L6-v2',
        rrf_k: int = 60
    ):
        """
        Initialize the Hybrid Retriever.
        
        Args:
            bm25_index_path: Path to BM25 index file
            dense_index_path: Path to FAISS dense index file
            dense_metadata_path: Path to dense index metadata JSON
            model_name: Name of the sentence-transformer model for dense retrieval
            rrf_k: Constant k for RRF formula (default: 60)
        """
        self.rrf_k = rrf_k
        
        # Initialize BM25 retriever
        self.bm25_retriever = BM25Retriever(index_path=bm25_index_path)
        
        # Initialize Dense retriever
        self.dense_retriever = ToolRetriever(model_name=model_name)
        if dense_index_path and dense_metadata_path:
            self.dense_retriever.load_index(dense_index_path, dense_metadata_path)
        
        print(f"Initialized HybridRetriever with RRF (k={rrf_k})")
    
    def load_bm25_index(self, index_path: str) -> None:
        """
        Load BM25 index.
        
        Args:
            index_path: Path to BM25 index file
        """
        self.bm25_retriever.load_index(index_path)
    
    def load_dense_index(self, index_path: str, metadata_path: str) -> None:
        """
        Load Dense retrieval index.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
        """
        self.dense_retriever.load_index(index_path, metadata_path)
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results from BM25 and Dense retrieval using Reciprocal Rank Fusion.
        
        Args:
            bm25_results: Results from BM25 retriever
            dense_results: Results from Dense retriever
            k: RRF constant (typically 60)
            
        Returns:
            Fused and re-ranked results
        """
        # Create mapping: tool_id -> RRF score
        rrf_scores = {}
        tool_lookup = {}  # tool_id -> tool dict (for final output)
        
        # Process BM25 results
        for rank, tool in enumerate(bm25_results, start=1):
            tool_id = tool['tool_id']
            rrf_score = 1.0 / (k + rank)
            rrf_scores[tool_id] = rrf_scores.get(tool_id, 0.0) + rrf_score
            
            # Store tool info with BM25 score
            if tool_id not in tool_lookup:
                tool_lookup[tool_id] = tool.copy()
                tool_lookup[tool_id]['bm25_rank'] = rank
                tool_lookup[tool_id]['bm25_score'] = tool.get('bm25_score', 0.0)
            else:
                tool_lookup[tool_id]['bm25_rank'] = rank
                tool_lookup[tool_id]['bm25_score'] = tool.get('bm25_score', 0.0)
        
        # Process Dense results
        for rank, tool in enumerate(dense_results, start=1):
            tool_id = tool['tool_id']
            rrf_score = 1.0 / (k + rank)
            rrf_scores[tool_id] = rrf_scores.get(tool_id, 0.0) + rrf_score
            
            # Store tool info with Dense score
            if tool_id not in tool_lookup:
                tool_lookup[tool_id] = tool.copy()
                tool_lookup[tool_id]['dense_rank'] = rank
                tool_lookup[tool_id]['dense_score'] = tool.get('similarity_score', 0.0)
            else:
                tool_lookup[tool_id]['dense_rank'] = rank
                tool_lookup[tool_id]['dense_score'] = tool.get('similarity_score', 0.0)
        
        # Sort by RRF score (descending)
        sorted_tool_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build final results
        fused_results = []
        for new_rank, (tool_id, rrf_score) in enumerate(sorted_tool_ids, start=1):
            tool = tool_lookup[tool_id].copy()
            tool['rank'] = new_rank
            tool['rrf_score'] = rrf_score
            
            # Add missing rank/score fields if tool was only in one retriever
            if 'bm25_rank' not in tool:
                tool['bm25_rank'] = None
                tool['bm25_score'] = 0.0
            if 'dense_rank' not in tool:
                tool['dense_rank'] = None
                tool['dense_score'] = 0.0
            
            fused_results.append(tool)
        
        return fused_results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        bm25_k: Optional[int] = None,
        dense_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve tools using hybrid RRF approach.
        
        Args:
            query: User query string
            top_k: Number of final results to return
            bm25_k: Number of results to retrieve from BM25 (default: 2 * top_k)
            dense_k: Number of results to retrieve from Dense (default: 2 * top_k)
            
        Returns:
            List of top-k tools ranked by RRF score
        """
        # Default: retrieve more from each system than final top_k
        if bm25_k is None:
            bm25_k = max(top_k * 2, 10)
        if dense_k is None:
            dense_k = max(top_k * 2, 10)
        
        # Retrieve from both systems
        bm25_results = self.bm25_retriever.retrieve(query, top_k=bm25_k)
        dense_results = self.dense_retriever.retrieve(query, k=dense_k, return_scores=True)
        
        # Apply RRF fusion
        fused_results = self._reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            k=self.rrf_k
        )
        
        # Return top-k
        return fused_results[:top_k]
    
    def retrieve_top1(self, query: str) -> Dict[str, Any]:
        """
        Retrieve only the top-1 tool using RRF.
        
        Args:
            query: User query string
            
        Returns:
            Single tool dictionary with RRF score
        """
        results = self.retrieve(query, top_k=1)
        return results[0] if results else None
    
    def retrieve_separate(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve from both systems separately (for analysis/comparison).
        
        Args:
            query: User query string
            top_k: Number of results from each system
            
        Returns:
            Dictionary with 'bm25', 'dense', and 'hybrid' results
        """
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k)
        dense_results = self.dense_retriever.retrieve(query, k=top_k, return_scores=True)
        hybrid_results = self.retrieve(query, top_k=top_k)
        
        return {
            'bm25': bm25_results,
            'dense': dense_results,
            'hybrid': hybrid_results
        }
    
    def compare_retrievers(
        self,
        query: str,
        top_k: int = 5
    ) -> None:
        """
        Print comparison of all three retrieval methods.
        
        Args:
            query: User query string
            top_k: Number of results to show from each
        """
        results = self.retrieve_separate(query, top_k=top_k)
        
        print(f"\n{'='*80}")
        print(f"Query: '{query}'")
        print(f"{'='*80}")
        
        # BM25 Results
        print(f"\n{'BM25 Results:':^80}")
        print(f"{'-'*80}")
        for i, tool in enumerate(results['bm25'], 1):
            print(f"[{i}] {tool['tool_name']}")
            print(f"    ID: {tool['tool_id']}, Score: {tool.get('bm25_score', 0):.4f}")
        
        # Dense Results
        print(f"\n{'Dense Results:':^80}")
        print(f"{'-'*80}")
        for i, tool in enumerate(results['dense'], 1):
            print(f"[{i}] {tool['tool_name']}")
            print(f"    ID: {tool['tool_id']}, Score: {tool.get('similarity_score', 0):.4f}")
        
        # Hybrid RRF Results
        print(f"\n{'Hybrid RRF Results:':^80}")
        print(f"{'-'*80}")
        for i, tool in enumerate(results['hybrid'], 1):
            print(f"[{i}] {tool['tool_name']}")
            print(f"    ID: {tool['tool_id']}, RRF: {tool.get('rrf_score', 0):.4f}")
            print(f"    BM25 Rank: {tool.get('bm25_rank', 'N/A')}, "
                  f"Dense Rank: {tool.get('dense_rank', 'N/A')}")
        
        print(f"\n{'='*80}\n")
    
    def get_server_from_tool(self, tool: Dict[str, Any]) -> str:
        """
        Extract server name from tool.
        
        Args:
            tool: Tool dictionary
            
        Returns:
            Server name
        """
        return tool.get('server', 'Unknown')
    
    def analyze_fusion_agreement(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze agreement between BM25 and Dense retrieval.
        
        Args:
            query: User query string
            top_k: Number of results to analyze
            
        Returns:
            Analysis dictionary with overlap metrics
        """
        results = self.retrieve_separate(query, top_k=top_k)
        
        bm25_ids = set(tool['tool_id'] for tool in results['bm25'])
        dense_ids = set(tool['tool_id'] for tool in results['dense'])
        
        overlap = bm25_ids & dense_ids
        overlap_ratio = len(overlap) / top_k if top_k > 0 else 0
        
        return {
            'query': query,
            'top_k': top_k,
            'bm25_ids': list(bm25_ids),
            'dense_ids': list(dense_ids),
            'overlap_ids': list(overlap),
            'overlap_count': len(overlap),
            'overlap_ratio': overlap_ratio,
            'bm25_only': list(bm25_ids - dense_ids),
            'dense_only': list(dense_ids - bm25_ids)
        }


if __name__ == "__main__":
    # Example usage
    print("Testing HybridRetriever with Reciprocal Rank Fusion...")
    
    # Initialize hybrid retriever
    retriever = HybridRetriever(
        bm25_index_path="data/indexes/bm25_index.pkl",
        dense_index_path="data/indexes/tools_all-MiniLM-L6-v2.index",
        dense_metadata_path="data/indexes/tools_all-MiniLM-L6-v2.metadata.json",
        model_name='all-MiniLM-L6-v2',
        rrf_k=60
    )
    
    # Test queries
    test_queries = [
        "get weather information",
        "query database",
        "read file contents",
        "execute SQL query"
    ]
    
    for query in test_queries:
        # Compare all three methods
        retriever.compare_retrievers(query, top_k=5)
        
        # Analyze agreement
        agreement = retriever.analyze_fusion_agreement(query, top_k=5)
        print(f"Overlap Analysis:")
        print(f"  Overlap: {agreement['overlap_count']}/{agreement['top_k']} "
              f"({agreement['overlap_ratio']:.1%})")
        print(f"  BM25 only: {agreement['bm25_only']}")
        print(f"  Dense only: {agreement['dense_only']}")
        print()
    
    # Test top-1 retrieval
    print("\nTop-1 Hybrid Retrieval:")
    for query in test_queries[:2]:
        top1 = retriever.retrieve_top1(query)
        print(f"\nQuery: '{query}'")
        print(f"  Result: {top1['tool_name']} (RRF: {top1['rrf_score']:.4f})")
        print(f"  BM25 Rank: {top1.get('bm25_rank', 'N/A')}, "
              f"Dense Rank: {top1.get('dense_rank', 'N/A')}")
