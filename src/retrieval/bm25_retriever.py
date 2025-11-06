from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.indexing.bm25_indexer import BM25Indexer

class BM25Retriever:

    def __init__(self, index_path: str= None):
        self.indexer = BM25Indexer()

        if index_path:
            self.load_index(index_path)


    def load_index(self, index_path:str)-> None:

        self.indexer.load_index(index_path)
        print(f"BM25Retriever ready with {len(self.indexer.tools)} tools")


    def retrieve(self, query:str, top_k:int= 5)-> List[Dict[str, Any]]:
        if self.indexer.bm25 is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Tokenize query
        tokenized_query = self.indexer.tokenize(query)

        if not tokenized_query:
            raise ValueError("Query produced no valid tokens")
        
        # Get BM25 scores
        scores = self.indexer.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build Results
        results = []
        for idx in top_indices:
            tool = self.indexer.tools[idx].copy()
            tool["bm25_score"] = float(scores[idx])
            tool["rank"] = len(results) + 1
            results.append(tool)
        
        return results
    

    def retrieve_top1(self, query: str) -> Dict[str, Any]:
        """
        Retrieve only the top-1 tool (for Approach 2: BM25 Only).
        
        Args:
            query: User query string
            
        Returns:
            Single tool dictionary with score
        """
        results = self.retrieve(query, top_k=1)
        return results[0] if results else None
    
    def get_scores(self, query: str) -> List[Tuple[str, float]]:
        """
        Get BM25 scores for all tools.
        
        Args:
            query: User query string
            
        Returns:
            List of (tool_id, score) tuples
        """
        if self.indexer.bm25 is None:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        tokenized_query = self.indexer.tokenize(query)
        scores = self.indexer.bm25.get_scores(tokenized_query)
        
        return [(self.indexer.tool_ids[i], float(scores[i])) 
                for i in range(len(scores))]


if __name__ == "__main__":
    # Example usage
    print("Testing BM25Retriever...")
    
    # Load the index we created earlier
    retriever = BM25Retriever(index_path="data/indexes/bm25_index.pkl")
    
    # Test queries
    test_queries = [
        "get weather information",
        "query database",
        "read file contents",
        "execute SQL query"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        print(f"{'='*60}")
        
        # Get top-3 results
        results = retriever.retrieve(query, top_k=3)
        
        for i, tool in enumerate(results, 1):
            print(f"\nRank {i}: {tool['tool_name']} (Score: {tool['bm25_score']:.4f})")
            print(f"  ID: {tool['tool_id']}")
            print(f"  Description: {tool['description']}")
        
        # Test top-1 retrieval
        print(f"\nTop-1 only:")
        top1 = retriever.retrieve_top1(query)
        print(f"  {top1['tool_name']} (Score: {top1['bm25_score']:.4f})")