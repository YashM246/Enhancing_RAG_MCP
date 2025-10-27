"""
Tool Retriever Module
Handles query encoding and top-k retrieval from FAISS index.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class ToolRetriever:
    """
    Retrieves top-k most relevant tools for a given query using FAISS similarity search.

    Supports multiple retrieval strategies:
    - Cosine similarity (via L2 on normalized vectors)
    - Configurable k values
    - Query preprocessing
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        index: Optional[faiss.Index] = None,
        tool_metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize the ToolRetriever.

        Args:
            model_name: Name of the sentence-transformer model (must match indexer)
            index: Pre-loaded FAISS index (optional)
            tool_metadata: List of tool dictionaries corresponding to index (optional)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = index
        self.tool_metadata = tool_metadata or []

        # Model-specific query prefix (e.g., E5 models require 'query: ' prefix)
        self.query_prefix = self._get_query_prefix(model_name)

        print(f"Initialized ToolRetriever with model: {model_name}")

    def _get_query_prefix(self, model_name: str) -> str:
        """
        Get the appropriate query prefix for the embedding model.

        Args:
            model_name: Name of the embedding model

        Returns:
            Query prefix string
        """
        # E5 models require 'query: ' prefix for queries
        if 'e5' in model_name.lower():
            return 'query: '
        return ''

    def load_index(self, index_path: str, metadata_path: str) -> None:
        """
        Load FAISS index and metadata from disk.

        Args:
            index_path: Path to the FAISS index file
            metadata_path: Path to metadata JSON file
        """
        import json
        from pathlib import Path

        # Load FAISS index
        self.index = faiss.read_index(index_path)
        print(f"Loaded FAISS index from: {index_path}")

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.tool_metadata = metadata['tools']
        print(f"Loaded metadata: {len(self.tool_metadata)} tools")

        # Verify model compatibility
        if metadata.get('model_name') != self.model_name:
            print(f"Warning: Index was built with {metadata.get('model_name')}, "
                  f"but retriever is using {self.model_name}")

    def set_index(self, index: faiss.Index, tool_metadata: List[Dict[str, Any]]) -> None:
        """
        Set the FAISS index and metadata directly.

        Args:
            index: FAISS index
            tool_metadata: List of tool dictionaries
        """
        self.index = index
        self.tool_metadata = tool_metadata
        print(f"Index set with {len(tool_metadata)} tools")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query into an embedding vector.

        Args:
            query: Query string

        Returns:
            Normalized embedding vector
        """
        # Add model-specific prefix if needed
        if self.query_prefix:
            query = self.query_prefix + query

        # Generate embedding
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Normalize for cosine similarity
        embedding = embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(embedding)

        return embedding

    def retrieve(
        self,
        query: str,
        k: int = 5,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant tools for a query.

        Args:
            query: Query string
            k: Number of tools to retrieve
            return_scores: Whether to include similarity scores in results

        Returns:
            List of tool dictionaries with optional scores
        """
        if self.index is None:
            raise ValueError("No index loaded. Load an index first using load_index() or set_index()")

        if not self.tool_metadata:
            raise ValueError("No tool metadata available")

        # Limit k to available tools
        k = min(k, len(self.tool_metadata))

        # Encode query
        query_embedding = self.encode_query(query)

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)

        # Convert distances to similarity scores (for L2 on normalized vectors)
        # L2 distance = 2 * (1 - cosine_similarity)
        # So: cosine_similarity = 1 - (L2_distance / 2)
        similarity_scores = 1.0 - (distances[0] / 2.0)

        # Retrieve tools
        results = []
        for idx, (tool_idx, score) in enumerate(zip(indices[0], similarity_scores)):
            tool = self.tool_metadata[tool_idx].copy()
            tool['rank'] = idx + 1
            tool['index_position'] = int(tool_idx)

            if return_scores:
                tool['similarity_score'] = float(score)

            results.append(tool)

        return results

    def batch_retrieve(
        self,
        queries: List[str],
        k: int = 5,
        return_scores: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve top-k tools for multiple queries.

        Args:
            queries: List of query strings
            k: Number of tools to retrieve per query
            return_scores: Whether to include similarity scores

        Returns:
            List of lists, where each inner list contains retrieved tools for one query
        """
        results = []
        for query in queries:
            query_results = self.retrieve(query, k=k, return_scores=return_scores)
            results.append(query_results)

        return results

    def retrieve_with_threshold(
        self,
        query: str,
        similarity_threshold: float = 0.5,
        max_k: int = 10,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve tools above a similarity threshold.

        Args:
            query: Query string
            similarity_threshold: Minimum similarity score (0-1)
            max_k: Maximum number of tools to consider
            return_scores: Whether to include similarity scores

        Returns:
            List of tools with similarity >= threshold
        """
        # Retrieve max_k tools
        results = self.retrieve(query, k=max_k, return_scores=True)

        # Filter by threshold
        filtered_results = [
            tool for tool in results
            if tool.get('similarity_score', 0) >= similarity_threshold
        ]

        # Remove scores if not requested
        if not return_scores:
            for tool in filtered_results:
                tool.pop('similarity_score', None)

        return filtered_results

    def get_tool_by_id(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool by its tool_id.

        Args:
            tool_id: Tool identifier

        Returns:
            Tool dictionary or None if not found
        """
        for tool in self.tool_metadata:
            if tool.get('tool_id') == tool_id:
                return tool.copy()
        return None

    def format_tools_for_llm(
        self,
        tools: List[Dict[str, Any]],
        include_rank: bool = True,
        include_score: bool = False
    ) -> str:
        """
        Format retrieved tools for LLM consumption.

        Args:
            tools: List of tool dictionaries
            include_rank: Whether to include rank numbers
            include_score: Whether to include similarity scores

        Returns:
            Formatted string representation of tools
        """
        if not tools:
            return "No tools available."

        formatted_parts = []
        for tool in tools:
            parts = []

            # Add rank
            if include_rank and 'rank' in tool:
                parts.append(f"[{tool['rank']}]")

            # Add tool name and ID
            parts.append(f"Tool: {tool.get('tool_name', 'Unknown')}")
            parts.append(f"(ID: {tool.get('tool_id', 'N/A')})")

            # Add score
            if include_score and 'similarity_score' in tool:
                parts.append(f"[Score: {tool['similarity_score']:.3f}]")

            header = " ".join(parts)

            # Add description
            description = tool.get('description', 'No description available.')

            # Add usage example
            usage = tool.get('usage_example', '')

            tool_text = f"{header}\n{description}"
            if usage:
                tool_text += f"\nExample: {usage}"

            formatted_parts.append(tool_text)

        return "\n\n".join(formatted_parts)

    def analyze_retrieval_quality(
        self,
        query: str,
        ground_truth_tool_id: str,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Analyze retrieval quality for a single query with ground truth.

        Args:
            query: Query string
            ground_truth_tool_id: ID of the correct tool
            k_values: List of k values to evaluate

        Returns:
            Dictionary with analysis results
        """
        results = {}

        for k in k_values:
            retrieved_tools = self.retrieve(query, k=k, return_scores=True)
            retrieved_ids = [tool['tool_id'] for tool in retrieved_tools]

            # Check if ground truth is in top-k
            is_retrieved = ground_truth_tool_id in retrieved_ids

            # Get rank if retrieved
            rank = None
            if is_retrieved:
                rank = retrieved_ids.index(ground_truth_tool_id) + 1

            results[f'recall@{k}'] = int(is_retrieved)
            results[f'rank@{k}'] = rank

        # Calculate reciprocal rank (for MRR)
        for k in sorted(k_values, reverse=True):
            rank = results[f'rank@{k}']
            if rank is not None:
                results['reciprocal_rank'] = 1.0 / rank
                break
        else:
            results['reciprocal_rank'] = 0.0

        # Store query info
        results['query'] = query
        results['ground_truth_tool_id'] = ground_truth_tool_id

        return results

    def __repr__(self) -> str:
        index_status = f"{len(self.tool_metadata)} tools" if self.tool_metadata else "No tools"
        return f"ToolRetriever(model={self.model_name}, status={index_status})"
