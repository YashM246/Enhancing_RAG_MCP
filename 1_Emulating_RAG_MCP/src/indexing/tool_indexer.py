"""
Tool Indexer Module
Handles embedding generation and FAISS index creation for MCP tools.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class ToolIndexer:
    """
    Creates and manages FAISS indexes for MCP tool descriptions.

    Supports multiple embedding models:
    - all-MiniLM-L6-v2: Fast, 384-dim, baseline
    - intfloat/e5-base-v2: Semantic search optimized, 768-dim
    """

    # Supported embedding models
    SUPPORTED_MODELS = {
        'all-MiniLM-L6-v2': {
            'dimension': 384,
            'prefix': '',
            'description': 'Fast baseline model'
        },
        'intfloat/e5-base-v2': {
            'dimension': 768,
            'prefix': 'passage: ',  # E5 requires prefix for passages
            'description': 'Semantic search optimized'
        }
    }

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the ToolIndexer with a specific embedding model.

        Args:
            model_name: Name of the sentence-transformer model to use
        """
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Model {model_name} not supported. "
                f"Choose from: {list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.model_config = self.SUPPORTED_MODELS[model_name]
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.tool_metadata: List[Dict[str, Any]] = []

        print(f"Initialized ToolIndexer with model: {model_name}")
        print(f"Embedding dimension: {self.model_config['dimension']}")

    def _combine_tool_text(self, tool: Dict[str, Any]) -> str:
        """
        Combine tool fields into a single text representation for embedding.

        Args:
            tool: Tool dictionary with name, description, and usage_example

        Returns:
            Combined text string optimized for embedding
        """
        # Combine tool name, description, and usage example
        tool_name = tool.get('tool_name', '')
        description = tool.get('description', '')
        usage_example = tool.get('usage_example', '')

        # Create rich text representation
        combined = f"{tool_name}\n{description}\n{usage_example}"

        # Add prefix if required by model (e.g., E5 models)
        prefix = self.model_config['prefix']
        if prefix:
            combined = prefix + combined

        return combined

    def build_index(self, tools: List[Dict[str, Any]], batch_size: int = 32) -> None:
        """
        Build FAISS index from a list of tools.

        Args:
            tools: List of tool dictionaries
            batch_size: Number of tools to embed at once
        """
        if not tools:
            raise ValueError("Cannot build index from empty tool list")

        print(f"Building index for {len(tools)} tools...")

        # Store metadata
        self.tool_metadata = tools

        # Combine tool texts
        tool_texts = [self._combine_tool_text(tool) for tool in tools]

        # Generate embeddings in batches
        print(f"Generating embeddings (batch_size={batch_size})...")
        embeddings = self.model.encode(
            tool_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        # Normalize embeddings for cosine similarity (optional but recommended)
        faiss.normalize_L2(embeddings)

        # Create FAISS index (using L2 distance for normalized vectors = cosine similarity)
        dimension = self.model_config['dimension']
        self.index = faiss.IndexFlatL2(dimension)

        # Add vectors to index
        self.index.add(embeddings.astype('float32'))

        print(f"Index built successfully with {self.index.ntotal} vectors")

    def save_index(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Save FAISS index and metadata to disk.

        Args:
            index_path: Path to save the FAISS index (.index file)
            metadata_path: Path to save metadata (optional, auto-generated if None)
        """
        if self.index is None:
            raise ValueError("No index to save. Build an index first.")

        # Create directory if it doesn't exist
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        print(f"FAISS index saved to: {index_path}")

        # Save metadata
        if metadata_path is None:
            metadata_path = index_path.with_suffix('.metadata.json')

        metadata = {
            'model_name': self.model_name,
            'dimension': self.model_config['dimension'],
            'num_tools': len(self.tool_metadata),
            'tools': self.tool_metadata
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"Metadata saved to: {metadata_path}")

    def load_index(self, index_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Load FAISS index and metadata from disk.

        Args:
            index_path: Path to the FAISS index file
            metadata_path: Path to metadata file (optional, auto-detected if None)
        """
        index_path = Path(index_path)

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        print(f"FAISS index loaded from: {index_path}")

        # Load metadata
        if metadata_path is None:
            metadata_path = index_path.with_suffix('.metadata.json')

        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Verify model compatibility
        if metadata['model_name'] != self.model_name:
            print(f"Warning: Index was built with {metadata['model_name']}, "
                  f"but current model is {self.model_name}")

        self.tool_metadata = metadata['tools']
        print(f"Metadata loaded: {len(self.tool_metadata)} tools")

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension for the current model."""
        return self.model_config['dimension']

    def get_tool_by_index(self, idx: int) -> Dict[str, Any]:
        """
        Get tool metadata by index position.

        Args:
            idx: Index position in the FAISS index

        Returns:
            Tool dictionary
        """
        if idx < 0 or idx >= len(self.tool_metadata):
            raise IndexError(f"Index {idx} out of range [0, {len(self.tool_metadata)})")

        return self.tool_metadata[idx]

    def __repr__(self) -> str:
        index_status = f"{self.index.ntotal} vectors" if self.index else "No index"
        return (f"ToolIndexer(model={self.model_name}, "
                f"dim={self.model_config['dimension']}, "
                f"status={index_status})")
