"""
Unit tests for ToolIndexer class.
Tests embedding generation, FAISS indexing, and save/load functionality.

HOW TO RUN TESTS:
-----------------
# Run all tests with verbose output
pytest tests/test_tool_indexer.py -v

# Run all tests with detailed output and show print statements
pytest tests/test_tool_indexer.py -v -s

# Run with coverage report
pytest tests/test_tool_indexer.py --cov=src/indexing --cov-report=term-missing

# Run specific test class
pytest tests/test_tool_indexer.py::TestIndexBuilding -v

# Run specific test function
pytest tests/test_tool_indexer.py::TestIndexBuilding::test_build_index_basic -v

# Run tests and stop at first failure
pytest tests/test_tool_indexer.py -x

# Run tests in parallel (requires pytest-xdist)
pytest tests/test_tool_indexer.py -n auto

REQUIREMENTS:
-------------
pip install pytest pytest-cov numpy faiss-cpu sentence-transformers
"""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np
import faiss

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from indexing.tool_indexer import ToolIndexer


# Test fixtures
@pytest.fixture
def sample_tools():
    """Create sample tool data for testing."""
    return [
        {
            "tool_id": "test_search_001",
            "tool_name": "Test Web Search",
            "description": "A test web search tool for finding information online.",
            "parameters": {"query": {"type": "string", "required": True}},
            "usage_example": "Search for recent news about technology",
            "category": "web_search"
        },
        {
            "tool_id": "test_db_002",
            "tool_name": "Test Database Query",
            "description": "A test database query tool for retrieving data.",
            "parameters": {"query": {"type": "string", "required": True}},
            "usage_example": "Get all users from the database",
            "category": "database"
        },
        {
            "tool_id": "test_file_003",
            "tool_name": "Test File Reader",
            "description": "A test file reading tool for accessing files.",
            "parameters": {"path": {"type": "string", "required": True}},
            "usage_example": "Read the configuration file at /etc/config.json",
            "category": "file_operations"
        }
    ]


@pytest.fixture
def temp_index_dir():
    """Create temporary directory for index files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestInitialization:
    """Test ToolIndexer initialization with different models."""

    def test_init_with_minilm_model(self):
        """Test initialization with all-MiniLM-L6-v2 model."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')
        assert indexer.model_name == 'all-MiniLM-L6-v2'
        assert indexer.get_embedding_dim() == 384
        assert indexer.model is not None
        assert indexer.index is None  # No index built yet

    def test_init_with_e5_model(self):
        """Test initialization with intfloat/e5-base-v2 model."""
        indexer = ToolIndexer('intfloat/e5-base-v2')
        assert indexer.model_name == 'intfloat/e5-base-v2'
        assert indexer.get_embedding_dim() == 768
        assert indexer.model is not None
        assert indexer.index is None

    def test_init_with_invalid_model(self):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            ToolIndexer('invalid-model-name')

    def test_model_config_loaded(self):
        """Test that model configuration is properly loaded."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')
        assert 'dimension' in indexer.model_config
        assert 'prefix' in indexer.model_config
        assert 'description' in indexer.model_config


# ============================================================================
# TEXT COMBINATION TESTS
# ============================================================================

class TestTextCombination:
    """Test tool text combination functionality."""

    def test_combine_standard_tool(self, sample_tools):
        """Test combining tool fields into text."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')
        tool = sample_tools[0]

        combined = indexer._combine_tool_text(tool)

        # Should contain all key fields
        assert tool['tool_name'] in combined
        assert tool['description'] in combined
        assert tool['usage_example'] in combined

    def test_combine_with_e5_prefix(self, sample_tools):
        """Test that E5 model adds 'passage: ' prefix."""
        indexer = ToolIndexer('intfloat/e5-base-v2')
        tool = sample_tools[0]

        combined = indexer._combine_tool_text(tool)

        # E5 model should add prefix
        assert combined.startswith('passage: ')

    def test_combine_without_prefix(self, sample_tools):
        """Test that MiniLM model does not add prefix."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')
        tool = sample_tools[0]

        combined = indexer._combine_tool_text(tool)

        # MiniLM should not add prefix
        assert not combined.startswith('passage: ')

    def test_combine_with_missing_fields(self):
        """Test handling of tools with missing optional fields."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')
        tool = {"tool_name": "Test", "description": "Desc"}  # No usage_example

        combined = indexer._combine_tool_text(tool)

        # Should still work, just with empty string for missing field
        assert "Test" in combined
        assert "Desc" in combined

    def test_combine_with_special_characters(self):
        """Test handling of special characters in tool descriptions."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')
        tool = {
            "tool_name": "Test™ Tool®",
            "description": "Handles UTF-8: 你好, émojis: 🔍",
            "usage_example": "Search for <special> & \"quoted\" text"
        }

        combined = indexer._combine_tool_text(tool)

        # Should preserve all special characters
        assert "Test™ Tool®" in combined
        assert "你好" in combined
        assert "🔍" in combined


# ============================================================================
# INDEX BUILDING TESTS
# ============================================================================

class TestIndexBuilding:
    """Test FAISS index building functionality."""

    def test_build_index_basic(self, sample_tools):
        """Test basic index building with sample tools."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')
        indexer.build_index(sample_tools)

        # Verify index was created
        assert indexer.index is not None
        assert indexer.index.ntotal == len(sample_tools)
        assert indexer.index.d == 384  # Dimension matches model

    def test_build_index_with_e5_model(self, sample_tools):
        """Test index building with E5 model (768-dim)."""
        indexer = ToolIndexer('intfloat/e5-base-v2')
        indexer.build_index(sample_tools)

        assert indexer.index is not None
        assert indexer.index.ntotal == len(sample_tools)
        assert indexer.index.d == 768  # E5 dimension

    def test_build_index_stores_metadata(self, sample_tools):
        """Test that tool metadata is stored during index building."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')
        indexer.build_index(sample_tools)

        assert len(indexer.tool_metadata) == len(sample_tools)
        assert indexer.tool_metadata[0]['tool_id'] == sample_tools[0]['tool_id']

    def test_build_index_with_empty_list(self):
        """Test that building index with empty list raises error."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')

        with pytest.raises(ValueError, match="empty tool list"):
            indexer.build_index([])

    def test_build_index_with_batching(self, sample_tools):
        """Test index building with different batch sizes."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')
        indexer.build_index(sample_tools, batch_size=1)

        assert indexer.index.ntotal == len(sample_tools)

    def test_embeddings_are_normalized(self, sample_tools):
        """Test that embeddings are L2 normalized."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')
        indexer.build_index(sample_tools)

        # Reconstruct a vector from index
        vector = faiss.rev_swig_ptr(indexer.index.get_xb(), indexer.index.ntotal * indexer.index.d)
        vector = np.array(vector).reshape(indexer.index.ntotal, indexer.index.d)

        # Check that vectors are approximately normalized (L2 norm ≈ 1)
        norms = np.linalg.norm(vector, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(sample_tools)), decimal=5)


# ============================================================================
# SAVE/LOAD TESTS
# ============================================================================

class TestSaveLoad:
    """Test index and metadata persistence."""

    def test_save_index(self, sample_tools, temp_index_dir):
        """Test saving index to disk."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')
        indexer.build_index(sample_tools)

        index_path = temp_index_dir / "test.index"
        indexer.save_index(str(index_path))

        # Verify files exist
        assert index_path.exists()
        assert index_path.with_suffix('.metadata.json').exists()

    def test_save_creates_directory(self, sample_tools, temp_index_dir):
        """Test that save_index creates parent directories if needed."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')
        indexer.build_index(sample_tools)

        index_path = temp_index_dir / "subdir" / "nested" / "test.index"
        indexer.save_index(str(index_path))

        assert index_path.exists()

    def test_load_index(self, sample_tools, temp_index_dir):
        """Test loading index from disk."""
        # Build and save
        indexer1 = ToolIndexer('all-MiniLM-L6-v2')
        indexer1.build_index(sample_tools)
        index_path = temp_index_dir / "test.index"
        indexer1.save_index(str(index_path))

        # Load in new instance
        indexer2 = ToolIndexer('all-MiniLM-L6-v2')
        indexer2.load_index(str(index_path))

        # Verify loaded data
        assert indexer2.index is not None
        assert indexer2.index.ntotal == len(sample_tools)
        assert len(indexer2.tool_metadata) == len(sample_tools)

    def test_load_nonexistent_index(self, temp_index_dir):
        """Test that loading nonexistent index raises FileNotFoundError."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')

        with pytest.raises(FileNotFoundError):
            indexer.load_index(str(temp_index_dir / "nonexistent.index"))

    def test_metadata_integrity(self, sample_tools, temp_index_dir):
        """Test that metadata is preserved through save/load cycle."""
        indexer1 = ToolIndexer('all-MiniLM-L6-v2')
        indexer1.build_index(sample_tools)
        index_path = temp_index_dir / "test.index"
        indexer1.save_index(str(index_path))

        indexer2 = ToolIndexer('all-MiniLM-L6-v2')
        indexer2.load_index(str(index_path))

        # Compare metadata
        assert indexer2.tool_metadata == sample_tools
        for i, tool in enumerate(sample_tools):
            assert indexer2.get_tool_by_index(i)['tool_id'] == tool['tool_id']

    def test_load_with_different_model_warning(self, sample_tools, temp_index_dir, capsys):
        """Test warning when loading index built with different model."""
        # Build with MiniLM
        indexer1 = ToolIndexer('all-MiniLM-L6-v2')
        indexer1.build_index(sample_tools)
        index_path = temp_index_dir / "test.index"
        indexer1.save_index(str(index_path))

        # Load with E5 (different model)
        indexer2 = ToolIndexer('intfloat/e5-base-v2')
        indexer2.load_index(str(index_path))

        # Should print warning
        captured = capsys.readouterr()
        assert "Warning" in captured.out


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, sample_tools, temp_index_dir):
        """Test complete workflow: build → save → load → retrieve."""
        # Build index
        indexer1 = ToolIndexer('all-MiniLM-L6-v2')
        indexer1.build_index(sample_tools)

        # Save
        index_path = temp_index_dir / "workflow.index"
        indexer1.save_index(str(index_path))

        # Load in new instance
        indexer2 = ToolIndexer('all-MiniLM-L6-v2')
        indexer2.load_index(str(index_path))

        # Verify everything matches
        assert indexer2.index.ntotal == indexer1.index.ntotal
        assert len(indexer2.tool_metadata) == len(indexer1.tool_metadata)
        assert indexer2.get_embedding_dim() == indexer1.get_embedding_dim()

    def test_rebuild_index_multiple_times(self, sample_tools):
        """Test that index can be rebuilt multiple times."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')

        # Build first time
        indexer.build_index(sample_tools[:2])
        assert indexer.index.ntotal == 2

        # Rebuild with different tools
        indexer.build_index(sample_tools)
        assert indexer.index.ntotal == 3

    def test_save_without_building_raises_error(self, temp_index_dir):
        """Test that saving without building index raises error."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')

        with pytest.raises(ValueError, match="No index to save"):
            indexer.save_index(str(temp_index_dir / "test.index"))


# ============================================================================
# TOOL RETRIEVAL TESTS
# ============================================================================

class TestToolRetrieval:
    """Test tool metadata retrieval by index."""

    def test_get_tool_by_index(self, sample_tools):
        """Test retrieving tool by index position."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')
        indexer.build_index(sample_tools)

        tool = indexer.get_tool_by_index(0)
        assert tool['tool_id'] == sample_tools[0]['tool_id']

        tool = indexer.get_tool_by_index(2)
        assert tool['tool_id'] == sample_tools[2]['tool_id']

    def test_get_tool_invalid_index(self, sample_tools):
        """Test that invalid index raises IndexError."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')
        indexer.build_index(sample_tools)

        with pytest.raises(IndexError):
            indexer.get_tool_by_index(999)

        with pytest.raises(IndexError):
            indexer.get_tool_by_index(-10)


# ============================================================================
# EDGE CASES AND ROBUSTNESS TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_very_long_description(self):
        """Test handling of very long tool descriptions."""
        long_tool = {
            "tool_id": "long_001",
            "tool_name": "Long Description Tool",
            "description": "Lorem ipsum " * 1000,  # Very long description
            "usage_example": "Use this tool"
        }

        indexer = ToolIndexer('all-MiniLM-L6-v2')
        indexer.build_index([long_tool])

        assert indexer.index.ntotal == 1

    def test_unicode_handling(self):
        """Test proper Unicode/UTF-8 handling."""
        unicode_tool = {
            "tool_id": "unicode_001",
            "tool_name": "Unicode Tool 🌍",
            "description": "Supports 中文, العربية, עברית, and emoji 🚀",
            "usage_example": "Process internationalized content"
        }

        indexer = ToolIndexer('all-MiniLM-L6-v2')
        indexer.build_index([unicode_tool])

        retrieved = indexer.get_tool_by_index(0)
        assert "🌍" in retrieved['tool_name']
        assert "🚀" in retrieved['description']

    def test_repr_method(self, sample_tools):
        """Test string representation of indexer."""
        indexer = ToolIndexer('all-MiniLM-L6-v2')

        # Before building index
        repr_str = repr(indexer)
        assert "all-MiniLM-L6-v2" in repr_str
        assert "No index" in repr_str

        # After building index
        indexer.build_index(sample_tools)
        repr_str = repr(indexer)
        assert "3 vectors" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
