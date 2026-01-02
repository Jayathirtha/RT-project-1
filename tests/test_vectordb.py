"""Tests for VectorDB class."""
import os
import sys
import logging
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from langchain_core.documents import Document

# Set up logger for tests
logger = logging.getLogger(__name__)

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from vectordb import VectorDB


class TestVectorDBInit:
    """Test VectorDB initialization."""

    @patch("vectordb.os.getenv")
    @patch("vectordb.chromadb.PersistentClient")
    @patch("vectordb.SentenceTransformer")
    def test_init_defaults(self, mock_transformer, mock_client_class, mock_getenv):
        """Test initialization with default parameters."""
        def getenv_side_effect(key, default=None):
            # Return None for env vars to use defaults
            return default
        mock_getenv.side_effect = getenv_side_effect
        
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        mock_transformer.return_value = MagicMock()

        db = VectorDB()

        assert db.collection_name == "rag_documents"
        assert db.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        mock_client_class.assert_called_once_with(path="./chroma_db")
        mock_transformer.assert_called_once_with(db.embedding_model_name)
        mock_client.get_or_create_collection.assert_called_once()

    @patch("vectordb.chromadb.PersistentClient")
    @patch("vectordb.SentenceTransformer")
    def test_init_custom_params(self, mock_transformer, mock_client_class, mock_env_vars):
        """Test initialization with custom parameters."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        mock_transformer.return_value = MagicMock()

        db = VectorDB(
            collection_name="custom_collection",
            embedding_model="custom-model"
        )

        assert db.collection_name == "custom_collection"
        assert db.embedding_model_name == "custom-model"


class TestChunkText:
    """Test text chunking functionality."""

    @patch("vectordb.chromadb.PersistentClient")
    @patch("vectordb.SentenceTransformer")
    def test_chunk_text_small(self, mock_transformer, mock_client_class, mock_env_vars):
        """Test chunking small text."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        mock_transformer.return_value = MagicMock()

        db = VectorDB()
        text = "This is a short text that should not be chunked."
        chunks = db.chunk_text(text, chunk_size=500)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)

    @patch("vectordb.chromadb.PersistentClient")
    @patch("vectordb.SentenceTransformer")
    def test_chunk_text_large(self, mock_transformer, mock_client_class, mock_env_vars):
        """Test chunking large text."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        mock_transformer.return_value = MagicMock()

        db = VectorDB()
        # Create a large text
        text = " ".join(["This is a sentence."] * 100)
        chunks = db.chunk_text(text, chunk_size=100)

        assert isinstance(chunks, list)
        assert len(chunks) > 1  # Should be split into multiple chunks

    @patch("vectordb.chromadb.PersistentClient")
    @patch("vectordb.SentenceTransformer")
    def test_chunk_text_empty(self, mock_transformer, mock_client_class, mock_env_vars):
        """Test chunking empty text."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        mock_transformer.return_value = MagicMock()

        db = VectorDB()
        chunks = db.chunk_text("", chunk_size=500)

        assert isinstance(chunks, list)


class TestAddDocuments:
    """Test adding documents to vector database."""

    @patch("vectordb.chromadb.PersistentClient")
    @patch("vectordb.SentenceTransformer")
    def test_add_documents(self, mock_transformer, mock_client_class, mock_env_vars, sample_documents):
        """Test adding documents to the database."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        mock_model = MagicMock()
        # Mock encode to return a numpy array-like object with tolist method
        mock_array = MagicMock()
        mock_array.tolist.return_value = [[0.1] * 100, [0.2] * 100]
        mock_model.encode.return_value = mock_array
        mock_transformer.return_value = mock_model

        db = VectorDB()
        db.add_documents(sample_documents[:1])  # Add one document

        # Verify embedding model was called
        assert mock_model.encode.called
        # Verify collection.add was called
        assert mock_collection.add.called

    @patch("vectordb.chromadb.PersistentClient")
    @patch("vectordb.SentenceTransformer")
    def test_add_documents_empty_list(self, mock_transformer, mock_client_class, mock_env_vars):
        """Test adding empty list of documents."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        mock_transformer.return_value = MagicMock()

        db = VectorDB()
        db.add_documents([])

        # Should not raise an error
        assert True

    @patch("vectordb.chromadb.PersistentClient")
    @patch("vectordb.SentenceTransformer")
    def test_add_documents_exception_handling(self, mock_transformer, mock_client_class, mock_env_vars, sample_documents):
        """Test exception handling when adding documents."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        mock_model = MagicMock()
        # Mock encode to return a numpy array-like object with tolist method
        mock_array = MagicMock()
        mock_array.tolist.return_value = [[0.1] * 100]
        mock_model.encode.return_value = mock_array
        mock_transformer.return_value = mock_model
        
        # Make collection.add raise an exception
        mock_collection.add.side_effect = Exception("Storage error")

        db = VectorDB()
        # Should not raise, but handle the exception
        db.add_documents(sample_documents[:1])
        
        # Verify add was called despite the exception
        assert mock_collection.add.called


class TestSearch:
    """Test search functionality."""

    @patch("vectordb.chromadb.PersistentClient")
    @patch("vectordb.SentenceTransformer")
    def test_search_success(self, mock_transformer, mock_client_class, mock_env_vars):
        """Test successful search."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        mock_model = MagicMock()
        # Mock encode to return a numpy array-like object with tolist method
        mock_array = MagicMock()
        mock_array.tolist.return_value = [[0.1] * 100]
        mock_model.encode.return_value = mock_array
        mock_transformer.return_value = mock_model

        # Mock query results
        mock_collection.query.return_value = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
            "distances": [[0.5, 0.8]],
            "ids": [["id1", "id2"]]
        }

        db = VectorDB()
        results = db.search("test query", n_results=5)

        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results
        assert "ids" in results
        assert len(results["documents"]) == 2
        mock_model.encode.assert_called_once()
        mock_collection.query.assert_called_once()

    @patch("vectordb.chromadb.PersistentClient")
    @patch("vectordb.SentenceTransformer")
    def test_search_empty_results(self, mock_transformer, mock_client_class, mock_env_vars):
        """Test search with empty results."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        mock_model = MagicMock()
        # Mock encode to return a numpy array-like object with tolist method
        mock_array = MagicMock()
        mock_array.tolist.return_value = [[0.1] * 100]
        mock_model.encode.return_value = mock_array
        mock_transformer.return_value = mock_model

        # Mock empty query results
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]]
        }

        db = VectorDB()
        results = db.search("test query", n_results=5)

        assert results["documents"] == []
        assert results["metadatas"] == []
        assert results["distances"] == []
        assert results["ids"] == []

    @patch("vectordb.chromadb.PersistentClient")
    @patch("vectordb.SentenceTransformer")
    def test_search_exception_handling(self, mock_transformer, mock_client_class, mock_env_vars):
        """Test exception handling during search."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        mock_model = MagicMock()
        # Mock encode to return a numpy array-like object with tolist method
        mock_array = MagicMock()
        mock_array.tolist.return_value = [[0.1] * 100]
        mock_model.encode.return_value = mock_array
        mock_transformer.return_value = mock_model

        # Make query raise an exception
        mock_collection.query.side_effect = Exception("Query error")

        db = VectorDB()
        results = db.search("test query", n_results=5)

        # Should return empty results on error
        assert results["documents"] == []
        assert results["metadatas"] == []
        assert results["distances"] == []
        assert results["ids"] == []

    @patch("vectordb.chromadb.PersistentClient")
    @patch("vectordb.SentenceTransformer")
    def test_search_custom_n_results(self, mock_transformer, mock_client_class, mock_env_vars):
        """Test search with custom n_results parameter."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client
        
        mock_model = MagicMock()
        # Mock encode to return a numpy array-like object with tolist method
        mock_array = MagicMock()
        mock_array.tolist.return_value = [[0.1] * 100]
        mock_model.encode.return_value = mock_array
        mock_transformer.return_value = mock_model

        mock_collection.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"source": "test1"}]],
            "distances": [[0.5]],
            "ids": [["id1"]]
        }

        db = VectorDB()
        db.search("test query", n_results=10)

        # Verify n_results parameter was passed
        call_args = mock_collection.query.call_args
        assert call_args is not None
        assert call_args.kwargs["n_results"] == 10

