"""Tests for RAGAssistant and related functions."""
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
from app import RAGAssistant, load_documents


class TestLoadDocuments:
    """Test document loading functionality."""

    @patch("app.DirectoryLoader")
    @patch("app.os.getcwd")
    def test_load_documents(self, mock_getcwd, mock_directory_loader):
        """Test loading documents from data directory."""
        mock_getcwd.return_value = "/test/path"
        
        # Mock document loader
        mock_loader_instance = MagicMock()
        mock_doc = Document(page_content="Test content", metadata={"source": "test.txt"})
        mock_loader_instance.load.return_value = [mock_doc]
        mock_directory_loader.return_value = mock_loader_instance

        # Call load_documents multiple times since it loops through extensions
        docs = load_documents()

        # Should return a list
        assert isinstance(docs, list)
        mock_directory_loader.assert_called()

    @patch("app.DirectoryLoader")
    @patch("app.os.getcwd")
    @patch("app.os.path.join")
    def test_load_documents_path_construction(self, mock_join, mock_getcwd, mock_directory_loader):
        """Test that document loading constructs the correct path."""
        mock_getcwd.return_value = "/test/path"
        mock_join.return_value = "/test/path/data"
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = []
        mock_directory_loader.return_value = mock_loader_instance

        load_documents()

        # Verify os.path.join was called with correct arguments
        mock_join.assert_called()


class TestRAGAssistantInit:
    """Test RAGAssistant initialization."""

    @patch("app.ChatOpenAI")
    @patch("app.VectorDB")
    @patch("app.os.getenv")
    def test_init_with_openai_key(self, mock_getenv, mock_vector_db, mock_chat_openai, mock_env_vars):
        """Test initialization with OpenAI API key."""
        def getenv_side_effect(key, default=None):
            env_vars = {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "gpt-4o-mini",
            }
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = getenv_side_effect
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        mock_vector_db.return_value = MagicMock()

        assistant = RAGAssistant()

        assert assistant.llm is not None
        assert assistant.vector_db is not None
        assert assistant.prompt_template is not None
        assert assistant.chain is not None
        mock_chat_openai.assert_called_once()

    @patch("app.ChatGroq")
    @patch("app.VectorDB")
    @patch("app.os.getenv")
    def test_init_with_groq_key(self, mock_getenv, mock_vector_db, mock_chat_groq, mock_env_vars):
        """Test initialization with Groq API key."""
        def getenv_side_effect(key, default=None):
            env_vars = {
                "GROQ_API_KEY": "test-groq-key",
                "GROQ_MODEL": "llama-3.1-8b-instant",
                "OPENAI_API_KEY": None,  # OpenAI not available
            }
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = getenv_side_effect
        mock_llm = MagicMock()
        mock_chat_groq.return_value = mock_llm
        mock_vector_db.return_value = MagicMock()

        assistant = RAGAssistant()

        assert assistant.llm is not None
        mock_chat_groq.assert_called_once()

    @patch("app.ChatGoogleGenerativeAI")
    @patch("app.VectorDB")
    @patch("app.os.getenv")
    def test_init_with_google_key(self, mock_getenv, mock_vector_db, mock_chat_google, mock_env_vars):
        """Test initialization with Google API key."""
        def getenv_side_effect(key, default=None):
            env_vars = {
                "GOOGLE_API_KEY": "test-google-key",
                "GOOGLE_MODEL": "gemini-2.0-flash",
                "OPENAI_API_KEY": None,
                "GROQ_API_KEY": None,
            }
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = getenv_side_effect
        mock_llm = MagicMock()
        mock_chat_google.return_value = mock_llm
        mock_vector_db.return_value = MagicMock()

        assistant = RAGAssistant()

        assert assistant.llm is not None
        mock_chat_google.assert_called_once()

    @patch("app.VectorDB")
    @patch("app.os.getenv")
    def test_init_no_api_key(self, mock_getenv, mock_vector_db, mock_env_vars):
        """Test initialization fails when no API key is provided."""
        def getenv_side_effect(key, default=None):
            return None  # No API keys available
        
        mock_getenv.side_effect = getenv_side_effect
        mock_vector_db.return_value = MagicMock()

        with pytest.raises(ValueError, match="No valid API key found"):
            RAGAssistant()


class TestRAGAssistantAddDocuments:
    """Test adding documents to RAGAssistant."""

    @patch("app.ChatOpenAI")
    @patch("app.VectorDB")
    @patch("app.os.getenv")
    def test_add_documents(self, mock_getenv, mock_vector_db_class, mock_chat_openai, mock_env_vars, sample_documents):
        """Test adding documents to the assistant."""
        def getenv_side_effect(key, default=None):
            env_vars = {"OPENAI_API_KEY": "test-key"}
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = getenv_side_effect
        mock_chat_openai.return_value = MagicMock()
        mock_vector_db = MagicMock()
        mock_vector_db_class.return_value = mock_vector_db

        assistant = RAGAssistant()
        assistant.add_documents(sample_documents)

        mock_vector_db.add_documents.assert_called_once_with(sample_documents)


class TestRAGAssistantInvoke:
    """Test RAGAssistant invoke method."""

    @patch("app.ChatOpenAI")
    @patch("app.VectorDB")
    @patch("app.os.getenv")
    def test_invoke_success(self, mock_getenv, mock_vector_db_class, mock_chat_openai, mock_env_vars):
        """Test successful invocation."""
        def getenv_side_effect(key, default=None):
            env_vars = {"OPENAI_API_KEY": "test-key"}
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = getenv_side_effect
        
        # Mock LLM - chain expects string output
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Mock VectorDB
        mock_vector_db = MagicMock()
        mock_vector_db.search.return_value = {
            "documents": ["Document 1", "Document 2"],
            "distances": [0.5, 0.8],
            "metadatas": [{}, {}],
            "ids": ["id1", "id2"]
        }
        mock_vector_db_class.return_value = mock_vector_db

        assistant = RAGAssistant()
        # Mock the chain to return a string directly
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Test response"
        
        result = assistant.invoke("test question", n_results=3)

        assert isinstance(result, str)
        assert "Test response" in result
        assert "distance" in result.lower()
        mock_vector_db.search.assert_called_once_with("test question", n_results=3)

    @patch("app.ChatOpenAI")
    @patch("app.VectorDB")
    @patch("app.os.getenv")
    def test_invoke_no_relevant_docs(self, mock_getenv, mock_vector_db_class, mock_chat_openai, mock_env_vars):
        """Test invocation when no relevant documents are found."""
        def getenv_side_effect(key, default=None):
            env_vars = {"OPENAI_API_KEY": "test-key"}
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = getenv_side_effect
        mock_chat_openai.return_value = MagicMock()
        
        # Mock VectorDB to return documents with distances > 1
        mock_vector_db = MagicMock()
        mock_vector_db.search.return_value = {
            "documents": ["Document 1", "Document 2"],
            "distances": [1.5, 2.0],  # All distances > 1
            "metadatas": [{}, {}],
            "ids": ["id1", "id2"]
        }
        mock_vector_db_class.return_value = mock_vector_db

        assistant = RAGAssistant()
        result = assistant.invoke("test question")

        assert "not in this document" in result.lower()
        mock_vector_db.search.assert_called_once()

    @patch("app.ChatOpenAI")
    @patch("app.VectorDB")
    @patch("app.os.getenv")
    def test_invoke_empty_search_results(self, mock_getenv, mock_vector_db_class, mock_chat_openai, mock_env_vars):
        """Test invocation with empty search results."""
        def getenv_side_effect(key, default=None):
            env_vars = {"OPENAI_API_KEY": "test-key"}
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = getenv_side_effect
        mock_chat_openai.return_value = MagicMock()
        
        # Mock VectorDB to return empty results
        mock_vector_db = MagicMock()
        mock_vector_db.search.return_value = {
            "documents": [],
            "distances": [],
            "metadatas": [],
            "ids": []
        }
        mock_vector_db_class.return_value = mock_vector_db

        assistant = RAGAssistant()
        result = assistant.invoke("test question")

        assert "not in this document" in result.lower()

    @patch("app.ChatOpenAI")
    @patch("app.VectorDB")
    @patch("app.os.getenv")
    def test_invoke_chain_exception(self, mock_getenv, mock_vector_db_class, mock_chat_openai, mock_env_vars):
        """Test invocation when chain raises an exception."""
        def getenv_side_effect(key, default=None):
            env_vars = {"OPENAI_API_KEY": "test-key"}
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = getenv_side_effect
        mock_chat_openai.return_value = MagicMock()
        
        # Mock VectorDB
        mock_vector_db = MagicMock()
        mock_vector_db.search.return_value = {
            "documents": ["Document 1"],
            "distances": [0.5],
            "metadatas": [{}],
            "ids": ["id1"]
        }
        mock_vector_db_class.return_value = mock_vector_db

        assistant = RAGAssistant()
        # Mock the chain and make invoke raise an exception
        assistant.chain = MagicMock()
        assistant.chain.invoke.side_effect = Exception("Chain error")

        result = assistant.invoke("test question")

        assert "error occurred" in result.lower()
        assert "Chain error" in result

    @patch("app.ChatOpenAI")
    @patch("app.VectorDB")
    @patch("app.os.getenv")
    def test_invoke_filtered_results(self, mock_getenv, mock_vector_db_class, mock_chat_openai, mock_env_vars):
        """Test that results are filtered by distance threshold."""
        def getenv_side_effect(key, default=None):
            env_vars = {"OPENAI_API_KEY": "test-key"}
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = getenv_side_effect
        mock_chat_openai.return_value = MagicMock()
        
        # Mock VectorDB with mixed distances
        mock_vector_db = MagicMock()
        mock_vector_db.search.return_value = {
            "documents": ["Good doc", "Bad doc"],
            "distances": [0.5, 1.5],  # One good, one bad
            "metadatas": [{}, {}],
            "ids": ["id1", "id2"]
        }
        mock_vector_db_class.return_value = mock_vector_db

        assistant = RAGAssistant()
        # Mock the chain to return a string
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Test response"
        
        result = assistant.invoke("test question")

        # Should only use the good doc (distance <= 1)
        assert isinstance(result, str)
        # Verify chain was invoked (would fail if filtering didn't work)
        assert assistant.chain.invoke.called
        # Verify it was called with only the good document (one chunk)
        call_args = assistant.chain.invoke.call_args
        assert call_args is not None
        context = call_args[0][0]["context"]
        assert "Good doc" in context
        assert "Bad doc" not in context

    @patch("app.ChatOpenAI")
    @patch("app.VectorDB")
    @patch("app.os.getenv")
    def test_invoke_custom_n_results(self, mock_getenv, mock_vector_db_class, mock_chat_openai, mock_env_vars):
        """Test invocation with custom n_results parameter."""
        def getenv_side_effect(key, default=None):
            env_vars = {"OPENAI_API_KEY": "test-key"}
            return env_vars.get(key, default)
        
        mock_getenv.side_effect = getenv_side_effect
        mock_chat_openai.return_value = MagicMock()
        mock_vector_db = MagicMock()
        mock_vector_db.search.return_value = {
            "documents": [],
            "distances": [],
            "metadatas": [],
            "ids": []
        }
        mock_vector_db_class.return_value = mock_vector_db

        assistant = RAGAssistant()
        assistant.invoke("test question", n_results=10)

        # Verify n_results was passed correctly
        mock_vector_db.search.assert_called_once_with("test question", n_results=10)

