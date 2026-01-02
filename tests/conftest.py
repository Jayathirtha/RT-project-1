"""Shared fixtures and configuration for tests."""
import os
import sys
import logging
import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document

# Configure logging for tests (only show warnings and above by default)
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set random seed for test reproducibility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "GROQ_API_KEY": "test-groq-key",
        "GOOGLE_API_KEY": "test-google-key",
        "CHROMA_COLLECTION_NAME": "test_collection",
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="This is a test document about artificial intelligence.",
            metadata={"source": "test_doc1.txt", "page": 0}
        ),
        Document(
            page_content="Machine learning is a subset of AI that focuses on algorithms.",
            metadata={"source": "test_doc2.txt", "page": 0}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers.",
            metadata={"source": "test_doc3.txt", "page": 0}
        ),
    ]


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client."""
    client = MagicMock()
    collection = MagicMock()
    client.get_or_create_collection.return_value = collection
    return client, collection


@pytest.fixture
def mock_embedding_model():
    """Mock SentenceTransformer embedding model."""
    model = MagicMock()
    # Mock encode to return a simple embedding vector
    model.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5] * 20]  # 100-dim embedding
    return model


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    llm = MagicMock()
    # Mock the invoke method to return a simple response
    response = MagicMock()
    response.content = "This is a test response from the LLM."
    llm.invoke.return_value = response
    return llm

