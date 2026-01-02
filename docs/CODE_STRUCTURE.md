# Code Structure Documentation

This document provides a detailed overview of the codebase structure, architecture, and design decisions.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Module Documentation](#module-documentation)
3. [Architecture Overview](#architecture-overview)
4. [Data Flow](#data-flow)
5. [Design Patterns](#design-patterns)
6. [Testing Strategy](#testing-strategy)

## Project Structure

```
RT-project-1/
├── src/                    # Source code
│   ├── __init__.py        # Package marker (optional)
│   ├── app.py             # Main RAG assistant implementation
│   ├── vectordb.py        # Vector database wrapper
│   └── utils.py           # Utility functions
│
├── tests/                  # Test suite
│   ├── __init__.py        # Test package marker
│   ├── conftest.py        # Shared fixtures and configuration
│   ├── test_app.py        # RAG assistant tests
│   └── test_vectordb.py   # Vector database tests
│
├── notebooks/              # Jupyter notebooks
│   └── RAG_Demo.ipynb     # Code structure demo
│
├── data/                   # Document storage
├── chroma_db/              # ChromaDB persistent storage
├── requirements.txt        # Python dependencies
├── pytest.ini             # Pytest configuration
└── README.md              # Project documentation
```

## Module Documentation

### 1. `src/utils.py`

**Purpose**: Utility functions for project-wide use.

**Functions**:
- `set_random_seed(seed: int = None)`: Sets random seeds for reproducibility

**Key Features**:
- Sets seeds for Python random, NumPy, and PyTorch
- Reads seed from `RANDOM_SEED` environment variable
- Default seed: 42
- Gracefully handles missing dependencies

**Usage**:
```python
from utils import set_random_seed
set_random_seed(42)  # Or set RANDOM_SEED env var
```

### 2. `src/vectordb.py`

**Purpose**: Vector database operations using ChromaDB.

**Class**: `VectorDB`

**Methods**:
- `__init__(collection_name=None, embedding_model=None)`: Initialize vector DB
- `chunk_text(text: str, chunk_size: int = 500) -> List[str]`: Chunk text into segments
- `add_documents(documents: List[Document]) -> None`: Add documents to vector store
- `search(query: str, n_results: int = 5) -> Dict[str, Any]`: Search for similar documents

**Key Features**:
- Uses RecursiveCharacterTextSplitter for intelligent chunking
- Sentence transformer embeddings (configurable model)
- ChromaDB for persistent storage
- Comprehensive error handling and logging

**Dependencies**:
- `chromadb`: Vector database
- `sentence-transformers`: Embedding generation
- `langchain-text-splitters`: Text chunking

### 3. `src/app.py`

**Purpose**: Main RAG assistant implementation.

**Class**: `RAGAssistant`

**Methods**:
- `__init__()`: Initialize RAG assistant
- `_initialize_llm()`: Initialize LLM (OpenAI, Groq, or Google)
- `add_documents(documents: List[Document]) -> None`: Add documents to knowledge base
- `invoke(input: str, n_results: int = 3) -> str`: Query the assistant

**Functions**:
- `load_documents() -> List[Document]`: Load documents from data directory
- `main()`: Entry point for command-line usage

**Key Features**:
- Multi-LLM support (OpenAI, Groq, Google Gemini)
- Automatic API key detection
- Distance-based result filtering
- Comprehensive prompt engineering
- Error handling and logging

**Dependencies**:
- `langchain`: LLM framework
- `langchain-openai`, `langchain-groq`, `langchain-google-genai`: LLM providers
- `vectordb`: Vector database operations

## Architecture Overview

### RAG Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     Document Ingestion                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │  PDF    │  │   TXT   │  │   CSV   │  │  JSON   │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│       │            │            │            │              │
│       └────────────┴────────────┴────────────┘              │
│                        │                                     │
│                        ▼                                     │
│              ┌──────────────────┐                            │
│              │  Document Loader │                            │
│              └──────────────────┘                            │
└────────────────────────│─────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Document Processing                        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Text Chunking                            │    │
│  │  (RecursiveCharacterTextSplitter)                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Embedding Generation                     │    │
│  │  (Sentence Transformers)                            │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Vector Storage                           │    │
│  │  (ChromaDB)                                         │    │
│  └─────────────────────────────────────────────────────┘    │
└────────────────────────│─────────────────────────────────────┘
                         │
                         │
┌────────────────────────┴─────────────────────────────────────┐
│                        Query Flow                             │
│                                                               │
│  User Query                                                   │
│      │                                                        │
│      ▼                                                        │
│  ┌──────────────┐                                            │
│  │ Query Embed │                                            │
│  └──────────────┘                                            │
│      │                                                        │
│      ▼                                                        │
│  ┌──────────────────┐                                        │
│  │ Similarity Search│◄──────────────────┐                   │
│  └──────────────────┘                   │                   │
│      │                                  │                   │
│      ▼                                  │                   │
│  ┌──────────────────┐                  │                   │
│  │ Distance Filter  │                  │                   │
│  │  (threshold ≤ 1) │                  │                   │
│  └──────────────────┘                  │                   │
│      │                                  │                   │
│      ▼                                  │                   │
│  ┌──────────────────┐                  │                   │
│  │ Context Assembly │                  │                   │
│  └──────────────────┘                  │                   │
│      │                                  │                   │
│      ▼                                  │                   │
│  ┌──────────────────┐                  │                   │
│  │  LLM Generation  │                  │                   │
│  │  (OpenAI/Groq/   │                  │                   │
│  │   Google)        │                  │                   │
│  └──────────────────┘                  │                   │
│      │                                  │                   │
│      ▼                                  │                   │
│  User Response                          │                   │
│                                         │                   │
│                                         │                   │
│                                         │                   │
└─────────────────────────────────────────┘                   │
         │                                                     │
         └─────────────────────────────────────────────────────┘
                          Vector Store
```

## Data Flow

### Document Ingestion Flow

1. **Document Loading** (`load_documents()`)
   - Scans `data/` directory for supported file types
   - Uses appropriate loader (PyPDFLoader, TextLoader, etc.)
   - Returns list of Document objects with metadata

2. **Document Processing** (`VectorDB.add_documents()`)
   - Chunks each document using RecursiveCharacterTextSplitter
   - Generates embeddings using sentence transformers
   - Stores chunks with metadata in ChromaDB

### Query Flow

1. **Query Processing** (`RAGAssistant.invoke()`)
   - Embeds user query using same embedding model
   - Searches vector database for similar chunks
   - Filters results by distance threshold (≤ 1.0)
   - Assembles context from filtered chunks

2. **Response Generation**
   - Formats context and question in prompt template
   - Invokes LLM with formatted prompt
   - Returns formatted response with metadata

## Design Patterns

### 1. Modular Design

- **Separation of Concerns**: VectorDB handles storage, RAGAssistant handles orchestration
- **Single Responsibility**: Each module has a clear, focused purpose
- **Dependency Injection**: Components are loosely coupled

### 2. Factory Pattern

- LLM initialization uses a factory-like pattern
- Automatically selects LLM provider based on available API keys
- Consistent interface across different providers

### 3. Strategy Pattern

- Different document loaders for different file types
- Configurable embedding models
- Pluggable LLM providers

### 4. Error Handling

- Try-except blocks with logging
- Graceful degradation (empty results on error)
- User-friendly error messages

## Testing Strategy

### Test Organization

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Mocking**: External dependencies are mocked

### Test Coverage

- **VectorDB Tests**: 13 tests covering all methods
- **RAGAssistant Tests**: 12 tests covering initialization and operations
- **Fixtures**: Shared test data and mocks in `conftest.py`

### Test Execution

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src --cov-report=html tests/

# Run specific test file
pytest tests/test_vectordb.py -v
```

## Logging Strategy

### Log Levels

- **DEBUG**: Detailed information for debugging
- **INFO**: General informational messages
- **WARNING**: Warning messages (e.g., no relevant docs found)
- **ERROR**: Error messages with exception details

### Logging Configuration

- Module-based loggers for easy filtering
- Structured logging with timestamps
- Configurable log levels

## Reproducibility

### Random Seed Setting

- Seeds set for Python random, NumPy, and PyTorch
- Configurable via `RANDOM_SEED` environment variable
- Default seed: 42
- Ensures consistent results across runs

## Extension Points

### Adding New Document Types

1. Add loader to `loader_mapping` in `load_documents()`
2. Ensure loader returns Document objects with metadata

### Adding New LLM Providers

1. Add API key check in `_initialize_llm()`
2. Import and instantiate new LLM class
3. Return LLM instance following existing pattern

### Customizing Embeddings

1. Set `EMBEDDING_MODEL` environment variable
2. Or pass `embedding_model` parameter to VectorDB constructor
3. Ensure model is compatible with sentence-transformers

### Customizing Chunking

1. Modify `chunk_text()` method in VectorDB
2. Adjust `chunk_size` and `chunk_overlap` parameters
3. Or implement custom chunking strategy

