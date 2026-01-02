# RAG Implementation Project

A comprehensive educational project demonstrating **Retrieval Augmented Generation (RAG)** implementation using vector databases, embeddings, and large language models. This project serves as a tutorial for students and educators learning about modern AI/ML systems.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Dependencies](#dependencies)
- [Development](#development)
- [License](#license)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a complete RAG system that combines:

- **Document Loading**: Multi-format document processing (PDF, CSV, TXT, JSON)
- **Vector Storage**: ChromaDB for efficient similarity search
- **Embeddings**: Sentence transformers for semantic understanding
- **Retrieval**: Semantic search to find relevant document chunks
- **Generation**: LLM-powered answer generation based on retrieved context

**Primary Focus**: Educational content providing tutorials and instructional guides for RAG implementation.

**Objective**: Showcase expertise in building production-ready RAG systems for professional opportunities.

**Target Audience**: Students and educators learning about RAG systems, vector databases, and LLM integration.

## ✨ Features

- 🔍 **Semantic Search**: Vector-based similarity search using ChromaDB
- 📄 **Multi-Format Support**: Handles PDF, CSV, TXT, and JSON documents
- 🤖 **Multi-LLM Support**: Compatible with OpenAI, Groq, and Google Gemini APIs
- 🧩 **Modular Design**: Clean separation between vector database and RAG components
- 🔄 **Batch Processing**: Optimized embedding generation and storage

## 📁 Project Structure

```
RT-project-1/
├── src/
│   ├── app.py              # Main RAG assistant implementation
│   └── vectordb.py         # Vector database wrapper (ChromaDB)
├── data/                   # Document storage directory
│   ├── *.pdf               # PDF documents
│   ├── *.csv               # CSV files
│   ├── *.txt               # Text files
│   └── *.json              # JSON files
├── chroma_db/              # ChromaDB persistent storage
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create from template)
└── README.md               # This file
```

## 🔧 Prerequisites

- **Python**: 3.11 or higher
- **API Keys**: At least one of the following:
  - OpenAI API key (for GPT models)
  - Groq API key (for Llama models)
  - Google API key (for Gemini models)

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd RT-project-1
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

1. **Create a `.env` file** in the project root:
   ```env
   # LLM Provider Configuration (choose at least one)
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=gpt-4o-mini
   
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   
   GOOGLE_API_KEY=your_google_api_key_here
   GOOGLE_MODEL=gemini-2.0-flash
   
   # Embedding Configuration
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   
   # Vector Database Configuration
   CHROMA_COLLECTION_NAME=rag_documents
   ```

2. **Add documents** to the `data/` directory:
   - Place PDF, CSV, TXT, or JSON files in the `data/` folder
   - The system will automatically detect and load them

## 🚀 Usage

### Basic Usage

Run the RAG assistant:

```bash
python src/app.py
```

The assistant will:
1. Load documents from the `data/` directory
2. Process and store them in the vector database
3. Start an interactive Q&A session

### Example Session

```
Initializing RAG Assistant...
Loading documents...
Loaded 5 sample documents
Processing 5 documents...
Documents added to vector database

Enter a question or 'quit' to exit: What is protein?
[Assistant provides answer based on retrieved context]

Enter a question or 'quit' to exit: quit
```
## 🏗️ Architecture

### RAG Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Documents  │ --> │   Chunking   │ --> │  Embedding  │ --> │ Vector Store │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                      │
┌─────────────┐     ┌──────────────┐     ┌─────────────┐            │
│ User Query │ --> │ Query Embed  │ --> │  Similarity  │ <----------┘
└─────────────┘     └──────────────┘     └─────────────┘
                                                      │
                                                      v
                                            ┌──────────────┐
                                            │   Context    │
                                            └──────────────┘
                                                      │
                                                      v
                                            ┌──────────────┐
                                            │     LLM      │ --> Answer
                                            └──────────────┘
```

### Components

- **VectorDB** (`src/vectordb.py`): Manages document ingestion, chunking, embedding, and similarity search
- **RAGAssistant** (`src/app.py`): Orchestrates the complete RAG pipeline
- **Document Loaders**: LangChain loaders for various file formats

## 📚 Dependencies

### Core Dependencies

- **chromadb** (1.0.12): Vector database for similarity search
- **langchain** (0.3.27): Framework for LLM applications
- **langchain-core** (0.3.76): Core LangChain functionality
- **sentence-transformers** (5.1.0): Embedding model library
- **langchain-text-splitters** (0.3.11): Text chunking utilities

### LLM Provider Libraries

- **langchain-openai** (0.3.33): OpenAI integration
- **langchain-groq** (0.3.8): Groq integration
- **langchain-google-genai** (2.1.10): Google Gemini integration

### Document Processing

- **langchain-community** (0.3.30): Community loaders
- **pypdf** (6.5.0): PDF processing
- **jq** (1.10.0): JSON processing

### Utilities

- **python-dotenv** (1.1.1): Environment variable management
- **numpy** (2.3.3): Numerical operations
- **torch** (2.8.0): PyTorch for transformer models

See `requirements.txt` for the complete list of dependencies.

## 💻 Development

### Development Dependencies

All dependencies are listed in `requirements.txt`. For development, you may also want:

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** (CC BY-NC-SA 4.0) license.

### License Summary

**You are free to:**
- ✅ Share: Copy and redistribute the material
- ✅ Adapt: Remix, transform, and build upon the material

## 🤝 Contributing

This is an educational project. Contributions, suggestions, and improvements are welcome!

### Contribution Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For questions, issues, or contributions:
- Open an issue on the repository
- Review the code comments for implementation details
- Check the `docs/` directory for additional documentation

## 🙏 Acknowledgments

- **ReadyTensor**: This capstone project was created using ReadyTensor's git repository as the foundation. Special thanks to ReadyTensor for providing the learning track, capstone guidance, and repository template that served as the starting point for this RAG implementation tutorial.
- **LangChain**: For the excellent framework and document loaders
- **ChromaDB**: For the open-source vector database
- **HuggingFace**: For sentence transformer models
- **OpenAI, Groq, Google**: For LLM APIs

---

**Note**: This project is intended for educational purposes. Ensure you comply with API usage terms and data privacy regulations when using this code in production environments.
