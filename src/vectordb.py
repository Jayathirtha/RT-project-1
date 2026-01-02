import os
import uuid
import logging
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up logger
logger = logging.getLogger(__name__)



class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")
        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        logger.info(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        # OPTION 2: Use LangChain's RecursiveCharacterTextSplitter
        #   - from langchain_text_splitters import RecursiveCharacterTextSplitter
        #   - Automatically handles sentence boundaries and preserves context better

        chunks = []

        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,   # The maximum number of characters in each chunk
        chunk_overlap=100, # Number of characters to overlap between chunks
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
        chunks = text_splitter.split_text(text)
        logger.debug(f"Chunked text into {len(chunks)} chunks")
        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """

        logger.info(f"Processing {len(documents)} documents...")

        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata

            chunks = self.chunk_text(content)
            chunk_texts = []
            chunk_metadatas = []
            chunk_ids = []

            for i, chunk in enumerate(chunks):
                # Create a unique ID for every chunk
                # Combining source name and index helps with traceability
                unique_id = f"{metadata.get('source', 'doc')}_{i}_{str(uuid.uuid4())[:8]}"

                chunk_texts.append(chunk)
                chunk_ids.append(unique_id)

                chunk_meta = metadata.copy()
                chunk_meta["chunk_index"] = i
                chunk_metadatas.append(chunk_meta)
            
            
            embeddings = self.embedding_model.encode(chunk_texts).tolist()
            
            # Add all chunks to the collection at once (outside the loop)
            try:
                self.collection.add(
                    ids=chunk_ids,
                    embeddings=embeddings,
                    metadatas=chunk_metadatas,
                    documents=chunk_texts
                )
            
            except Exception as e:
                logger.error(f"Failed to store chunks for {metadata.get('source', 'unknown')}: {e}")

        logger.info("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
    

        try:
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            return {
                "documents": results["documents"][0],
                "metadatas": results["metadatas"][0],
                "distances": results["distances"][0],
                "ids": results["ids"][0]
            }
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": []
            }