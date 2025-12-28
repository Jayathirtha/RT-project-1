import os
import uuid
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document collection"},
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        # TODO: Implement text chunking logic
        # You have several options for chunking text - choose one or experiment with multiple:
        #
        # OPTION 1: Simple word-based splitting
        #   - Split text by spaces and group words into chunks of ~chunk_size characters
        #   - Keep track of current chunk length and start new chunks when needed
        #
        # OPTION 2: Use LangChain's RecursiveCharacterTextSplitter
        #   - from langchain_text_splitters import RecursiveCharacterTextSplitter
        #   - Automatically handles sentence boundaries and preserves context better
        #
        # OPTION 3: Semantic splitting (advanced)
        #   - Split by sentences using nltk or spacy
        #   - Group semantically related sentences together
        #   - Consider paragraph boundaries and document structure
        #
        # Feel free to try different approaches and see what works best!

        chunks = []
        # Your implementation here

        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,   # The maximum number of characters in each chunk
        chunk_overlap=100, # Number of characters to overlap between chunks
        length_function=len,
        is_separator_regex=False,
    )
        #print("TODO: Implement text chunking logic")
        chunks = text_splitter.split_text(text)
        print(f"Split text into {len(chunks)} chunks.")

        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        # TODO: Implement document ingestion logic
        # HINT: Loop through each document in the documents list
        # HINT: Extract 'content' and 'metadata' from each document dict
        # HINT: Use self.chunk_text() to split each document into chunks
        # HINT: Create unique IDs for each chunk (e.g., "doc_0_chunk_0")
        # HINT: Use self.embedding_model.encode() to create embeddings for all chunks
        # HINT: Store the embeddings, documents, metadata, and IDs in your vector database
        # HINT: Print progress messages to inform the user

        print(f"Processing {len(documents)} documents...")
        # Your implementation here

        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata

            #print(f'content: {doc.page_content[:100]}...')

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

        try:
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                metadatas=chunk_metadatas,
                documents=chunk_texts
            )
            print(f"Successfully ingested {len(chunks)} chunks from {metadata.get('source')}")
            
        except Exception as e:
            print(f"Failed to store chunks for {metadata.get('source')}: {e}")

        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # TODO: Implement similarity search logic
        # HINT: Use self.embedding_model.encode([query]) to create query embedding
        # HINT: Convert the embedding to appropriate format for your vector database
        # HINT: Use your vector database's search/query method with the query embedding and n_results
        # HINT: Return a dictionary with keys: 'documents', 'metadatas', 'distances', 'ids'
        # HINT: Handle the case where results might be empty

        # Your implementation here

        try:
            # 1. Create embedding for the query
            # Ensure it is converted to a list format for ChromaDB
            query_embedding = self.embedding_model.encode([query]).tolist()

        # 2. Search the ChromaDB collection
        # We pass the embedding to 'query_embeddings'
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
            print(f"Error during similarity search: {e}")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": []
            }