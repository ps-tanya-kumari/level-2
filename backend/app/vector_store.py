import os
import re
import logging
from typing import List, Dict, Any
import chromadb
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def sanitize_collection_name(owner: str, repo: str) -> str:
    """
    Sanitizes owner and repo names to create a valid ChromaDB collection name.
    Rules: 3-63 chars, alphanumeric/underscore/hyphen, starts/ends with alphanumeric.
    """
    name = f"repo_{owner}_{repo}".lower()
    # Replace non-alphanumeric/hyphen/underscore with underscore
    name = re.sub(r'[^a-z0-9_-]', '_', name)
    
    # Ensure it starts and ends with alphanumeric
    if not name[0].isalnum():
        name = 'c' + name[1:]
    if not name[-1].isalnum():
        name = name[:-1] + '0'
        
    # Ensure proper length (3 to 63 characters)
    if len(name) < 3:
        name = name.ljust(3, '0')
    elif len(name) > 63:
        name = name[:63]
        # Re-ensure ending char is alphanumeric after truncation
        if not name[-1].isalnum():
            name = name[:-1] + '0'
            
    return name

class VectorStore:
    def __init__(self):
        db_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
        # Ensure directory exists
        os.makedirs(db_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_dir)
        logger.info(f"ChromaDB PersistentClient initialized at: {db_dir}")

    def get_collection(self, owner: str, repo: str):
        """Retrieve a collection for a specific repository."""
        collection_name = sanitize_collection_name(owner, repo)
        try:
            return self.client.get_collection(name=collection_name)
        except Exception:
            return None

    def get_or_create_collection(self, owner: str, repo: str):
        """Get or create collection for a repository."""
        collection_name = sanitize_collection_name(owner, repo)
        return self.client.get_or_create_collection(name=collection_name)

    def delete_collection(self, owner: str, repo: str):
        """Delete collection for a repository (useful to re-index)."""
        collection_name = sanitize_collection_name(owner, repo)
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted ChromaDB collection: {collection_name}")
            return True
        except Exception as e:
            logger.warning(f"Collection {collection_name} does not exist or couldn't be deleted: {e}")
            return False

    def add_chunks(self, owner: str, repo: str, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Insert list of chunks and their corresponding embeddings into the collection."""
        if not chunks:
            return
            
        collection = self.get_or_create_collection(owner, repo)
        
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        # ChromaDB allows batched addition. Add in batches of 200 to prevent payload issues
        batch_size = 200
        for i in range(0, len(ids), batch_size):
            end_idx = i + batch_size
            collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=metadatas[i:end_idx],
                documents=documents[i:end_idx]
            )
        logger.info(f"Indexed {len(chunks)} chunks in collection {collection.name}")

    def query_similar_chunks(self, owner: str, repo: str, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """Query the collection to retrieve top matching chunks."""
        collection = self.get_collection(owner, repo)
        if not collection:
            logger.warning(f"No collection found for {owner}/{repo}")
            return []
            
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format results into readable dictionary list
            formatted_results = []
            if results and results.get("documents"):
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                ids = results["ids"][0]
                distances = results.get("distances", [[]])[0]
                
                for idx in range(len(documents)):
                    formatted_results.append({
                        "id": ids[idx],
                        "content": documents[idx],
                        "metadata": metadatas[idx],
                        "distance": distances[idx] if idx < len(distances) else None
                    })
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            return []
