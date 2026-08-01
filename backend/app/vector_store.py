#vector_store.py
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import json
import os
from datetime import datetime

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client with persistence"""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="laptop_documentation",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documentation_chunk(
        self,
        chunk_id: str,
        text: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ):
        """
        Add a documentation chunk to the vector store
        
        Args:
            chunk_id: Unique identifier for the chunk
            text: The text content of the chunk
            metadata: Metadata including source_url, laptop_model, section, etc.
            embedding: Optional pre-computed embedding
        """
        try:
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[chunk_id],
                embeddings=[embedding] if embedding else None
            )
            return True
        except Exception as e:
            print(f"Error adding chunk to vector store: {e}")
            return False
    
    def search_similar_chunks(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documentation chunks
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"brand": "Dell"})
        
        Returns:
            List of matching chunks with metadata and scores
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        "text": doc,
                        "metadata": results['metadatas'][0][i],
                        "score": results['distances'][0][i]
                    })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching vector store: {e}")
            return []
    
    def get_chunks_by_laptop(
        self,
        brand: str,
        model: str,
        n_results: int = 10
    ) -> List[Dict]:
        """Get all chunks for a specific laptop model"""
        filter_metadata = {
            "$and": [
                {"brand": {"$eq": brand}},
                {"model": {"$eq": model}}
            ]
        }
        return self.search_similar_chunks(
            query=f"{brand} {model}",
            n_results=n_results,
            filter_metadata=filter_metadata
        )
    
    def delete_laptop_chunks(self, brand: str, model: str):
        """Delete all chunks for a specific laptop model"""
        try:
            # Get all chunks for this laptop
            results = self.get_chunks_by_laptop(brand, model, n_results=1000)
            chunk_ids = [r['metadata']['chunk_id'] for r in results]
            
            if chunk_ids:
                self.collection.delete(ids=chunk_ids)
                return True
            return False
        except Exception as e:
            print(f"Error deleting chunks: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection.name
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_chunks": 0, "error": str(e)}


# Global instance
_vector_store_instance = None

def get_vector_store() -> VectorStore:
    """Get or create the global vector store instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        persist_dir = os.getenv("CHROMADB_PATH", "./chroma_db")
        _vector_store_instance = VectorStore(persist_directory=persist_dir)
    return _vector_store_instance