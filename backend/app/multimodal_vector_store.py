#multimodal_vector_store.py
"""
Multimodal Vector Store Module

Stores and retrieves text chunks with linked images.
Uses ChromaDB for vector storage with caption-based image embeddings.
Images are stored as optimized base64 data URIs, never as external URLs.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import json
import uuid
import os
import logging

logger = logging.getLogger(__name__)


class MultimodalVectorStore:
    """
    Vector store that handles both text and images.

    Architecture:
    - text_collection: Stores text chunks with metadata linking to image_ids
    - image_collection: Stores image captions for embedding-based retrieval
    - image_store: In-memory dict (or Redis in production) for base64 image data

    Images are NEVER stored as URLs. They are always base64 data URIs.
    """

    def __init__(self, persist_directory: str = None):
        """
        Initialize the multimodal vector store.

        Args:
            persist_directory: Directory for ChromaDB persistence
        """
        self.persist_directory = persist_directory or os.getenv(
            "MULTIMODAL_DB_PATH",
            "./multimodal_db"
        )
        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Text collection - stores text chunks with image links
        self.text_collection = self.client.get_or_create_collection(
            name="text_chunks",
            metadata={"hnsw:space": "cosine"}
        )

        # Image caption collection - stores captions for embedding
        self.image_collection = self.client.get_or_create_collection(
            name="image_captions",
            metadata={"hnsw:space": "cosine"}
        )

        # In-memory store for base64 images
        # In production, use Redis or database
        self.image_store: Dict[str, Dict[str, str]] = {}

        logger.info(f"MultimodalVectorStore initialized at {self.persist_directory}")

    def add_text_chunk(
        self,
        text: str,
        metadata: Dict[str, Any],
        image_ids: List[str] = None,
        embedding: List[float] = None
    ) -> str:
        """
        Add a text chunk with linked image IDs.

        Args:
            text: Text content of the chunk
            metadata: Metadata dict (source, section, laptop_brand, etc.)
            image_ids: List of image IDs associated with this chunk
            embedding: Optional pre-computed embedding

        Returns:
            chunk_id: Unique identifier for the stored chunk
        """
        chunk_id = str(uuid.uuid4())

        # Add image_ids to metadata
        metadata = metadata.copy()
        metadata["image_ids"] = json.dumps(image_ids or [])
        metadata["chunk_id"] = chunk_id

        try:
            self.text_collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[chunk_id],
                embeddings=[embedding] if embedding else None
            )
            logger.debug(f"Added text chunk {chunk_id} with {len(image_ids or [])} images")
            return chunk_id
        except Exception as e:
            logger.error(f"Failed to add text chunk: {e}")
            raise

    def add_image(
        self,
        caption: str,
        base64_data: str,
        mime_type: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Add an image with caption for embedding-based retrieval.

        The base64 data is stored separately (in memory or Redis).
        The caption is stored in ChromaDB for text-based retrieval.

        Args:
            caption: Text description of the image (for embedding)
            base64_data: Base64 data URI string (data:image/jpeg;base64,...)
            mime_type: MIME type (e.g., "image/jpeg")
            metadata: Additional metadata

        Returns:
            image_id: Unique identifier for the stored image
        """
        image_id = str(uuid.uuid4())

        # Store base64 data separately (never in ChromaDB metadata)
        self.image_store[image_id] = {
            "mime_type": mime_type,
            "data": base64_data
        }

        # Store caption in vector DB for retrieval
        image_metadata = {
            **(metadata or {}),
            "image_id": image_id,
            "mime_type": mime_type,
            "stored_at": str(uuid.uuid4())  # Timestamp placeholder
        }

        try:
            self.image_collection.add(
                documents=[caption],
                metadatas=[image_metadata],
                ids=[image_id]
            )
            logger.debug(f"Added image {image_id} with caption: {caption[:50]}...")
            return image_id
        except Exception as e:
            logger.error(f"Failed to add image: {e}")
            # Clean up image store if DB insert failed
            del self.image_store[image_id]
            raise

    def retrieve_with_images(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Dict = None
    ) -> Dict[str, Any]:
        """
        Retrieve text chunks and their associated images.

        Args:
            query: Search query text
            top_k: Number of text chunks to retrieve
            filter_metadata: Optional metadata filters

        Returns:
            {
                "chunks": List of {"text", "metadata", "score"},
                "images": List of {"mime_type", "data"}
            }
        """
        try:
            # Query text collection
            text_results = self.text_collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )

            chunks = []
            image_ids = set()

            # Process text results
            if text_results and text_results['documents']:
                for i, doc in enumerate(text_results['documents'][0]):
                    metadata = text_results['metadatas'][0][i]

                    # Extract linked image IDs
                    chunk_image_ids = json.loads(metadata.get('image_ids', '[]'))
                    image_ids.update(chunk_image_ids)

                    chunks.append({
                        "text": doc,
                        "metadata": metadata,
                        "score": text_results['distances'][0][i]
                    })

            # Retrieve associated images
            images = []
            for img_id in image_ids:
                if img_id in self.image_store:
                    images.append(self.image_store[img_id])
                else:
                    logger.warning(f"Image {img_id} not found in image store")

            logger.info(f"Retrieved {len(chunks)} chunks with {len(images)} images for query: {query[:50]}...")

            return {
                "chunks": chunks,
                "images": images
            }

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                "chunks": [],
                "images": []
            }

    def search_images_by_caption(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, str]]:
        """
        Search for images using caption-based retrieval.

        Args:
            query: Search query
            top_k: Number of images to retrieve

        Returns:
            List of {"mime_type", "data"} dicts
        """
        try:
            results = self.image_collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )

            images = []
            if results and results['metadatas']:
                for metadata in results['metadatas'][0]:
                    img_id = metadata.get('image_id')
                    if img_id and img_id in self.image_store:
                        images.append(self.image_store[img_id])

            return images

        except Exception as e:
            logger.error(f"Image search failed: {e}")
            return []

    def get_chunks_by_laptop(
        self,
        brand: str,
        model: str,
        n_results: int = 10
    ) -> Dict[str, Any]:
        """
        Get all chunks and images for a specific laptop model.

        Args:
            brand: Laptop brand
            model: Laptop model
            n_results: Max chunks to retrieve

        Returns:
            Dict with chunks and images
        """
        filter_metadata = {
            "$and": [
                {"laptop_brand": {"$eq": brand}},
                {"laptop_model": {"$eq": model}}
            ]
        }
        return self.retrieve_with_images(
            query=f"{brand} {model}",
            top_k=n_results,
            filter_metadata=filter_metadata
        )

    def delete_laptop_data(self, brand: str, model: str) -> bool:
        """
        Delete all chunks and images for a specific laptop model.

        Args:
            brand: Laptop brand
            model: Laptop model

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all chunks for this laptop
            results = self.retrieve_with_images(
                query=f"{brand} {model}",
                top_k=1000
            )

            chunk_ids = [c["metadata"].get("chunk_id") for c in results["chunks"] if c["metadata"].get("chunk_id")]

            if chunk_ids:
                self.text_collection.delete(ids=chunk_ids)

            # Clean up image store
            for chunk in results["chunks"]:
                image_ids = json.loads(chunk["metadata"].get("image_ids", "[]"))
                for img_id in image_ids:
                    if img_id in self.image_store:
                        del self.image_store[img_id]

            logger.info(f"Deleted {len(chunk_ids)} chunks for {brand} {model}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete laptop data: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dict with counts and collection info
        """
        try:
            return {
                "text_chunks": self.text_collection.count(),
                "image_captions": self.image_collection.count(),
                "images_in_store": len(self.image_store),
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def clear_all(self) -> bool:
        """
        Clear all data from the vector store.

        Returns:
            True if successful
        """
        try:
            # Delete all text chunks
            all_text = self.text_collection.get()
            if all_text and all_text['ids']:
                self.text_collection.delete(ids=all_text['ids'])

            # Delete all image captions
            all_images = self.image_collection.get()
            if all_images and all_images['ids']:
                self.image_collection.delete(ids=all_images['ids'])

            # Clear image store
            self.image_store.clear()

            logger.info("Cleared all data from multimodal vector store")
            return True

        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            return False


# Global instance
_multimodal_vector_store_instance = None


def get_multimodal_vector_store() -> MultimodalVectorStore:
    """Get or create the global multimodal vector store instance"""
    global _multimodal_vector_store_instance
    if _multimodal_vector_store_instance is None:
        _multimodal_vector_store_instance = MultimodalVectorStore()
    return _multimodal_vector_store_instance