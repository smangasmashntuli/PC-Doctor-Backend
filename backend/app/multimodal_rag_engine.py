#multimodal_rag_engine.py
"""
Multimodal RAG Engine Module

Orchestrates the complete multimodal RAG pipeline:
1. Ingestion: Process text + images, store in vector DB
2. Retrieval: Get relevant text chunks + linked images
3. Generation: LLM generates answer using retrieved context
4. Response: Return answer + base64 images in required format

Output format:
{
    "answer": "Generated textual response...",
    "images": [
        {"mime_type": "image/jpeg", "data": "data:image/jpeg;base64,..."}
    ]
}
"""
import os
import logging
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import re

from backend.app.image_processor import ImageProcessor, ImageProcessingError, get_image_processor
from backend.app.multimodal_vector_store import get_multimodal_vector_store

logger = logging.getLogger(__name__)


class RetrievalError(Exception):
    """Raised when retrieval fails"""
    pass


class GenerationError(Exception):
    """Raised when LLM generation fails"""
    pass


class MultimodalRAGEngine:
    """
    Main RAG engine that orchestrates multimodal retrieval and generation.

    Pipeline:
    1. Ingest documents with text and images
    2. Store optimized images as base64 data URIs
    3. Store text chunks with links to image IDs
    4. On query: retrieve relevant chunks + images
    5. Generate answer using LLM with context
    6. Return answer + images in required format
    """

    def __init__(self):
        self.vector_store = get_multimodal_vector_store()
        self.image_processor = get_image_processor()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        # LLM configuration (uses existing Gemini service)
        self._llm_service = None

    @property
    def llm_service(self):
        """Lazy load the LLM service (Gemini)"""
        if self._llm_service is None:
            try:
                from backend.app.gemini_service import get_gemini_service
                self._llm_service = get_gemini_service()
            except Exception as e:
                logger.error(f"Failed to load LLM service: {e}")
                raise GenerationError(f"LLM service unavailable: {e}")
        return self._llm_service

    async def ingest_document(
        self,
        text: str,
        images: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest a document with text and associated images.

        Args:
            text: Document text content
            images: List of {"bytes": bytes, "caption": str} dicts
            metadata: Additional metadata (source, section, laptop_brand, etc.)

        Returns:
            {"status": "success", "chunks_stored": int, "images_stored": int}

        Raises:
            ImageProcessingError: If image processing fails
        """
        try:
            metadata = metadata or {}
            image_ids = []

            # Process and store images
            for img in images or []:
                try:
                    # Optimize image and convert to base64
                    optimized = self.image_processor.process_image_to_base64(
                        img["bytes"]
                    )

                    # Store image with caption
                    image_id = self.vector_store.add_image(
                        caption=img.get("caption", "Laptop component image"),
                        base64_data=optimized["data"],
                        mime_type=optimized["mime_type"],
                        metadata=metadata
                    )
                    image_ids.append(image_id)

                except ImageProcessingError as e:
                    logger.error(f"Failed to process image: {e}")
                    # Continue with other images
                    continue

            # Chunk text
            chunks = self._chunk_text(text)

            # Store text chunks with image links
            for chunk in chunks:
                self.vector_store.add_text_chunk(
                    text=chunk,
                    metadata=metadata,
                    image_ids=image_ids
                )

            logger.info(f"Ingested document: {len(chunks)} chunks, {len(image_ids)} images")

            return {
                "status": "success",
                "chunks_stored": len(chunks),
                "images_stored": len(image_ids)
            }

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise

    async def ingest_web_page(
        self,
        url: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Ingest a web page: scrape text and images, process, and store.

        Args:
            url: Web page URL to ingest
            metadata: Additional metadata

        Returns:
            Ingestion results
        """
        try:
            metadata = metadata or {}
            metadata["source_url"] = url

            # Fetch the web page
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text content
            text_content = soup.get_text(separator=' ', strip=True)
            text_content = re.sub(r'\s+', ' ', text_content)

            # Extract and download images
            image_urls = self.image_processor.extract_images_from_html(
                response.text,
                base_url=url
            )

            images = []
            for img_url in image_urls[:5]:  # Limit to 5 images per page
                try:
                    optimized = self.image_processor.download_and_optimize(img_url)
                    images.append({
                        "bytes": None,  # Already processed
                        "caption": f"Image from {url}",
                        "optimized": optimized
                    })
                except ImageProcessingError as e:
                    logger.warning(f"Failed to download image {img_url}: {e}")
                    continue

            # Process images that were downloaded
            image_ids = []
            for img in images:
                if img.get("optimized"):
                    image_id = self.vector_store.add_image(
                        caption=img["caption"],
                        base64_data=img["optimized"]["data"],
                        mime_type=img["optimized"]["mime_type"],
                        metadata=metadata
                    )
                    image_ids.append(image_id)

            # Chunk and store text
            chunks = self._chunk_text(text_content)

            for chunk in chunks:
                self.vector_store.add_text_chunk(
                    text=chunk,
                    metadata=metadata,
                    image_ids=image_ids
                )

            logger.info(f"Ingested web page {url}: {len(chunks)} chunks, {len(image_ids)} images")

            return {
                "status": "success",
                "url": url,
                "chunks_stored": len(chunks),
                "images_stored": len(image_ids)
            }

        except requests.RequestException as e:
            logger.error(f"Failed to fetch web page {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Web page ingestion failed: {e}")
            raise

    async def query(
        self,
        question: str,
        top_k: int = 5,
        laptop_brand: str = None,
        laptop_model: str = None
    ) -> Dict[str, Any]:
        """
        Query the multimodal RAG system.

        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            laptop_brand: Optional filter by laptop brand
            laptop_model: Optional filter by laptop model

        Returns:
            {
                "answer": "Generated textual response...",
                "images": [{"mime_type": str, "data": str}]
            }
        """
        try:
            # Build filter metadata if laptop info provided
            filter_metadata = None
            if laptop_brand and laptop_model:
                filter_metadata = {
                    "$and": [
                        {"laptop_brand": {"$eq": laptop_brand}},
                        {"laptop_model": {"$eq": laptop_model}}
                    ]
                }

            # Retrieve relevant chunks and images
            results = self.vector_store.retrieve_with_images(
                query=question,
                top_k=top_k,
                filter_metadata=filter_metadata
            )

            # Check if we found any relevant content
            if not results["chunks"]:
                return {
                    "answer": "I couldn't find relevant information for your query. Please try rephrasing your question or ingesting more documentation.",
                    "images": []
                }

            # Build context for LLM
            context = self._build_context(results["chunks"])

            # Generate answer using LLM
            answer = await self._generate_answer(question, context)

            # Return in required format
            return {
                "answer": answer,
                "images": results["images"]
            }

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your request. Please try again.",
                "images": []
            }

    async def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM with retrieved context.

        Args:
            question: User's question
            context: Retrieved context text

        Returns:
            Generated answer text

        Raises:
            GenerationError: If generation fails
        """
        try:
            # Build prompt for LLM
            prompt = self._build_llm_prompt(question, context)

            # Use Gemini service to generate response
            response = self.llm_service.client.models.generate_content(
                model=self.llm_service.model_name,
                contents=prompt,
                config={
                    "temperature": 0.7,
                    "max_output_tokens": 1024,
                    "top_p": 0.95,
                }
            )

            return response.text.strip()

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise GenerationError(f"Failed to generate answer: {e}")

    def _build_context(self, chunks: List[Dict]) -> str:
        """
        Build context string from retrieved chunks.

        Args:
            chunks: List of chunk dicts with "text" and "metadata"

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})

            # Add source if available
            source = metadata.get("source_url", metadata.get("source", "Unknown"))
            section = metadata.get("section", "")

            context_parts.append(f"[{i}] {section.upper() if section else 'CONTEXT'}")
            context_parts.append(f"Source: {source}")
            context_parts.append(f"Content: {text}")
            context_parts.append("")  # Empty line for readability

        return "\n".join(context_parts)

    def _build_llm_prompt(self, question: str, context: str) -> str:
        """
        Build the LLM prompt with context and question.

        Args:
            question: User's question
            context: Retrieved context

        Returns:
            Formatted prompt string
        """
        return f"""You are PC-Doctor-AI, a helpful laptop troubleshooting assistant.

Use the following retrieved context to answer the user's question. If the context doesn't contain enough information, say so honestly.

**Retrieved Context:**
{context}

**User Question:**
{question}

**Instructions:**
- Provide a clear, non-technical answer suitable for everyday users
- Use numbered steps if providing instructions
- Be concise but thorough
- If recommending physical repairs, include safety warnings
- Reference images if they are relevant to the answer

**Answer:**"""

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Split text into chunks for storage.

        Args:
            text: Text to chunk
            chunk_size: Maximum words per chunk

        Returns:
            List of text chunks
        """
        # Simple word-based chunking
        # In production, use a proper text chunker (e.g., langchain's RecursiveCharacterTextSplitter)
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    async def explain_component_with_images(
        self,
        component_name: str,
        laptop_brand: str = None,
        laptop_model: str = None
    ) -> Dict[str, Any]:
        """
        Explain a laptop component with relevant images.

        Args:
            component_name: Name of the component (e.g., "RAM", "SSD")
            laptop_brand: Optional laptop brand for context
            laptop_model: Optional laptop model for context

        Returns:
            {"answer": str, "images": [{"mime_type", "data"}]}
        """
        try:
            # Search for relevant images by caption
            images = self.vector_store.search_images_by_caption(
                query=component_name,
                top_k=3
            )

            # Generate explanation using LLM
            prompt = f"""You are PC-Doctor-AI. Explain the {component_name} in a laptop in simple, everyday language.

Include:
1. What the component does (its function)
2. Common failure symptoms
3. Safe handling tips
4. Upgrade possibilities (if applicable)

Keep the answer short, calm, and easy to understand for non-technical users."""

            if laptop_brand and laptop_model:
                prompt = f"""You are PC-Doctor-AI. Explain the {component_name} in a {laptop_brand} {laptop_model} laptop in simple, everyday language.

Include:
1. What the component does (its function)
2. Common failure symptoms for this model
3. Safe handling tips
4. Upgrade possibilities (if applicable)

Keep the answer short, calm, and easy to understand for non-technical users."""

            response = self.llm_service.client.models.generate_content(
                model=self.llm_service.model_name,
                contents=prompt,
                config={
                    "temperature": 0.3,
                    "max_output_tokens": 256,
                    "top_p": 0.9,
                }
            )

            return {
                "answer": response.text.strip(),
                "images": images
            }

        except Exception as e:
            logger.error(f"Component explanation failed: {e}")
            return {
                "answer": f"I apologize, but I couldn't generate an explanation for {component_name} at this time.",
                "images": []
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the multimodal RAG system"""
        return self.vector_store.get_stats()


# Global instance
_multimodal_rag_engine_instance = None


def get_multimodal_rag_engine() -> MultimodalRAGEngine:
    """Get or create the global multimodal RAG engine instance"""
    global _multimodal_rag_engine_instance
    if _multimodal_rag_engine_instance is None:
        _multimodal_rag_engine_instance = MultimodalRAGEngine()
    return _multimodal_rag_engine_instance