# Multimodal RAG Pipeline Design

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MULTIMODAL RAG PIPELINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │
│  │  INGESTION   │───▶│  PROCESSING  │───▶│   VECTOR STORAGE     │ │
│  │              │    │              │    │                      │ │
│  │ • Documents  │    │ • Text Chunk│    │ • ChromaDB           │ │
│  │ • Images     │    │ • Image Opt │    │ • Text Embeddings    │ │
│  │ • PDFs       │    │ • Caption   │    │ • Image Embeddings   │ │
│  │ • Web Pages  │    │ • Embed     │    │ • Metadata Links     │ │
│  └──────────────┘    └──────────────┘    └──────────────────────┘ │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │
│  │  RETRIEVAL   │───▶│  GENERATION │───▶│   RESPONSE FORMAT    │ │
│  │              │    │              │    │                      │ │
│  │ • Text Query │    │ • LLM Gen   │    │ • Answer Text        │ │
│  │ • Top-K Text │    │ • Context   │    │ • Base64 Images      │ │
│  │ • Linked Img │    │ • Images    │    │ • Metadata           │ │
│  └──────────────┘    └──────────────┘    └──────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Image Processor (`image_processor.py`)

**Purpose**: Optimize and convert images to base64 data URIs

**Key Functions**:

- `optimize_image(image_bytes, max_size=800, quality=70)` → optimized bytes
- `image_to_base64(image_bytes, mime_type="image/jpeg")` → data URI string
- `download_and_optimize(url)` → optimized base64 data URI
- `extract_images_from_pdf(pdf_bytes)` → list of optimized images

**Libraries**: Pillow (PIL), io, base64

### 2. Multimodal Vector Store (`multimodal_vector_store.py`)

**Purpose**: Store and retrieve text chunks with linked images

**Key Features**:

- Text chunks with metadata linking to optimized images
- Caption-based embeddings (text descriptions of images)
- Separate collections for text and image metadata
- Cross-reference between text chunks and images

**Schema**:

```python
TextChunk:
  - id: str
  - text: str
  - embedding: List[float]
  - metadata: {
      "source": str,
      "section": str,
      "image_ids": List[str],  # Links to associated images
      "laptop_brand": str,
      "laptop_model": str
    }

ImageRecord:
  - id: str
  - caption: str  # Text description for embedding
  - base64_data: str  # Optimized base64 data URI
  - mime_type: str
  - metadata: {
      "source": str,
      "width": int,
      "height": int,
      "quality": int
    }
```

### 3. Multimodal RAG Engine (`multimodal_rag_engine.py`)

**Purpose**: Orchestrate ingestion, retrieval, and generation

**Key Functions**:

- `ingest_document(text, images)` → chunk and store with images
- `ingest_web_page(url)` → scrape text and images, process
- `retrieve(query, top_k=5)` → text chunks + linked images
- `generate_answer(query, context, images)` → LLM response with images

### 4. Response Formatter (`response_formatter.py`)

**Purpose**: Ensure consistent output format

**Output Schema**:

```json
{
  "answer": "Generated textual response here...",
  "images": [
    {
      "mime_type": "image/jpeg",
      "data": "data:image/jpeg;base64,/9j/4AAQ..."
    }
  ]
}
```

## Image Optimization Strategy

### Libraries

- **Pillow (PIL)**: Image processing and format conversion
- **io.BytesIO**: In-memory byte streams
- **base64**: Encoding to data URIs

### Optimization Parameters

```python
MAX_DIMENSION = 800      # Max width/height in pixels
JPEG_QUALITY = 70        # Compression quality (60-75 range)
WEBP_QUALITY = 75        # WebP alternative (slightly higher)
MIN_DIMENSION = 100      # Don't go below this
```

### Optimization Pipeline

1. **Load**: Open image from bytes/file/URL
2. **Resize**: Maintain aspect ratio, max 800px on longest side
3. **Convert**: Convert to RGB (remove alpha channel)
4. **Compress**: Save as JPEG with quality=70
5. **Encode**: Convert to base64 data URI
6. **Validate**: Ensure size < 100KB (typical)

### Size Estimation

- Original 1920x1080 PNG: ~2MB
- After optimization (800px, JPEG q70): ~30-50KB
- Base64 encoded: ~40-67KB (33% overhead)
- Target: < 100KB per image

## Implementation Details

### Image Processor Implementation

```python
from PIL import Image
import io
import base64
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class ImageProcessor:
    MAX_DIMENSION = 800
    JPEG_QUALITY = 70
    WEBP_QUALITY = 75

    @classmethod
    def optimize_image(
        cls,
        image_bytes: bytes,
        max_dimension: int = 800,
        quality: int = 70,
        output_format: str = "JPEG"
    ) -> Tuple[bytes, str]:
        """
        Optimize image: resize, compress, convert format

        Returns:
            Tuple of (optimized_bytes, mime_type)
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB (remove alpha)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')

            # Resize maintaining aspect ratio
            width, height = img.size
            if max(width, height) > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int((height * max_dimension) / width)
                else:
                    new_height = max_dimension
                    new_width = int((width * max_dimension) / height)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save optimized
            output = io.BytesIO()
            if output_format.upper() == "WEBP":
                img.save(output, format="WEBP", quality=quality, method=6)
                mime_type = "image/webp"
            else:
                img.save(output, format="JPEG", quality=quality, optimize=True)
                mime_type = "image/jpeg"

            return output.getvalue(), mime_type

        except Exception as e:
            logger.error(f"Image optimization failed: {e}")
            raise

    @classmethod
    def to_base64_data_uri(
        cls,
        image_bytes: bytes,
        mime_type: str = "image/jpeg"
    ) -> str:
        """Convert image bytes to base64 data URI"""
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:{mime_type};base64,{b64}"

    @classmethod
    def process_image_to_base64(
        cls,
        image_bytes: bytes,
        max_dimension: int = 800,
        quality: int = 70
    ) -> dict:
        """
        Full pipeline: optimize + convert to base64

        Returns:
            {"mime_type": str, "data": str}
        """
        optimized_bytes, mime_type = cls.optimize_image(
            image_bytes, max_dimension, quality
        )
        data_uri = cls.to_base64_data_uri(optimized_bytes, mime_type)

        return {
            "mime_type": mime_type,
            "data": data_uri
        }
```

### Multimodal Vector Store Implementation

```python
import chromadb
from typing import List, Dict, Optional, Any
import json
import uuid

class MultimodalVectorStore:
    def __init__(self, persist_directory: str = "./multimodal_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Text collection
        self.text_collection = self.client.get_or_create_collection(
            name="text_chunks",
            metadata={"hnsw:space": "cosine"}
        )

        # Image metadata collection (stores captions, not base64)
        self.image_collection = self.client.get_or_create_collection(
            name="image_captions",
            metadata={"hnsw:space": "cosine"}
        )

        # In-memory store for base64 images (or use Redis/DB in production)
        self.image_store = {}  # image_id -> {"mime_type", "data"}

    def add_text_chunk(
        self,
        text: str,
        metadata: Dict,
        image_ids: List[str] = None
    ) -> str:
        """Add text chunk with linked image IDs"""
        chunk_id = str(uuid.uuid4())

        metadata["image_ids"] = json.dumps(image_ids or [])
        metadata["chunk_id"] = chunk_id

        self.text_collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[chunk_id]
        )

        return chunk_id

    def add_image(
        self,
        caption: str,
        base64_data: str,
        mime_type: str,
        metadata: Dict
    ) -> str:
        """Add image with caption for embedding"""
        image_id = str(uuid.uuid4())

        # Store base64 data separately
        self.image_store[image_id] = {
            "mime_type": mime_type,
            "data": base64_data
        }

        # Store caption in vector DB for retrieval
        self.image_collection.add(
            documents=[caption],
            metadatas=[{
                **metadata,
                "image_id": image_id,
                "mime_type": mime_type
            }],
            ids=[image_id]
        )

        return image_id

    def retrieve_with_images(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:
        """Retrieve text chunks and their associated images"""
        # Get text chunks
        text_results = self.text_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        image_ids = set()

        for i, doc in enumerate(text_results['documents'][0]):
            metadata = text_results['metadatas'][0][i]
            chunk_image_ids = json.loads(metadata.get('image_ids', '[]'))
            image_ids.update(chunk_image_ids)

            chunks.append({
                "text": doc,
                "metadata": metadata,
                "score": text_results['distances'][0][i]
            })

        # Get associated images
        images = []
        for img_id in image_ids:
            if img_id in self.image_store:
                images.append(self.image_store[img_id])

        return {
            "chunks": chunks,
            "images": images
        }
```

### Multimodal RAG Engine Implementation

```python
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class MultimodalRAGEngine:
    def __init__(self):
        self.vector_store = MultimodalVectorStore()
        self.image_processor = ImageProcessor()

    async def ingest_document(
        self,
        text: str,
        images: List[Dict] = None,
        metadata: Dict = None
    ) -> Dict:
        """
        Ingest document with text and images

        Args:
            text: Document text content
            images: List of {"bytes": bytes, "caption": str}
            metadata: Additional metadata
        """
        try:
            # Process and store images
            image_ids = []
            for img in images or []:
                optimized = self.image_processor.process_image_to_base64(
                    img["bytes"]
                )

                image_id = self.vector_store.add_image(
                    caption=img.get("caption", "Laptop component image"),
                    base64_data=optimized["data"],
                    mime_type=optimized["mime_type"],
                    metadata=metadata or {}
                )
                image_ids.append(image_id)

            # Chunk text (simplified - use proper chunker in production)
            chunks = self._chunk_text(text)

            # Store text chunks with image links
            for chunk in chunks:
                self.vector_store.add_text_chunk(
                    text=chunk,
                    metadata=metadata or {},
                    image_ids=image_ids
                )

            return {
                "status": "success",
                "chunks_stored": len(chunks),
                "images_stored": len(image_ids)
            }

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise

    async def query(self, question: str, top_k: int = 5) -> Dict:
        """
        Query the multimodal RAG system

        Returns:
            {"answer": str, "images": [{"mime_type", "data"}]}
        """
        # Retrieve relevant chunks and images
        results = self.vector_store.retrieve_with_images(question, top_k)

        # Build context for LLM
        context = "\n\n".join([c["text"] for c in results["chunks"]])

        # Generate answer using LLM
        answer = await self._generate_answer(question, context)

        # Return in required format
        return {
            "answer": answer,
            "images": results["images"]
        }

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)

        return chunks
```

## Final Response Schema

```json
{
  "answer": "The RAM (Random Access Memory) in your Dell XPS 15 is responsible for temporary data storage while the computer is running. If you're experiencing slow performance or frequent crashes, it could indicate RAM issues. Here's what to check:\n\n1. **Check RAM usage** in Task Manager\n2. **Run Windows Memory Diagnostic**\n3. **Reseat RAM modules** if comfortable opening the laptop\n\nCommon failure symptoms include:\n- Random blue screens\n- System freezes\n- Beep codes on startup",
  "images": [
    {
      "mime_type": "image/jpeg",
      "data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAFAAUADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTE4NTo5KSEiJicuNDU2NzY3PDk/Q0NFR0dOT1NUVldYWVhZ2VnZ2hpanN0dXZ3eHl6g4SFhoeIiYqLjI2Oj4+Pk5KSk5OTk5ORkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGR/9k="
    }
  ]
}
```

## Error Handling

```python
class ImageProcessingError(Exception):
    """Raised when image processing fails"""
    pass

class RetrievalError(Exception):
    """Raised when retrieval fails"""
    pass

class GenerationError(Exception):
    """Raised when LLM generation fails"""
    pass

# In the RAG engine:
async def query(self, question: str) -> Dict:
    try:
        results = self.vector_store.retrieve_with_images(question)
        if not results["chunks"]:
            return {
                "answer": "I couldn't find relevant information for your query.",
                "images": []
            }

        answer = await self._generate_answer(question, context)
        return {"answer": answer, "images": results["images"]}

    except Exception as e:
        logger.error(f"Query failed: {e}")
        return {
            "answer": "I apologize, but I encountered an error processing your request.",
            "images": []
        }
```

## Production Considerations

1. **Image Storage**: Use Redis or S3 for base64 image storage in production
2. **Embedding Model**: Use `sentence-transformers/clip-ViT-B-32` for multimodal embeddings
3. **Vector DB**: ChromaDB for development, Qdrant or Weaviate for production
4. **Caching**: Cache optimized images to avoid reprocessing
5. **Rate Limiting**: Implement rate limiting for image downloads
6. **CDN**: Consider CDN for frequently accessed images (though base64 is preferred)

## Dependencies

```
Pillow>=10.0.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
google-generativeai>=0.3.0
```
