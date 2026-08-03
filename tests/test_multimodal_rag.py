#test_multimodal_rag.py
"""
Test Suite for Multimodal RAG Pipeline

Tests the complete multimodal RAG pipeline:
1. Image Processor - optimization and base64 conversion
2. Multimodal Vector Store - storing and retrieving text + images
3. Response Formatter - validating response format
4. Multimodal RAG Engine - end-to-end pipeline

All tests ensure images are returned as base64 data URIs, never as URLs.
"""
import pytest
import sys
import os
import io
import base64
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock external dependencies
sys.modules['chromadb'] = Mock()
sys.modules['chromadb.config'] = Mock()

from backend.app.image_processor import ImageProcessor, ImageProcessingError
from backend.app.response_formatter import ResponseFormatter
from backend.app.multimodal_vector_store import MultimodalVectorStore
from backend.app.multimodal_rag_engine import MultimodalRAGEngine, RetrievalError, GenerationError


class TestImageProcessor:
    """Test suite for Image Processor"""

    @pytest.fixture
    def sample_image_bytes(self):
        """Create a sample image as bytes"""
        img = Image.new('RGB', (1920, 1080), color='red')
        output = io.BytesIO()
        img.save(output, format='PNG')
        return output.getvalue()

    @pytest.fixture
    def small_image_bytes(self):
        """Create a small image as bytes"""
        img = Image.new('RGB', (100, 100), color='blue')
        output = io.BytesIO()
        img.save(output, format='PNG')
        return output.getvalue()

    def test_optimize_image_resizes_large_image(self, sample_image_bytes):
        """Test 1: Large image is resized to max 800px"""
        optimized_bytes, mime_type = ImageProcessor.optimize_image(sample_image_bytes)

        # Verify image was resized
        img = Image.open(io.BytesIO(optimized_bytes))
        assert max(img.size) <= 800
        assert mime_type == "image/jpeg"

    def test_optimize_image_maintains_aspect_ratio(self, sample_image_bytes):
        """Test 2: Aspect ratio is maintained after resize"""
        optimized_bytes, _ = ImageProcessor.optimize_image(sample_image_bytes)

        img = Image.open(io.BytesIO(optimized_bytes))
        # Original was 1920x1080 (16:9 ratio)
        # After resize to 800px max, should be 800x450 (maintaining 16:9)
        assert img.size[0] == 800
        assert img.size[1] == 450

    def test_optimize_image_compression(self, sample_image_bytes):
        """Test 3: Image is compressed to under 100KB"""
        optimized_bytes, _ = ImageProcessor.optimize_image(sample_image_bytes)

        # Should be under 100KB
        assert len(optimized_bytes) < 100 * 1024

    def test_optimize_image_small_image_unchanged(self, small_image_bytes):
        """Test 4: Small images are not upscaled"""
        optimized_bytes, _ = ImageProcessor.optimize_image(small_image_bytes)

        img = Image.open(io.BytesIO(optimized_bytes))
        # Small image should remain at original size (or close)
        assert img.size[0] == 100
        assert img.size[1] == 100

    def test_to_base64_data_uri_format(self, sample_image_bytes):
        """Test 5: Base64 data URI has correct format"""
        optimized_bytes, mime_type = ImageProcessor.optimize_image(sample_image_bytes)
        data_uri = ImageProcessor.to_base64_data_uri(optimized_bytes, mime_type)

        # Check format: data:image/jpeg;base64,...
        assert data_uri.startswith("data:image/jpeg;base64,")
        assert len(data_uri) > 50

    def test_process_image_to_base64_returns_correct_dict(self, sample_image_bytes):
        """Test 6: Full pipeline returns correct dict structure"""
        result = ImageProcessor.process_image_to_base64(sample_image_bytes)

        assert "mime_type" in result
        assert "data" in result
        assert result["mime_type"] == "image/jpeg"
        assert result["data"].startswith("data:image/jpeg;base64,")

    def test_optimize_image_invalid_bytes_raises_error(self):
        """Test 7: Invalid image bytes raises ImageProcessingError"""
        with pytest.raises(ImageProcessingError):
            ImageProcessor.optimize_image(b"not an image")

    def test_optimize_image_with_alpha_channel(self):
        """Test 8: RGBA image is converted to RGB"""
        img = Image.new('RGBA', (500, 500), color=(255, 0, 0, 128))
        output = io.BytesIO()
        img.save(output, format='PNG')
        rgba_bytes = output.getvalue()

        optimized_bytes, mime_type = ImageProcessor.optimize_image(rgba_bytes)

        # Should successfully convert to JPEG (no alpha)
        img_result = Image.open(io.BytesIO(optimized_bytes))
        assert img_result.mode == 'RGB'

    def test_optimize_image_webp_format(self, sample_image_bytes):
        """Test 9: WebP format option works"""
        optimized_bytes, mime_type = ImageProcessor.optimize_image(
            sample_image_bytes,
            output_format="WEBP"
        )

        assert mime_type == "image/webp"
        assert len(optimized_bytes) < 100 * 1024

    def test_get_image_info(self, sample_image_bytes):
        """Test 10: Get image info returns correct metadata"""
        info = ImageProcessor.get_image_info(sample_image_bytes)

        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["format"] == "PNG"
        assert info["size_bytes"] > 0


class TestResponseFormatter:
    """Test suite for Response Formatter"""

    def test_format_response_valid(self):
        """Test 1: Valid response is formatted correctly"""
        answer = "This is a test answer."
        images = [
            {
                "mime_type": "image/jpeg",
                "data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAFAAUADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTE4NTo5KSEiJicuNDU2NzY3PDk/Q0NFR0dOT1NUVldYWVhZ2VnZ2hpanN0dXZ3eHl6g4SFhoeIiYqLjI2Oj4+Pk5KSk5OTk5ORkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGR/9k="
            }
        ]

        response = ResponseFormatter.format_response(answer, images)

        assert response["answer"] == "This is a test answer."
        assert len(response["images"]) == 1
        assert response["images"][0]["mime_type"] == "image/jpeg"
        assert response["images"][0]["data"].startswith("data:image/jpeg;base64,")

    def test_format_response_empty_images(self):
        """Test 2: Empty images list is valid"""
        response = ResponseFormatter.format_response("Answer", [])

        assert response["answer"] == "Answer"
        assert response["images"] == []

    def test_format_response_no_images(self):
        """Test 3: No images parameter defaults to empty list"""
        response = ResponseFormatter.format_response("Answer")

        assert response["answer"] == "Answer"
        assert response["images"] == []

    def test_format_response_empty_answer(self):
        """Test 4: Empty answer gets default message"""
        response = ResponseFormatter.format_response("", [])

        assert "couldn't generate" in response["answer"].lower()

    def test_format_error_response(self):
        """Test 5: Error response is formatted correctly"""
        response = ResponseFormatter.format_error_response("Test error")

        assert "error occurred" in response["answer"]
        assert "Test error" in response["answer"]
        assert response["images"] == []

    def test_normalize_images_filters_invalid(self):
        """Test 6: Invalid images are filtered out"""
        images = [
            {"mime_type": "image/jpeg", "data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAFAAUADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTE4NTo5KSEiJicuNDU2NzY3PDk/Q0NFR0dOT1NUVldYWVhZ2VnZ2hpanN0dXZ3eHl6g4SFhoeIiYqLjI2Oj4+Pk5KSk5OTk5ORkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGR/9k="},
            {"mime_type": "image/jpeg"},  # Missing data
            "not a dict",
            {"data": "not a data uri"},  # Invalid format
        ]

        normalized = ResponseFormatter._normalize_images(images)

        assert len(normalized) == 1
        assert normalized[0]["mime_type"] == "image/jpeg"

    def test_validate_response_valid(self):
        """Test 7: Valid response passes validation"""
        response = {
            "answer": "Test answer",
            "images": [
                {
                    "mime_type": "image/jpeg",
                    "data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAFAAUADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTE4NTo5KSEiJicuNDU2NzY3PDk/Q0NFR0dOT1NUVldYWVhZ2VnZ2hpanN0dXZ3eHl6g4SFhoeIiYqLjI2Oj4+Pk5KSk5OTk5ORkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGR/9k="
                }
            ]
        }

        assert ResponseFormatter.validate_response(response) is True

    def test_validate_response_invalid(self):
        """Test 8: Invalid response fails validation"""
        # Missing images field
        assert ResponseFormatter.validate_response({"answer": "test"}) is False

        # Invalid image format
        response = {
            "answer": "test",
            "images": [{"mime_type": "image/jpeg", "data": "not a data uri"}]
        }
        assert ResponseFormatter.validate_response(response) is False

    def test_get_response_size(self):
        """Test 9: Response size calculation"""
        response = {
            "answer": "Test",
            "images": [
                {
                    "mime_type": "image/jpeg",
                    "data": "data:image/jpeg;base64,/9j/4AAQ="
                }
            ]
        }

        size = ResponseFormatter.get_response_size(response)

        assert "answer_bytes" in size
        assert "images_bytes" in size
        assert "total_bytes" in size
        assert "image_count" in size
        assert size["image_count"] == 1


class TestMultimodalVectorStore:
    """Test suite for Multimodal Vector Store"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mocked vector store"""
        with patch('backend.app.multimodal_vector_store.chromadb'):
            store = MultimodalVectorStore(persist_directory="./test_db")
            store.text_collection = Mock()
            store.image_collection = Mock()
            store.image_store = {}
            return store

    def test_add_text_chunk(self, mock_vector_store):
        """Test 1: Text chunk is stored with image links"""
        mock_vector_store.text_collection.add.return_value = None

        chunk_id = mock_vector_store.add_text_chunk(
            text="Test chunk",
            metadata={"source": "test"},
            image_ids=["img1", "img2"]
        )

        assert chunk_id is not None
        mock_vector_store.text_collection.add.assert_called_once()

        # Verify image_ids were added to metadata
        call_args = mock_vector_store.text_collection.add.call_args
        metadata = call_args.kwargs['metadatas'][0]
        assert "image_ids" in metadata
        assert json.loads(metadata["image_ids"]) == ["img1", "img2"]

    def test_add_image_stores_base64(self, mock_vector_store):
        """Test 2: Image base64 data is stored separately"""
        mock_vector_store.image_collection.add.return_value = None

        image_id = mock_vector_store.add_image(
            caption="RAM module",
            base64_data="data:image/jpeg;base64,/9j/4AAQ=",
            mime_type="image/jpeg",
            metadata={"source": "test"}
        )

        assert image_id is not None
        assert image_id in mock_vector_store.image_store
        assert mock_vector_store.image_store[image_id]["data"] == "data:image/jpeg;base64,/9j/4AAQ="
        assert mock_vector_store.image_store[image_id]["mime_type"] == "image/jpeg"

    def test_retrieve_with_images(self, mock_vector_store):
        """Test 3: Retrieval returns chunks with linked images"""
        # Setup mock text results
        mock_vector_store.text_collection.query.return_value = {
            'documents': [["Test chunk 1", "Test chunk 2"]],
            'metadatas': [[
                {"image_ids": json.dumps(["img1"]), "chunk_id": "c1"},
                {"image_ids": json.dumps(["img2"]), "chunk_id": "c2"}
            ]],
            'distances': [[0.1, 0.2]]
        }

        # Add images to store
        mock_vector_store.image_store["img1"] = {
            "mime_type": "image/jpeg",
            "data": "data:image/jpeg;base64,/9j/4AAQ="
        }
        mock_vector_store.image_store["img2"] = {
            "mime_type": "image/jpeg",
            "data": "data:image/jpeg;base64,/9j/4BBQ="
        }

        results = mock_vector_store.retrieve_with_images("test query")

        assert len(results["chunks"]) == 2
        assert len(results["images"]) == 2
        assert results["images"][0]["mime_type"] == "image/jpeg"
        assert results["images"][0]["data"].startswith("data:image/jpeg;base64,")

    def test_retrieve_no_results(self, mock_vector_store):
        """Test 4: Empty results when no matches"""
        mock_vector_store.text_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        results = mock_vector_store.retrieve_with_images("no match query")

        assert results["chunks"] == []
        assert results["images"] == []

    def test_search_images_by_caption(self, mock_vector_store):
        """Test 5: Image search by caption works"""
        mock_vector_store.image_collection.query.return_value = {
            'documents': [["RAM module image"]],
            'metadatas': [[{"image_id": "img1"}]],
            'distances': [[0.1]]
        }

        mock_vector_store.image_store["img1"] = {
            "mime_type": "image/jpeg",
            "data": "data:image/jpeg;base64,/9j/4AAQ="
        }

        images = mock_vector_store.search_images_by_caption("RAM", top_k=1)

        assert len(images) == 1
        assert images[0]["data"].startswith("data:image/jpeg;base64,")

    def test_get_stats(self, mock_vector_store):
        """Test 6: Stats return correct counts"""
        mock_vector_store.text_collection.count.return_value = 10
        mock_vector_store.image_collection.count.return_value = 5
        mock_vector_store.image_store = {"img1": {}, "img2": {}}

        stats = mock_vector_store.get_stats()

        assert stats["text_chunks"] == 10
        assert stats["image_captions"] == 5
        assert stats["images_in_store"] == 2


class TestMultimodalRAGEngine:
    """Test suite for Multimodal RAG Engine"""

    @pytest.fixture
    def mock_rag_engine(self):
        """Create a mocked RAG engine"""
        with patch('backend.app.multimodal_rag_engine.get_multimodal_vector_store'), \
             patch('backend.app.multimodal_rag_engine.get_image_processor'):

            engine = MultimodalRAGEngine()
            engine.vector_store = Mock()
            engine.image_processor = Mock()
            engine._llm_service = Mock()
            return engine

    @pytest.mark.asyncio
    async def test_ingest_document_with_images(self, mock_rag_engine):
        """Test 1: Document with images is ingested correctly"""
        # Mock image processing
        mock_rag_engine.image_processor.process_image_to_base64.return_value = {
            "mime_type": "image/jpeg",
            "data": "data:image/jpeg;base64,/9j/4AAQ="
        }

        # Mock vector store
        mock_rag_engine.vector_store.add_image.return_value = "img1"
        mock_rag_engine.vector_store.add_text_chunk.return_value = "chunk1"

        result = await mock_rag_engine.ingest_document(
            text="This is a test document about laptop RAM.",
            images=[{"bytes": b"fake_image", "caption": "RAM module"}],
            metadata={"source": "test"}
        )

        assert result["status"] == "success"
        assert result["images_stored"] == 1
        assert result["chunks_stored"] > 0

    @pytest.mark.asyncio
    async def test_query_returns_answer_and_images(self, mock_rag_engine):
        """Test 2: Query returns both answer and images"""
        # Mock retrieval
        mock_rag_engine.vector_store.retrieve_with_images.return_value = {
            "chunks": [{"text": "RAM info", "metadata": {}, "score": 0.1}],
            "images": [{"mime_type": "image/jpeg", "data": "data:image/jpeg;base64,/9j/4AAQ="}]
        }

        # Mock LLM generation
        mock_response = Mock()
        mock_response.text = "RAM is Random Access Memory."
        mock_rag_engine.llm_service.client.models.generate_content.return_value = mock_response

        result = await mock_rag_engine.query("What is RAM?")

        assert "answer" in result
        assert "images" in result
        assert len(result["answer"]) > 0
        assert len(result["images"]) == 1
        assert result["images"][0]["data"].startswith("data:image/jpeg;base64,")

    @pytest.mark.asyncio
    async def test_query_no_results(self, mock_rag_engine):
        """Test 3: Query with no results returns empty response"""
        mock_rag_engine.vector_store.retrieve_with_images.return_value = {
            "chunks": [],
            "images": []
        }

        result = await mock_rag_engine.query("nonexistent topic")

        assert "couldn't find" in result["answer"].lower()
        assert result["images"] == []

    @pytest.mark.asyncio
    async def test_query_error_handling(self, mock_rag_engine):
        """Test 4: Query handles errors gracefully"""
        mock_rag_engine.vector_store.retrieve_with_images.side_effect = Exception("DB error")

        result = await mock_rag_engine.query("test")

        assert "error" in result["answer"].lower()
        assert result["images"] == []

    @pytest.mark.asyncio
    async def test_explain_component_with_images(self, mock_rag_engine):
        """Test 5: Component explanation returns images"""
        # Mock image search
        mock_rag_engine.vector_store.search_images_by_caption.return_value = [
            {"mime_type": "image/jpeg", "data": "data:image/jpeg;base64,/9j/4AAQ="}
        ]

        # Mock LLM
        mock_response = Mock()
        mock_response.text = "SSD is a Solid State Drive."
        mock_rag_engine.llm_service.client.models.generate_content.return_value = mock_response

        result = await mock_rag_engine.explain_component_with_images("SSD")

        assert "answer" in result
        assert "images" in result
        assert len(result["images"]) == 1
        assert "SSD" in result["answer"]

    def test_chunk_text(self, mock_rag_engine):
        """Test 6: Text is chunked correctly"""
        text = "word " * 1000  # 1000 words
        chunks = mock_rag_engine._chunk_text(text, chunk_size=500)

        assert len(chunks) == 2
        assert len(chunks[0].split()) == 500

    def test_build_context(self, mock_rag_engine):
        """Test 7: Context is built from chunks"""
        chunks = [
            {"text": "Chunk 1", "metadata": {"source": "url1", "section": "ram"}},
            {"text": "Chunk 2", "metadata": {"source": "url2", "section": "ssd"}}
        ]

        context = mock_rag_engine._build_context(chunks)

        assert "Chunk 1" in context
        assert "Chunk 2" in context
        assert "url1" in context
        assert "url2" in context


class TestEndToEndPipeline:
    """End-to-end integration tests"""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_mock_data(self):
        """Test complete pipeline from ingestion to query"""
        # This test uses mocks but verifies the full flow

        with patch('backend.app.multimodal_rag_engine.get_multimodal_vector_store') as mock_vs, \
             patch('backend.app.multimodal_rag_engine.get_image_processor') as mock_ip:

            # Setup mocks
            mock_vector_store = Mock()
            mock_vs.return_value = mock_vector_store

            mock_image_processor = Mock()
            mock_ip.return_value = mock_image_processor

            # Create engine
            engine = MultimodalRAGEngine()
            engine._llm_service = Mock()
            mock_llm = engine._llm_service

            # Test ingestion
            mock_image_processor.process_image_to_base64.return_value = {
                "mime_type": "image/jpeg",
                "data": "data:image/jpeg;base64,/9j/4AAQ="
            }
            mock_vector_store.add_image.return_value = "img1"
            mock_vector_store.add_text_chunk.return_value = "chunk1"

            ingest_result = await engine.ingest_document(
                text="Laptop RAM information",
                images=[{"bytes": b"fake", "caption": "RAM"}]
            )

            assert ingest_result["status"] == "success"

            # Test query
            mock_vector_store.retrieve_with_images.return_value = {
                "chunks": [{"text": "RAM info", "metadata": {}, "score": 0.1}],
                "images": [{"mime_type": "image/jpeg", "data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAFAAUADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTE4NTo5KSEiJicuNDU2NzY3PDk/Q0NFR0dOT1NUVldYWVhZ2VnZ2hpanN0dXZ3eHl6g4SFhoeIiYqLjI2Oj4+Pk5KSk5OTk5ORkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGR/9k="}]
            }

            mock_response = Mock()
            mock_response.text = "RAM is memory."
            mock_llm.client.models.generate_content.return_value = mock_response

            query_result = await engine.query("What is RAM?")

            # Verify response format
            assert "answer" in query_result
            assert "images" in query_result
            assert isinstance(query_result["answer"], str)
            assert isinstance(query_result["images"], list)

            # Verify images are base64 data URIs (not URLs)
            for img in query_result["images"]:
                assert img["data"].startswith("data:image/")
                assert ";base64," in img["data"]
                assert "mime_type" in img

            # Validate with ResponseFormatter
            assert ResponseFormatter.validate_response(query_result) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])