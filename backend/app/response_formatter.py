#response_formatter.py
"""
Response Formatter Module

Ensures consistent output format for the multimodal RAG pipeline.
All responses follow this exact schema:

{
    "answer": "Generated textual response here...",
    "images": [
        {
            "mime_type": "image/jpeg",
            "data": "data:image/jpeg;base64,/9j/4AAQ..."
        }
    ]
}

Key Rules:
- Images are ALWAYS base64 data URIs, never external URLs
- mime_type is always present
- data field contains the full data URI (data:image/jpeg;base64,...)
- Empty images list is valid if no images are relevant
"""
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    Formats responses to ensure consistent output schema.

    This class validates and normalizes all responses to match
    the required output format.
    """

    REQUIRED_FIELDS = {"answer", "images"}
    IMAGE_REQUIRED_FIELDS = {"mime_type", "data"}

    @classmethod
    def format_response(
        cls,
        answer: str,
        images: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Format a response with answer and images.

        Args:
            answer: Generated text answer
            images: List of {"mime_type": str, "data": str} dicts

        Returns:
            Formatted response dict

        Example:
            {
                "answer": "The RAM is responsible for...",
                "images": [
                    {
                        "mime_type": "image/jpeg",
                        "data": "data:image/jpeg;base64,/9j/4AAQ..."
                    }
                ]
            }
        """
        # Validate answer
        if not answer or not isinstance(answer, str):
            answer = "I apologize, but I couldn't generate a response."

        # Validate and normalize images
        normalized_images = cls._normalize_images(images or [])

        response = {
            "answer": answer.strip(),
            "images": normalized_images
        }

        logger.debug(f"Formatted response with {len(normalized_images)} images")
        return response

    @classmethod
    def format_error_response(
        cls,
        error_message: str,
        images: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Format an error response.

        Args:
            error_message: Error description
            images: Optional images to include

        Returns:
            Formatted error response
        """
        return cls.format_response(
            answer=f"I apologize, but an error occurred: {error_message}",
            images=images
        )

    @classmethod
    def format_empty_response(cls) -> Dict[str, Any]:
        """
        Format an empty response when no results found.

        Returns:
            Empty response with no images
        """
        return cls.format_response(
            answer="I couldn't find relevant information for your query.",
            images=[]
        )

    @classmethod
    def _normalize_images(
        cls,
        images: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Normalize image list to ensure consistent format.

        Args:
            images: List of image dicts

        Returns:
            Normalized list of {"mime_type": str, "data": str}
        """
        normalized = []

        for img in images:
            if not isinstance(img, dict):
                logger.warning(f"Skipping non-dict image: {type(img)}")
                continue

            # Ensure required fields exist
            mime_type = img.get("mime_type", "image/jpeg")
            data = img.get("data")

            if not data:
                logger.warning("Skipping image with no data field")
                continue

            # Validate data is a base64 data URI
            if not cls._is_valid_data_uri(data):
                logger.warning(f"Invalid data URI format, skipping image")
                continue

            normalized.append({
                "mime_type": mime_type,
                "data": data
            })

        return normalized

    @staticmethod
    def _is_valid_data_uri(data: str) -> bool:
        """
        Validate that a string is a proper base64 data URI.

        Args:
            data: String to validate

        Returns:
            True if valid data URI format
        """
        if not isinstance(data, str):
            return False

        # Check for data URI prefix
        if not data.startswith("data:"):
            return False

        # Check for base64 marker
        if ";base64," not in data:
            return False

        # Check for minimum length (header + some data)
        if len(data) < 50:
            return False

        return True

    @classmethod
    def validate_response(cls, response: Dict) -> bool:
        """
        Validate that a response matches the required schema.

        Args:
            response: Response dict to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(response, dict):
            return False

        # Check required fields
        if not cls.REQUIRED_FIELDS.issubset(response.keys()):
            return False

        # Check answer is a string
        if not isinstance(response["answer"], str):
            return False

        # Check images is a list
        if not isinstance(response["images"], list):
            return False

        # Validate each image
        for img in response["images"]:
            if not isinstance(img, dict):
                return False
            if not cls.IMAGE_REQUIRED_FIELDS.issubset(img.keys()):
                return False
            if not cls._is_valid_data_uri(img["data"]):
                return False

        return True

    @classmethod
    def get_response_size(cls, response: Dict) -> Dict[str, int]:
        """
        Get size information about a response.

        Args:
            response: Formatted response

        Returns:
            Dict with size metrics
        """
        answer_size = len(response.get("answer", "").encode('utf-8'))

        images_size = 0
        for img in response.get("images", []):
            images_size += len(img.get("data", "").encode('utf-8'))

        return {
            "answer_bytes": answer_size,
            "images_bytes": images_size,
            "total_bytes": answer_size + images_size,
            "image_count": len(response.get("images", []))
        }


# Global instance
_response_formatter_instance = None


def get_response_formatter() -> ResponseFormatter:
    """Get or create the global response formatter instance"""
    global _response_formatter_instance
    if _response_formatter_instance is None:
        _response_formatter_instance = ResponseFormatter()
    return _response_formatter_instance