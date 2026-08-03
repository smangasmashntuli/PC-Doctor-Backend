#image_processor.py
"""
Image Processor Module for Multimodal RAG Pipeline

Handles image optimization, compression, and base64 encoding.
All images are returned as base64 data URIs, never as external URLs.
"""
from PIL import Image
import io
import base64
import logging
import requests
from typing import Optional, Tuple, Dict, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class ImageProcessingError(Exception):
    """Raised when image processing fails"""
    pass


class ImageProcessor:
    """
    Image optimization and base64 conversion utility.

    Optimization Strategy:
    - Resize: Max 800px on longest side, maintain aspect ratio
    - Format: JPEG (quality 70) or WebP (quality 75)
    - Size target: < 100KB per image
    - Output: Base64 data URI (data:image/jpeg;base64,...)
    """

    MAX_DIMENSION = 800
    JPEG_QUALITY = 70
    WEBP_QUALITY = 75
    MIN_DIMENSION = 100
    MAX_FILE_SIZE_BYTES = 100 * 1024  # 100KB target

    @classmethod
    def optimize_image(
        cls,
        image_bytes: bytes,
        max_dimension: int = None,
        quality: int = None,
        output_format: str = "JPEG"
    ) -> Tuple[bytes, str]:
        """
        Optimize image: resize, compress, convert format.

        Args:
            image_bytes: Raw image bytes
            max_dimension: Max width/height in pixels (default: 800)
            quality: Compression quality 60-75 (default: 70)
            output_format: "JPEG" or "WEBP"

        Returns:
            Tuple of (optimized_bytes, mime_type)

        Raises:
            ImageProcessingError: If optimization fails
        """
        max_dimension = max_dimension or cls.MAX_DIMENSION
        quality = quality or cls.JPEG_QUALITY

        try:
            # Open image from bytes
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB (remove alpha channel for JPEG)
            if img.mode in ('RGBA', 'LA', 'P', 'L'):
                img = img.convert('RGB')

            # Get original dimensions
            width, height = img.size

            # Resize maintaining aspect ratio if exceeds max_dimension
            if max(width, height) > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int((height * max_dimension) / width)
                else:
                    new_height = max_dimension
                    new_width = int((width * max_dimension) / height)

                # Ensure minimum dimension
                new_width = max(new_width, cls.MIN_DIMENSION)
                new_height = max(new_height, cls.MIN_DIMENSION)

                img = img.resize(
                    (new_width, new_height),
                    Image.Resampling.LANCZOS
                )
                logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")

            # Save optimized image
            output = io.BytesIO()

            if output_format.upper() == "WEBP":
                img.save(output, format="WEBP", quality=quality, method=6)
                mime_type = "image/webp"
            else:
                img.save(output, format="JPEG", quality=quality, optimize=True)
                mime_type = "image/jpeg"

            optimized_bytes = output.getvalue()

            # Check if we need to reduce quality further
            if len(optimized_bytes) > cls.MAX_FILE_SIZE_BYTES:
                logger.debug(f"Image still {len(optimized_bytes)} bytes, reducing quality")
                # Try lower quality
                for reduced_quality in [quality - 10, quality - 20, 50]:
                    if reduced_quality < 30:
                        break
                    output = io.BytesIO()
                    img.save(output, format="JPEG", quality=reduced_quality, optimize=True)
                    optimized_bytes = output.getvalue()
                    if len(optimized_bytes) <= cls.MAX_FILE_SIZE_BYTES:
                        logger.debug(f"Reduced to {len(optimized_bytes)} bytes at quality {reduced_quality}")
                        break

            logger.info(f"Optimized image: {len(image_bytes)} -> {len(optimized_bytes)} bytes")
            return optimized_bytes, mime_type

        except Exception as e:
            logger.error(f"Image optimization failed: {e}")
            raise ImageProcessingError(f"Failed to optimize image: {e}")

    @classmethod
    def to_base64_data_uri(
        cls,
        image_bytes: bytes,
        mime_type: str = "image/jpeg"
    ) -> str:
        """
        Convert image bytes to base64 data URI.

        Args:
            image_bytes: Raw image bytes
            mime_type: MIME type (e.g., "image/jpeg")

        Returns:
            Base64 data URI string (data:image/jpeg;base64,...)
        """
        b64 = base64.b64encode(image_bytes).decode('utf-8')
        return f"data:{mime_type};base64,{b64}"

    @classmethod
    def process_image_to_base64(
        cls,
        image_bytes: bytes,
        max_dimension: int = None,
        quality: int = None,
        output_format: str = "JPEG"
    ) -> Dict[str, str]:
        """
        Full pipeline: optimize + convert to base64 data URI.

        Args:
            image_bytes: Raw image bytes
            max_dimension: Max width/height in pixels
            quality: Compression quality
            output_format: "JPEG" or "WEBP"

        Returns:
            {"mime_type": str, "data": str} where data is base64 data URI

        Raises:
            ImageProcessingError: If processing fails
        """
        try:
            optimized_bytes, mime_type = cls.optimize_image(
                image_bytes,
                max_dimension,
                quality,
                output_format
            )

            data_uri = cls.to_base64_data_uri(optimized_bytes, mime_type)

            return {
                "mime_type": mime_type,
                "data": data_uri
            }

        except ImageProcessingError:
            raise
        except Exception as e:
            logger.error(f"Full image processing failed: {e}")
            raise ImageProcessingError(f"Failed to process image: {e}")

    @classmethod
    def download_and_optimize(
        cls,
        url: str,
        max_dimension: int = None,
        quality: int = None
    ) -> Dict[str, str]:
        """
        Download image from URL and optimize to base64.

        Args:
            url: Image URL to download
            max_dimension: Max width/height in pixels
            quality: Compression quality

        Returns:
            {"mime_type": str, "data": str} with base64 data URI

        Raises:
            ImageProcessingError: If download or processing fails
        """
        try:
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ImageProcessingError(f"Invalid URL: {url}")

            # Download image
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            # Process the downloaded image
            return cls.process_image_to_base64(
                response.content,
                max_dimension,
                quality
            )

        except requests.RequestException as e:
            logger.error(f"Failed to download image from {url}: {e}")
            raise ImageProcessingError(f"Download failed: {e}")
        except ImageProcessingError:
            raise
        except Exception as e:
            logger.error(f"Download and optimize failed for {url}: {e}")
            raise ImageProcessingError(f"Failed to download and optimize: {e}")

    @classmethod
    def extract_images_from_html(
        cls,
        html_content: str,
        base_url: str = None
    ) -> List[str]:
        """
        Extract image URLs from HTML content.

        Args:
            html_content: HTML string
            base_url: Base URL for relative paths

        Returns:
            List of image URLs
        """
        from bs4 import BeautifulSoup

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            img_tags = soup.find_all('img')

            image_urls = []
            for img in img_tags:
                src = img.get('src') or img.get('data-src')
                if src:
                    if base_url and not src.startswith(('http://', 'https://')):
                        # Handle relative URLs
                        if src.startswith('//'):
                            src = f"https:{src}"
                        elif src.startswith('/'):
                            parsed = urlparse(base_url)
                            src = f"{parsed.scheme}://{parsed.netloc}{src}"
                        else:
                            src = f"{base_url.rstrip('/')}/{src}"
                    image_urls.append(src)

            return image_urls

        except Exception as e:
            logger.error(f"Failed to extract images from HTML: {e}")
            return []

    @classmethod
    def create_placeholder_image(
        cls,
        text: str = "No Image",
        width: int = 400,
        height: int = 300
    ) -> Dict[str, str]:
        """
        Create a placeholder image as base64 data URI.

        Args:
            text: Text to display on placeholder
            width: Image width
            height: Image height

        Returns:
            {"mime_type": str, "data": str} with base64 data URI
        """
        try:
            from PIL import ImageDraw, ImageFont

            img = Image.new('RGB', (width, height), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)

            # Try to use default font
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

            # Calculate text position (center)
            if font:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                text_width = len(text) * 6
                text_height = 10

            x = (width - text_width) // 2
            y = (height - text_height) // 2

            draw.text((x, y), text, fill=(100, 100, 100), font=font)

            return cls.process_image_to_base64(
                cls._image_to_bytes(img)
            )

        except Exception as e:
            logger.error(f"Failed to create placeholder: {e}")
            # Return minimal valid base64
            return {
                "mime_type": "image/jpeg",
                "data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAFAAUADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTE4NTo5KSEiJicuNDU2NzY3PDk/Q0NFR0dOT1NUVldYWVhZ2VnZ2hpanN0dXZ3eHl6g4SFhoeIiYqLjI2Oj4+Pk5KSk5OTk5ORkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGRkZGR/9k="
            }

    @staticmethod
    def _image_to_bytes(img: Image.Image, format: str = "JPEG") -> bytes:
        """Convert PIL Image to bytes"""
        output = io.BytesIO()
        img.save(output, format=format)
        return output.getvalue()

    @classmethod
    def get_image_info(cls, image_bytes: bytes) -> Dict:
        """
        Get image information without full processing.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Dict with width, height, format, mode
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "size_bytes": len(image_bytes)
            }
        except Exception as e:
            logger.error(f"Failed to get image info: {e}")
            return {"error": str(e)}


# Global instance
_image_processor_instance = None


def get_image_processor() -> ImageProcessor:
    """Get or create the global image processor instance"""
    global _image_processor_instance
    if _image_processor_instance is None:
        _image_processor_instance = ImageProcessor()
    return _image_processor_instance