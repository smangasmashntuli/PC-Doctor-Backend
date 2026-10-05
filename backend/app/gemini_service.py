#gemini_service.py
import os
import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import base64

from google import genai
from google.genai import types

from backend.app.vector_store import get_vector_store
from backend.app.models import LaptopSpecs, LaptopSetup, ChatMessage, DiagnosticSession
from backend.app.database import SessionLocal

logger = logging.getLogger(__name__)

# Ordered fallbacks used when the primary chat model is unavailable.
# gemini-1.5-flash was retired (404), so never add it back here.
FALLBACK_TEXT_MODELS = (
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash-lite",
)


class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_FLASH_API_KEY")
        configured_image_key = os.getenv("GEMINI_IMAGEN_API_KEY") or os.getenv("GEMINI_IMAGE_API_KEY")
        self.image_api_key = (
            configured_image_key
            if configured_image_key and not configured_image_key.startswith("your_")
            else self.api_key
        )
        if not self.api_key:
            raise ValueError("GEMINI_FLASH_API_KEY not found in environment variables")
        
        self.client = genai.Client(api_key=self.api_key)
        # gemini-1.5-flash was retired and now returns 404 NOT_FOUND, which
        # silently broke laptop spec enrichment (all fields stayed None and
        # the UI rendered "N/A"). Default to a model that is actually served.
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        # Image-capable model used with generate_content + response_modalities.
        self.image_model_name = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
        # Legacy Imagen model, kept only as a secondary attempt.
        self.imagen_model_name = os.getenv("GEMINI_IMAGEN_MODEL", "imagen-3.0-generate-002")
        # Image-model calls are quota-limited to 0 requests/day on the free
        # tier, so we degrade to a neutral laptop photo instead of leaving the
        # card blank. Set LAPTOP_FALLBACK_IMAGE= (empty) to disable this.
        self.fallback_image_url = os.getenv(
            "LAPTOP_FALLBACK_IMAGE",
            "https://images.unsplash.com/photo-1593642632823-8f785ba67e45"
            "?auto=format&fit=crop&w=900&q=80",
        )
        self.vector_store = get_vector_store()
    
    async def generate_troubleshooting_response(
        self,
        user_query: str,
        laptop_specs: Optional[LaptopSpecs] = None,
        laptop_setup: Optional[LaptopSetup] = None,
        rag_context_chunks: Optional[List[Dict]] = None,
        chat_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate empathetic, non-technical troubleshooting response
        
        Args:
            user_query: User's question/issue
            laptop_specs: Laptop specifications from database
            laptop_setup: Laptop setup (brand/model)
            rag_context_chunks: Relevant documentation chunks from vector DB
            chat_history: Previous conversation messages
        
        Returns:
            AI-generated response text
        """
        try:
            # Build system prompt
            system_prompt = self._build_system_prompt(laptop_specs, laptop_setup, rag_context_chunks)
            
            # Build conversation history
            conversation = self._build_conversation(chat_history, user_query, system_prompt)
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=conversation,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                    top_p=0.95,
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again or contact support if the issue persists."

    async def generate_laptop_image(self, brand: str, model_name: str) -> str:
        """Generate a representative laptop image and return a data URI.

        Primary path uses an image-capable Gemini model through
        ``generate_content`` with ``response_modalities=["IMAGE", "TEXT"]``.
        The legacy Imagen ``generate_images`` endpoint is attempted as a
        fallback for accounts/plans where it is still enabled, because in
        Gemini Developer API mode it raises
        "This method is only supported in Gemini Enterprise Agent Platform mode".
        """
        if not self.image_api_key:
            if self.fallback_image_url:
                logger.warning("No image API key configured; using fallback laptop image.")
                return self.fallback_image_url
            raise ValueError("GEMINI_IMAGEN_API_KEY not found in environment variables")

        prompt = (
            f"Create a clean, realistic studio product render of a {brand} {model_name} laptop. "
            "Show the closed laptop at a three-quarter angle on a neutral background, with accurate materials, "
            "no text, no watermark, and no people."
        )

        errors = []

        # 1) Image-capable Gemini model via generate_content
        try:
            image_client = genai.Client(api_key=self.image_api_key)
            response = image_client.models.generate_content(
                model=self.image_model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )
            image_url = self._extract_image_url(response)
            if image_url:
                return image_url
            errors.append(f"{self.image_model_name} returned no image data")
        except Exception as e:
            logger.warning(f"Gemini image generation via {self.image_model_name} failed: {e}")
            errors.append(f"{self.image_model_name}: {e}")

        # 2) Legacy Imagen endpoint
        try:
            image_client = genai.Client(api_key=self.image_api_key)
            response = image_client.models.generate_images(
                model=self.imagen_model_name,
                prompt=prompt,
            )
            image_url = self._extract_image_url(response)
            if image_url:
                return image_url
            errors.append(f"{self.imagen_model_name} returned no image data")
        except Exception as e:
            logger.warning(f"Imagen image generation failed: {e}")
            errors.append(f"{self.imagen_model_name}: {e}")

        detail = " | ".join(errors)
        if self.fallback_image_url:
            logger.warning(
                f"AI laptop image generation unavailable ({detail}). "
                f"Using fallback image: {self.fallback_image_url}"
            )
            return self.fallback_image_url

        raise ValueError("Laptop image generation failed -> " + detail)

    async def enrich_laptop_specs(self, brand: str, model_name: str, current_specs: Dict[str, Any]) -> Dict[str, Any]:
        """Fill missing specification fields for a specific laptop model.

        This is the primary source of specs whenever web scraping returns
        nothing (DuckDuckGo frequently answers HTTP 202 with no results), so
        it retries transient errors and walks a list of known-good models
        instead of giving up on the first failure.
        """
        prompt = f"""
Return verified public specifications for the exact laptop: {brand} {model_name}.
Use your knowledge of this exact model and only fill fields that are reasonably known.
Return JSON only with these keys: cpu, gpu, ram, storage, display, ports, os, known_issues.
Use strings for cpu, gpu, ram, storage, display, and os. Use arrays of strings for ports and known_issues.
Do not include markdown or explanatory text.

"""
        models_to_try = []
        for model in (self.model_name, *FALLBACK_TEXT_MODELS):
            if model and model not in models_to_try:
                models_to_try.append(model)

        errors = []
        for model in models_to_try:
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.1,
                            max_output_tokens=768,
                            response_mime_type="application/json",
                        ),
                    )
                    parsed = self._parse_json_object(response.text)
                    if not isinstance(parsed, dict):
                        errors.append(f"{model}: response was not a JSON object")
                        break

                    enriched = self._filter_spec_fields(parsed)
                    if enriched:
                        return enriched
                    errors.append(f"{model}: no usable spec fields returned")
                    break
                except Exception as e:
                    errors.append(f"{model}: {e}")
                    logger.warning(
                        f"Spec enrichment attempt {attempt + 1} on {model} failed: {e}"
                    )
                    # Retry the same model once on transient failures
                    # (503 UNAVAILABLE / 429 RESOURCE_EXHAUSTED).
                    if attempt == 0:
                        await asyncio.sleep(1.5)

        logger.warning("Laptop spec enrichment skipped -> " + " | ".join(errors))
        return {}

    @staticmethod
    def _parse_json_object(raw_text: Optional[str]) -> Optional[Dict[str, Any]]:
        """Parse a JSON object out of a model response, tolerating fences."""
        if not raw_text:
            return None
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Last resort: pull the outermost {...} block.
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    return None
            return None

    @staticmethod
    def _filter_spec_fields(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only well-typed, non-empty spec fields."""
        enriched = {}
        for field in ("cpu", "gpu", "ram", "storage", "display", "ports", "os", "known_issues"):
            value = parsed.get(field)
            if field in ("ports", "known_issues"):
                if isinstance(value, list):
                    cleaned = [str(item).strip() for item in value if str(item).strip()]
                    if cleaned:
                        enriched[field] = cleaned
            elif isinstance(value, str) and value.strip():
                enriched[field] = value.strip()
        return enriched

    async def explain_component(
        self,
        component_name: str,
        laptop_brand: str,
        laptop_model: str,
    ) -> str:
        """Explain a laptop component in plain language for non-technical users."""
        prompt = (
            f"You are PC-Doctor-AI. Explain the {component_name} in a {laptop_brand} {laptop_model} in simple, everyday language. "
            "Do not use technical jargon unless you immediately define it. Include: what it does, common failure symptoms, and safe handling tips. "
            "Keep the answer short, calm, and easy to understand."
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=256,
                top_p=0.9,
            )
        )
        return response.text.strip()
    
    def _build_system_prompt(
        self,
        laptop_specs: Optional[LaptopSpecs],
        laptop_setup: Optional[LaptopSetup],
        rag_context: Optional[List[Dict]]
    ) -> str:
        """Build system prompt with laptop context"""
        
        prompt = """You are PC-Doctor-AI, an empathetic and patient laptop troubleshooting assistant designed for non-technical users.

**Your Personality:**
- Friendly, encouraging, and never condescending
- Use simple, everyday language (no jargon unless absolutely necessary)
- Break down complex steps into numbered, easy-to-follow instructions
- Always explain WHY each step helps
- Be honest about limitations - if something is too risky, say so

**Response Format:**
- Use numbered steps (1, 2, 3...) for all troubleshooting
- Keep each step short and clear
- Use bold text for important warnings
- End with a follow-up question to check if the issue is resolved

**Safety Rules:**
- NEVER suggest opening the laptop case unless absolutely necessary
- ALWAYS warn about battery/power risks first
- If you're unsure, recommend professional help
- Never suggest repairs that could cause data loss without warning

"""
        
        # Add laptop-specific context
        if laptop_setup and laptop_specs:
            prompt += f"\n**User's Laptop:**\n"
            prompt += f"- Brand: {laptop_setup.brand}\n"
            prompt += f"- Model: {laptop_setup.model}\n"
            
            if laptop_specs.cpu:
                prompt += f"- Processor: {laptop_specs.cpu}\n"
            if laptop_specs.ram:
                prompt += f"- Memory: {laptop_specs.ram}\n"
            if laptop_specs.storage:
                prompt += f"- Storage: {laptop_specs.storage}\n"
            if laptop_specs.gpu:
                prompt += f"- Graphics: {laptop_specs.gpu}\n"
            if laptop_specs.known_issues:
                import json
                known_issues = json.loads(laptop_specs.known_issues)
                if known_issues:
                    prompt += f"\n**Known Issues for this model:**\n"
                    for issue in known_issues[:3]:
                        prompt += f"- {issue}\n"
        
        # Add RAG context
        if rag_context and len(rag_context) > 0:
            prompt += f"\n**Technical Documentation:**\n"
            for i, chunk in enumerate(rag_context[:3], 1):
                prompt += f"{i}. {chunk.get('text', '')}\n"
                if chunk.get('metadata', {}).get('source_urls'):
                    import json
                    sources = json.loads(chunk['metadata']['source_urls'])
                    if sources:
                        prompt += f"   Source: {sources[0]}\n"
        
        prompt += "\n**Remember:** You're helping a non-technical user. Be patient, clear, and always prioritize safety over quick fixes.\n"
        
        return prompt
    
    def _build_conversation(
        self,
        chat_history: Optional[List[Dict]],
        user_query: str,
        system_prompt: str
    ) -> List[Dict]:
        """Build conversation for Gemini API"""
        
        conversation = []
        
        # Add system prompt as first message
        conversation.append({
            "role": "user",
            "parts": [system_prompt]
        })
        
        # Add chat history
        if chat_history:
            for msg in chat_history[-10:]:  # Last 10 messages for context
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                conversation.append({
                    "role": role,
                    "parts": [content]
                })
        
        # Add current user query
        conversation.append({
            "role": "user",
            "parts": [user_query]
        })
        
        return conversation

    def _extract_image_url(self, response: Any) -> Optional[str]:
        """Best-effort extraction of a usable URL or data URI.

        Handles both the ``generate_content`` shape
        (candidates -> content -> parts -> inline_data) and the legacy
        ``generate_images`` shape (generated_images -> image -> bytes/url).

        Returns ``None`` when nothing usable is found so callers can fall
        back instead of persisting a placeholder URL that 404s in the UI.
        """
        # 1) generate_content shape
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                inline_data = getattr(part, "inline_data", None)
                data = getattr(inline_data, "data", None)
                if not data:
                    continue
                if isinstance(data, bytes):
                    data = base64.b64encode(data).decode("utf-8")
                data = str(data)
                if data.startswith("data:"):
                    return data
                mime_type = getattr(inline_data, "mime_type", None) or "image/png"
                return f"data:{mime_type};base64,{data}"

        # 2) Legacy generate_images shape
        generated_images = (
            getattr(response, "generated_images", None)
            or getattr(response, "images", None)
            or []
        )
        if not isinstance(generated_images, (list, tuple)) or not generated_images:
            return None

        first_image = generated_images[0]
        for attr_name in ("url", "uri"):
            direct_value = getattr(first_image, attr_name, None)
            if direct_value:
                return direct_value

        nested_image = getattr(first_image, "image", None)
        if nested_image:
            for attr_name in ("url", "uri"):
                direct_value = getattr(nested_image, attr_name, None)
                if direct_value:
                    return direct_value

            image_bytes = (
                getattr(nested_image, "image_bytes", None)
                or getattr(nested_image, "bytes", None)
                or getattr(nested_image, "data", None)
            )
            if image_bytes:
                if isinstance(image_bytes, str):
                    return image_bytes
                return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

        return None
    
    async def get_chat_response(
        self,
        user_query: str,
        user_id: int,
        session_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for chat - fetches context and generates response
        
        Returns:
            Dict with response_text, is_high_risk, warning_data, session_id
        """
        db = SessionLocal()
        try:
            # Get laptop specs
            laptop_setup = db.query(LaptopSetup).filter(
                LaptopSetup.user_id == user_id
            ).first()
            
            laptop_specs = None
            if laptop_setup:
                laptop_specs = db.query(LaptopSpecs).filter(
                    LaptopSpecs.laptop_setup_id == laptop_setup.id
                ).first()
            
            # Get RAG context
            rag_context = []
            if laptop_setup:
                rag_context = self.vector_store.get_chunks_by_laptop(
                    brand=laptop_setup.brand,
                    model=laptop_setup.model,
                    n_results=5
                )
            
            # Get chat history
            chat_history = []
            if session_id:
                messages = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                ).order_by(ChatMessage.created_at.asc()).limit(20).all()
                
                chat_history = [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ]
            
            # Generate response
            response_text = await self.generate_troubleshooting_response(
                user_query=user_query,
                laptop_specs=laptop_specs,
                laptop_setup=laptop_setup,
                rag_context_chunks=rag_context,
                chat_history=chat_history
            )
            
            return {
                "response_text": response_text,
                "is_high_risk": False,
                "warning_data": None,
                "session_id": session_id
            }
            
        finally:
            db.close()


# Global instance
_gemini_service_instance = None

def get_gemini_service() -> GeminiService:
    """Get or create the global Gemini service instance"""
    global _gemini_service_instance
    if _gemini_service_instance is None:
        _gemini_service_instance = GeminiService()
    return _gemini_service_instance