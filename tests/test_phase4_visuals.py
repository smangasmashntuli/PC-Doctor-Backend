import os
import sys
import types
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

sys.modules['chromadb'] = Mock()
sys.modules['chromadb.config'] = Mock()

# ``main`` pulls in heavy/optional modules at import time. Stub them out, but
# ONLY for the moment ``main`` is imported: permanently overwriting entries in
# sys.modules leaks into every other test module collected in the same pytest
# process. That previously made ``from backend.app.rag_engine import
# RAGEngine`` resolve to this stub ("unknown location") and replaced the real
# ``google`` package with a Mock, breaking unrelated test files.
_STUBBED_MODULES = {
    'backend.app.rag_engine': {'get_rag_engine': lambda: Mock()},
    'backend.app.youtube_service': {'get_youtube_service': lambda: Mock()},
    'backend.app.notifications': {'get_notification_service': lambda: Mock()},
}


def _import_main():
    """Import the FastAPI app with temporary stubs, then restore sys.modules.

    ``main`` caches its imported callables in its own namespace, so it keeps
    using the stubs afterwards even though sys.modules is restored.
    """
    if 'main' in sys.modules:
        return sys.modules['main']

    saved = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    saved_google = {name: sys.modules.get(name) for name in ('google', 'google.genai')}
    try:
        for name, attrs in _STUBBED_MODULES.items():
            stub = types.ModuleType(name)
            for attr_name, attr_value in attrs.items():
                setattr(stub, attr_name, attr_value)
            sys.modules[name] = stub
        sys.modules['google'] = Mock()
        sys.modules['google.genai'] = Mock()

        import main  # noqa: F401
        return sys.modules['main']
    finally:
        for name, previous in {**saved, **saved_google}.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

from fastapi.testclient import TestClient

from backend.app.gemini_service import GeminiService, get_gemini_service
from backend.app.models import LaptopSetup, LaptopSpecs


@pytest.fixture
def mock_gemini_service():
    with patch('backend.app.gemini_service.genai') as mock_genai, \
         patch('backend.app.gemini_service.get_vector_store') as mock_vs, \
         patch.dict('os.environ', {
             'GEMINI_FLASH_API_KEY': 'test_flash_key',
             'GEMINI_IMAGEN_API_KEY': 'test_image_key',
         }):

        mock_flash_client = types.SimpleNamespace()
        mock_image_client = types.SimpleNamespace()
        mock_vs.return_value = Mock()

        image_response = Mock()
        image_response.generated_images = [types.SimpleNamespace(image=types.SimpleNamespace(url='https://example.com/laptop.png'))]
        mock_image_client.models = types.SimpleNamespace(generate_images=lambda **kwargs: image_response)

        text_response = Mock()
        text_response.text = (
            'The SSD is like the laptop\'s filing cabinet. It stores files even when the power is off. '
            'If it fails, the laptop may take a long time to start, lose access to files, or freeze while opening apps. '
            'Keep it dry, avoid bumps, and do not remove it while the laptop is powered on.'
        )
        mock_flash_client.models = types.SimpleNamespace(generate_content=lambda **kwargs: text_response)

        def fake_client(api_key=None):
            return mock_image_client if api_key == 'test_image_key' else mock_flash_client

        mock_genai.Client.side_effect = fake_client

        service = GeminiService()
        return service


@pytest.mark.asyncio
async def test_generate_laptop_image_returns_valid_url(mock_gemini_service):
    image_url = await mock_gemini_service.generate_laptop_image('Dell', 'XPS 15 9520')

    assert isinstance(image_url, str)
    assert image_url.startswith('https://') or image_url.startswith('data:image/')


@pytest.mark.asyncio
async def test_explain_component_returns_non_technical_text(mock_gemini_service):
    explanation = await mock_gemini_service.explain_component('SSD', 'Lenovo', 'ThinkPad T14')

    assert isinstance(explanation, str)
    assert len(explanation) > 20
    assert 'ssd' in explanation.lower() or 'storage' in explanation.lower()


def test_laptop_3d_model_endpoint_returns_mapping():
    app = _import_main().app
    from backend.app.auth import get_current_user
    from backend.app.database import get_db

    client = TestClient(app)

    mock_user = Mock()
    mock_user.id = 1

    mock_laptop = Mock(spec=LaptopSetup)
    mock_laptop.id = 7
    mock_laptop.user_id = 1
    mock_laptop.brand = 'Dell'
    mock_laptop.model = 'XPS 15 9520'

    mock_specs = Mock(spec=LaptopSpecs)
    mock_specs.image_url = 'https://example.com/laptop.png'
    mock_specs.gpu = 'NVIDIA RTX 4050'

    mock_db = Mock()
    mock_db.query.return_value.filter.return_value.first.side_effect = [mock_laptop, mock_specs]

    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_db] = lambda: mock_db

    try:
        response = client.get('/laptop/7/3d-model')
        payload = response.json()

        assert response.status_code == 200
        assert payload['laptop_id'] == 7
        assert payload['model_url'].endswith('.glb')
        assert payload['component_nodes'] == {
            'RAM': 'mesh_ram_01',
            'SSD': 'mesh_nvme_01',
            'Battery': 'mesh_batt_01',
            'Fan': 'mesh_fan_01',
            'Cover': 'mesh_bottom_cover',
        }
    finally:
        app.dependency_overrides = {}


def test_chat_explain_component_endpoint_returns_plain_language():
    app = _import_main().app
    from backend.app.auth import get_current_user

    client = TestClient(app)

    mock_user = Mock()
    mock_user.id = 1

    mock_service = Mock()
    mock_service.explain_component = AsyncMock(return_value=(
        'The RAM is short-term memory. It helps the laptop work on things right now. '
        'If it has problems, apps may slow down, freeze, or restart unexpectedly. '
        'Avoid touching the chips directly and keep the device powered off before opening anything.'
    ))

    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_gemini_service] = lambda: mock_service

    try:
        response = client.post(
            '/chat/explain-component',
            json={
                'component_name': 'RAM',
                'laptop_brand': 'HP',
                'laptop_model': 'Pavilion 15',
            },
        )

        payload = response.json()

        assert response.status_code == 200
        assert payload['component_name'] == 'RAM'
        assert payload['explanation']
        assert 'memory' in payload['explanation'].lower() or 'apps' in payload['explanation'].lower()
    finally:
        app.dependency_overrides = {}