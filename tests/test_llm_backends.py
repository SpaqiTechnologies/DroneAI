"""
Tests for LLM backends.
"""

import pytest
import json
from ai.llm.llm_backends import (
    LLMConfig,
    LLMResponse,
    LLMBackend,
    MockBackend,
    OllamaBackend,
    TransformersBackend,
    OpenAIBackend,
    ModelSize,
    get_available_backend,
)


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_defaults(self):
        config = LLMConfig()

        assert config.model_name == "phi3:mini"
        assert config.temperature == 0.3
        assert config.max_tokens == 500
        assert config.timeout == 30.0
        assert config.max_memory_mb == 4096

    def test_custom_config(self):
        config = LLMConfig(
            model_name="tinyllama",
            temperature=0.5,
            max_tokens=1000,
            api_key="test-key",
        )

        assert config.model_name == "tinyllama"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.api_key == "test-key"


class TestLLMResponse:
    """Tests for LLMResponse."""

    def test_creation(self):
        response = LLMResponse(
            text="Test response",
            model="test-model",
            tokens_used=50,
            latency_ms=100.5,
        )

        assert response.text == "Test response"
        assert response.model == "test-model"
        assert response.tokens_used == 50
        assert response.latency_ms == 100.5
        assert response.success is True
        assert response.error is None

    def test_error_response(self):
        response = LLMResponse(
            text="",
            model="test-model",
            success=False,
            error="Test error",
        )

        assert response.success is False
        assert response.error == "Test error"

    def test_to_dict(self):
        response = LLMResponse(
            text="Hello",
            model="gpt-3.5",
            tokens_used=10,
            latency_ms=50.0,
        )

        d = response.to_dict()

        assert d['text'] == "Hello"
        assert d['model'] == "gpt-3.5"
        assert d['tokens_used'] == 10
        assert d['latency_ms'] == 50.0
        assert d['success'] is True


class TestMockBackend:
    """Tests for MockBackend (always available for testing)."""

    def test_name(self):
        backend = MockBackend()
        assert backend.name == "mock"

    def test_always_available(self):
        backend = MockBackend()
        assert backend.is_available() is True

    def test_takeoff_command(self):
        backend = MockBackend()
        response = backend.generate("take off to 50 meters")

        assert response.success
        data = json.loads(response.text)
        assert data['command_type'] == 'takeoff'
        assert data['parameters']['altitude'] == 50.0
        assert data['confidence'] >= 0.9

    def test_takeoff_feet(self):
        backend = MockBackend()
        response = backend.generate("take off to 100 feet")

        data = json.loads(response.text)
        assert data['command_type'] == 'takeoff'
        # 100 feet ~ 30.48 meters
        assert abs(data['parameters']['altitude'] - 30.48) < 0.1

    def test_land_command(self):
        backend = MockBackend()
        response = backend.generate("land now")

        data = json.loads(response.text)
        assert data['command_type'] == 'land'
        assert data['confidence'] >= 0.9

    def test_rtl_command(self):
        backend = MockBackend()
        response = backend.generate("return to home")

        data = json.loads(response.text)
        assert data['command_type'] == 'rtl'

    def test_follow_person_command(self):
        backend = MockBackend()
        response = backend.generate("follow me")

        data = json.loads(response.text)
        assert data['command_type'] == 'follow'
        assert data['parameters']['target'] == 'person'

    def test_follow_vehicle_command(self):
        backend = MockBackend()
        response = backend.generate("follow that car")

        data = json.loads(response.text)
        assert data['command_type'] == 'follow'
        assert data['parameters']['target'] == 'vehicle'

    def test_hover_command(self):
        backend = MockBackend()
        response = backend.generate("hover here")

        data = json.loads(response.text)
        assert data['command_type'] == 'hover'

    def test_survey_grid_command(self):
        backend = MockBackend()
        response = backend.generate("survey the field in a grid pattern")

        data = json.loads(response.text)
        assert data['command_type'] == 'survey'
        assert data['parameters']['pattern'] == 'grid'

    def test_orbit_command(self):
        backend = MockBackend()
        response = backend.generate("orbit at 20 meters")

        data = json.loads(response.text)
        assert data['command_type'] == 'orbit'
        assert data['parameters']['radius'] == 20.0

    def test_goto_command(self):
        backend = MockBackend()
        response = backend.generate("go to the park")

        data = json.loads(response.text)
        assert data['command_type'] == 'goto'
        assert 'description' in data['parameters']

    def test_unknown_command(self):
        backend = MockBackend()
        response = backend.generate("do something random")

        data = json.loads(response.text)
        assert data['command_type'] == 'unknown'
        assert data['confidence'] == 0.5

    def test_response_has_latency(self):
        backend = MockBackend()
        response = backend.generate("takeoff")

        assert response.latency_ms >= 0

    def test_system_prompt_ignored(self):
        backend = MockBackend()
        response = backend.generate("take off", system_prompt="Custom system prompt")

        # Mock backend ignores system prompt but still works
        assert response.success
        data = json.loads(response.text)
        assert data['command_type'] == 'takeoff'


class TestOllamaBackend:
    """Tests for OllamaBackend."""

    def test_name(self):
        backend = OllamaBackend()
        assert backend.name == "ollama"

    def test_recommended_models(self):
        assert "phi3:mini" in OllamaBackend.RECOMMENDED_MODELS
        assert "tinyllama" in OllamaBackend.RECOMMENDED_MODELS

    def test_unavailable_returns_error(self):
        # When Ollama isn't running, should return error response
        config = LLMConfig(base_url="http://localhost:99999")  # Invalid port
        backend = OllamaBackend(config)

        response = backend.generate("takeoff")

        assert response.success is False
        assert response.error is not None

    def test_custom_base_url(self):
        config = LLMConfig(base_url="http://custom-host:11434")
        backend = OllamaBackend(config)

        # Should use custom URL (will fail connection but config is correct)
        assert backend.config.base_url == "http://custom-host:11434"

    def test_get_drone_system_prompt(self):
        backend = OllamaBackend()
        prompt = backend.get_drone_system_prompt()

        assert "drone command interpreter" in prompt.lower()
        assert "JSON" in prompt
        assert "command_type" in prompt


class TestTransformersBackend:
    """Tests for TransformersBackend."""

    def test_name(self):
        backend = TransformersBackend()
        assert backend.name == "transformers"

    def test_recommended_models(self):
        assert "microsoft/Phi-3-mini-4k-instruct" in TransformersBackend.RECOMMENDED_MODELS

    def test_unavailable_without_transformers(self):
        # This test checks behavior when transformers isn't imported
        # In a real env, transformers may or may not be available
        backend = TransformersBackend()

        # If transformers not installed, should return False
        # If installed, returns True
        result = backend.is_available()
        assert isinstance(result, bool)


class TestOpenAIBackend:
    """Tests for OpenAIBackend."""

    def test_name(self):
        backend = OpenAIBackend()
        assert backend.name == "openai"

    def test_unavailable_without_key(self):
        config = LLMConfig(api_key=None)
        backend = OpenAIBackend(config)

        # Without API key, should not be available
        # (unless OPENAI_API_KEY env var is set)
        # We test that generate returns error without key
        response = backend.generate("test")

        # Either not available or returns error
        assert response.success is False or not backend.is_available()

    def test_with_api_key_config(self):
        config = LLMConfig(api_key="test-key-12345")
        backend = OpenAIBackend(config)

        # With API key, should be available
        assert backend.is_available() is True


class TestGetAvailableBackend:
    """Tests for get_available_backend function."""

    def test_returns_backend(self):
        backend = get_available_backend()

        assert backend is not None
        assert isinstance(backend, LLMBackend)
        assert backend.is_available() is True

    def test_mock_always_fallback(self):
        # With no external services, should fall back to mock
        config = LLMConfig(
            base_url="http://localhost:99999",  # Invalid Ollama
            api_key=None,  # No OpenAI
        )
        backend = get_available_backend(config)

        # Should get something that works
        assert backend.is_available()

    def test_with_custom_config(self):
        config = LLMConfig(
            model_name="custom-model",
            temperature=0.7,
        )
        backend = get_available_backend(config)

        assert backend.config.model_name == "custom-model"
        assert backend.config.temperature == 0.7


class TestModelSize:
    """Tests for ModelSize enum."""

    def test_sizes(self):
        assert ModelSize.TINY.value == "tiny"
        assert ModelSize.SMALL.value == "small"
        assert ModelSize.MEDIUM.value == "medium"
        assert ModelSize.LARGE.value == "large"
