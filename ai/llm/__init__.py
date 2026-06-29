"""
LLM Integration Module

Provides natural language command interpretation for drone missions.
Supports multiple backends:
- Local small models (Phi-3, TinyLlama, Qwen2) via Ollama or transformers
- API-based models (OpenAI, Anthropic) as fallback
- Rule-based parsing as ultimate fallback
"""

from .command_interpreter import (
    CommandInterpreter,
    DroneCommand,
    CommandType,
    InterpretedMission,
)

from .llm_backends import (
    LLMBackend,
    OllamaBackend,
    TransformersBackend,
    OpenAIBackend,
    MockBackend,
    get_available_backend,
)

__all__ = [
    'CommandInterpreter',
    'DroneCommand',
    'CommandType',
    'InterpretedMission',
    'LLMBackend',
    'OllamaBackend',
    'TransformersBackend',
    'OpenAIBackend',
    'MockBackend',
    'get_available_backend',
]
