"""
LLM Backend Implementations

Supports multiple LLM providers with automatic fallback:
1. Ollama (local) - Recommended for edge deployment
2. Transformers (local) - For systems without Ollama
3. OpenAI API - Cloud fallback
4. Mock - Testing and offline mode

Recommended small models for drone applications:
- phi3:mini (3.8B) - Best quality/size ratio
- qwen2:1.5b - Very fast, good for edge
- tinyllama (1.1B) - Minimal resource usage
- gemma2:2b - Good balance
"""

from __future__ import annotations

import os
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ModelSize(Enum):
    """Model size categories."""
    TINY = "tiny"       # <2B parameters
    SMALL = "small"     # 2-4B parameters
    MEDIUM = "medium"   # 4-8B parameters
    LARGE = "large"     # >8B parameters


@dataclass
class LLMConfig:
    """Configuration for LLM backend."""
    model_name: str = "phi3:mini"
    temperature: float = 0.3          # Lower for more deterministic outputs
    max_tokens: int = 500
    timeout: float = 30.0
    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # Resource constraints for edge deployment
    max_memory_mb: int = 4096
    use_quantization: bool = True


@dataclass
class LLMResponse:
    """Response from LLM."""
    text: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'model': self.model,
            'tokens_used': self.tokens_used,
            'latency_ms': self.latency_ms,
            'success': self.success,
            'error': self.error,
        }


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._available = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response from prompt."""
        pass

    def get_drone_system_prompt(self) -> str:
        """Get system prompt for drone command interpretation."""
        return """You are a drone command interpreter. Convert natural language to structured drone commands.

Output JSON with this structure:
{
    "command_type": "takeoff|land|goto|mission|follow|rtl|hover|orbit|survey",
    "parameters": {
        // command-specific parameters
    },
    "waypoints": [  // for missions
        {"lat": float, "lon": float, "alt": float}
    ],
    "confidence": 0.0-1.0
}

Examples:
- "fly to the park" -> {"command_type": "goto", "parameters": {"description": "park"}, "confidence": 0.7}
- "take off to 50 meters" -> {"command_type": "takeoff", "parameters": {"altitude": 50}, "confidence": 0.95}
- "follow that car" -> {"command_type": "follow", "parameters": {"target": "vehicle"}, "confidence": 0.85}
- "survey the field in a grid pattern" -> {"command_type": "survey", "parameters": {"pattern": "grid"}, "confidence": 0.9}

Only output valid JSON, no explanation."""


class OllamaBackend(LLMBackend):
    """
    Ollama backend for local LLM inference.

    Recommended models for drones:
    - phi3:mini - Best quality (3.8B params, ~2.3GB)
    - qwen2:1.5b - Fast (1.5B params, ~1GB)
    - tinyllama - Tiny (1.1B params, ~600MB)
    """

    RECOMMENDED_MODELS = [
        "phi3:mini",
        "qwen2:1.5b",
        "tinyllama",
        "gemma2:2b",
        "llama3.2:1b",
    ]

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            import requests
            url = self.config.base_url or "http://localhost:11434"
            response = requests.get(f"{url}/api/tags", timeout=2)
            self._available = response.status_code == 200
            return self._available
        except Exception:
            self._available = False
            return False

    def list_models(self) -> List[str]:
        """List available Ollama models."""
        try:
            import requests
            url = self.config.base_url or "http://localhost:11434"
            response = requests.get(f"{url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [m['name'] for m in data.get('models', [])]
        except Exception:
            pass
        return []

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate using Ollama API."""
        import time

        if not self.is_available():
            return LLMResponse(
                text="",
                model=self.config.model_name,
                success=False,
                error="Ollama not available"
            )

        try:
            import requests

            start_time = time.time()
            url = self.config.base_url or "http://localhost:11434"

            payload = {
                "model": self.config.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            }

            if system_prompt:
                payload["system"] = system_prompt

            response = requests.post(
                f"{url}/api/generate",
                json=payload,
                timeout=self.config.timeout
            )

            latency = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                return LLMResponse(
                    text=data.get('response', ''),
                    model=self.config.model_name,
                    tokens_used=data.get('eval_count', 0),
                    latency_ms=latency,
                    success=True,
                )
            else:
                return LLMResponse(
                    text="",
                    model=self.config.model_name,
                    success=False,
                    error=f"HTTP {response.status_code}"
                )

        except Exception as e:
            return LLMResponse(
                text="",
                model=self.config.model_name,
                success=False,
                error=str(e)
            )


class TransformersBackend(LLMBackend):
    """
    Hugging Face Transformers backend for local inference.

    Good for systems without Ollama but with GPU/good CPU.
    Uses quantized models for efficiency.
    """

    RECOMMENDED_MODELS = [
        "microsoft/Phi-3-mini-4k-instruct",
        "Qwen/Qwen2-1.5B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]

    def __init__(self, config: Optional[LLMConfig] = None):
        super().__init__(config)
        self._model = None
        self._tokenizer = None

    @property
    def name(self) -> str:
        return "transformers"

    def is_available(self) -> bool:
        """Check if transformers is available."""
        try:
            import transformers
            import torch
            self._available = True
            return True
        except ImportError:
            self._available = False
            return False

    def _load_model(self):
        """Lazy load model."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_name = self.config.model_name
            if model_name.startswith("phi3"):
                model_name = "microsoft/Phi-3-mini-4k-instruct"
            elif model_name.startswith("qwen"):
                model_name = "Qwen/Qwen2-1.5B-Instruct"
            elif model_name.startswith("tinyllama"):
                model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)

            load_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": "auto",
            }

            if self.config.use_quantization:
                try:
                    load_kwargs["load_in_4bit"] = True
                except Exception:
                    pass

            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )

        except Exception as e:
            print(f"Failed to load model: {e}")
            self._model = None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate using transformers."""
        import time

        if not self.is_available():
            return LLMResponse(
                text="",
                model=self.config.model_name,
                success=False,
                error="Transformers not available"
            )

        try:
            self._load_model()
            if self._model is None:
                return LLMResponse(
                    text="",
                    model=self.config.model_name,
                    success=False,
                    error="Model failed to load"
                )

            start_time = time.time()

            # Format prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"

            inputs = self._tokenizer(full_prompt, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

            response_text = self._tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )

            latency = (time.time() - start_time) * 1000

            return LLMResponse(
                text=response_text,
                model=self.config.model_name,
                tokens_used=outputs.shape[1],
                latency_ms=latency,
                success=True,
            )

        except Exception as e:
            return LLMResponse(
                text="",
                model=self.config.model_name,
                success=False,
                error=str(e)
            )


class OpenAIBackend(LLMBackend):
    """
    OpenAI API backend (GPT-3.5/4).

    Used as fallback when local models aren't available.
    Requires OPENAI_API_KEY environment variable.
    """

    @property
    def name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        """Check if OpenAI API key is available."""
        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        self._available = bool(api_key)
        return self._available

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate using OpenAI API."""
        import time

        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return LLMResponse(
                text="",
                model="gpt-3.5-turbo",
                success=False,
                error="No API key"
            )

        try:
            import requests

            start_time = time.time()

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
                timeout=self.config.timeout
            )

            latency = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                return LLMResponse(
                    text=data['choices'][0]['message']['content'],
                    model="gpt-3.5-turbo",
                    tokens_used=data.get('usage', {}).get('total_tokens', 0),
                    latency_ms=latency,
                    success=True,
                )
            else:
                return LLMResponse(
                    text="",
                    model="gpt-3.5-turbo",
                    success=False,
                    error=f"HTTP {response.status_code}"
                )

        except Exception as e:
            return LLMResponse(
                text="",
                model="gpt-3.5-turbo",
                success=False,
                error=str(e)
            )


class MockBackend(LLMBackend):
    """
    Mock backend for testing and offline mode.

    Uses rule-based parsing instead of LLM.
    """

    @property
    def name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        """Always available."""
        self._available = True
        return True

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate using rule-based parsing."""
        import time
        start_time = time.time()

        # Simple rule-based command extraction
        prompt_lower = prompt.lower()

        result = {"command_type": "unknown", "parameters": {}, "confidence": 0.5}

        # Emergency stop (check FIRST - highest priority)
        if "emergency" in prompt_lower or "kill" in prompt_lower:
            result["command_type"] = "emergency_stop"
            result["confidence"] = 0.95

        # Arm/disarm (check before other patterns)
        elif "disarm" in prompt_lower:
            result["command_type"] = "disarm"
            result["confidence"] = 0.9

        elif "arm" in prompt_lower and "disarm" not in prompt_lower:
            result["command_type"] = "arm"
            result["confidence"] = 0.9

        # Takeoff patterns
        elif any(word in prompt_lower for word in ["take off", "takeoff", "lift off", "launch"]):
            result["command_type"] = "takeoff"
            # Extract altitude
            alt_match = re.search(r'(\d+)\s*(m|meter|meters|feet|ft)?', prompt_lower)
            if alt_match:
                alt = float(alt_match.group(1))
                if alt_match.group(2) in ['feet', 'ft']:
                    alt *= 0.3048
                result["parameters"]["altitude"] = alt
            else:
                result["parameters"]["altitude"] = 10.0
            result["confidence"] = 0.9

        # Land patterns
        elif any(word in prompt_lower for word in ["land", "touch down", "set down"]):
            result["command_type"] = "land"
            result["confidence"] = 0.95

        # RTL patterns
        elif any(phrase in prompt_lower for phrase in ["return", "come back", "go home", "rtl"]):
            result["command_type"] = "rtl"
            result["confidence"] = 0.9

        # Follow patterns
        elif any(word in prompt_lower for word in ["follow", "track", "chase"]):
            result["command_type"] = "follow"
            if any(v in prompt_lower for v in ["car", "vehicle", "truck", "van", "bus"]):
                result["parameters"]["target"] = "vehicle"
            elif "person" in prompt_lower or "me" in prompt_lower:
                result["parameters"]["target"] = "person"
            else:
                result["parameters"]["target"] = "person"
            result["confidence"] = 0.85

        # Hover/hold patterns
        elif any(word in prompt_lower for word in ["hover", "hold", "stay", "stop"]):
            result["command_type"] = "hover"
            result["confidence"] = 0.9

        # Survey patterns
        elif any(word in prompt_lower for word in ["survey", "scan", "map", "inspect"]):
            result["command_type"] = "survey"
            if "grid" in prompt_lower:
                result["parameters"]["pattern"] = "grid"
            elif "circle" in prompt_lower or "orbit" in prompt_lower:
                result["parameters"]["pattern"] = "orbit"
            elif "corridor" in prompt_lower or "line" in prompt_lower:
                result["parameters"]["pattern"] = "corridor"
            else:
                result["parameters"]["pattern"] = "grid"
            result["confidence"] = 0.8

        # Orbit patterns
        elif "orbit" in prompt_lower or "circle" in prompt_lower:
            result["command_type"] = "orbit"
            radius_match = re.search(r'(\d+)\s*(m|meter|meters)?', prompt_lower)
            if radius_match:
                result["parameters"]["radius"] = float(radius_match.group(1))
            result["confidence"] = 0.85

        # Goto patterns
        elif any(word in prompt_lower for word in ["go to", "fly to", "move to", "navigate"]):
            result["command_type"] = "goto"
            # Try to extract location description
            result["parameters"]["description"] = prompt
            result["confidence"] = 0.7

        latency = (time.time() - start_time) * 1000

        return LLMResponse(
            text=json.dumps(result),
            model="rule-based",
            tokens_used=0,
            latency_ms=latency,
            success=True,
        )


def get_available_backend(config: Optional[LLMConfig] = None) -> LLMBackend:
    """
    Get the best available LLM backend.

    Priority:
    1. Ollama (local, fast)
    2. Transformers (local, slower to load)
    3. OpenAI (cloud, requires API key)
    4. Mock (always available, rule-based)
    """
    config = config or LLMConfig()

    # Try Ollama first
    ollama = OllamaBackend(config)
    if ollama.is_available():
        return ollama

    # Try Transformers
    transformers = TransformersBackend(config)
    if transformers.is_available():
        return transformers

    # Try OpenAI
    openai = OpenAIBackend(config)
    if openai.is_available():
        return openai

    # Fall back to mock
    return MockBackend(config)
