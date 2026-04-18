"""
LLM provider adapters for naming game experiments.

Supports:
- OpenAI GPT models (API-based)
- Google Gemini models (API-based)
- HuggingFace Transformers (local, 4-bit quantized)
- llama.cpp GGUF models (local)

Key interfaces:
- BaseLLMClient: Abstract base class all adapters implement
- ChoiceResult: Standard result from generate_choice()
- CandidateScore: Result from score_candidates()
"""

from .base import BaseLLMClient, LLMResponse, ChoiceResult, CandidateScore
from .openai_client import OpenAIClient
from .groq_client import GroqClient
from .gemini_client import GeminiClient
from .hf_api_client import HFAPIClient
from .huggingface_client import HFClient, GGUFClient
from .presets import (
    PRESETS, 
    resolve_preset, 
    get_available_presets,
    get_preset_info,
    create_client,
)

__all__ = [
    # Base classes
    "BaseLLMClient",
    "LLMResponse",
    "ChoiceResult",
    "CandidateScore",
    # Clients
    "OpenAIClient",
    "GroqClient",
    "GeminiClient",
    "HFAPIClient",
    "HFClient",
    "GGUFClient",
    # Presets
    "PRESETS",
    "resolve_preset",
    "get_available_presets",
    "get_preset_info",
    "create_client",
]
