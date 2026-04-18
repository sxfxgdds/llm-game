"""
Model presets for easy configuration.
"""

from typing import Tuple, List, Dict, Any


# Model presets: name -> (backend, model_id, default_params)
PRESETS: Dict[str, Dict[str, Any]] = {
    # HuggingFace models (4-bit quantized)
    "qwen2_7b_instruct": {
        "backend": "hf",
        "model_id": "Qwen/Qwen2-7B-Instruct",
        "short_name": "qwen2",
    },
    "qwen2p5_7b_instruct": {
        "backend": "hf",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "short_name": "qwen2p5",
    },
    "yi6b": {
        "backend": "hf",
        "model_id": "01-ai/Yi-1.5-6B-Chat",
        "short_name": "yi",
    },
    "tinyllama": {
        "backend": "hf",
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "short_name": "tinyllama",
    },
    "phi3_mini_4k_instruct": {
        "backend": "hf",
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "short_name": "phi3",
    },
    "llama3-8b": {
        "backend": "hf",
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "short_name": "llama3-8b",
    },
    "mistral-7b": {
        "backend": "hf",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "short_name": "mistral-7b",
    },
    "gemma2-9b": {
        "backend": "hf",
        "model_id": "google/gemma-2-9b-it",
        "short_name": "gemma2-9b",
    },

    # Hugging Face hosted inference (API)
    "hfapi-qwen2p5-7b-instruct": {
        "backend": "hf_api",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "short_name": "hfapi-qwen2p5",
    },
    "hfapi-yi-1.5-6b-chat": {
        "backend": "hf_api",
        "model_id": "01-ai/Yi-1.5-6B-Chat",
        "short_name": "hfapi-yi6b",
    },
    "hfapi-tinyllama-1.1b-chat": {
        "backend": "hf_api",
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "short_name": "hfapi-tinyllama",
    },
    # Confirmed small text-generation models under hf-inference provider.
    # We pin provider with `:hf-inference` so routing is explicit.
    "hfapi-smollm3-3b-hfi": {
        "backend": "hf_api",
        "model_id": "HuggingFaceTB/SmolLM3-3B:hf-inference",
        "short_name": "hfapi-smollm3-3b-hfi",
    },
    "hfapi-arch-router-1p5b-hfi": {
        "backend": "hf_api",
        "model_id": "katanemo/Arch-Router-1.5B:hf-inference",
        "short_name": "hfapi-arch-router-1p5b-hfi",
    },
    # Additional small routed models (provider selected by HF router).
    "hfapi-llama-3.2-1b-instruct": {
        "backend": "hf_api",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct:novita",
        "short_name": "hfapi-llama-3.2-1b",
    },
    "hfapi-llama-3.2-3b-instruct": {
        "backend": "hf_api",
        "model_id": "meta-llama/Llama-3.2-3B-Instruct:hyperbolic",
        "short_name": "hfapi-llama-3.2-3b",
    },
    "hfapi-qwen2p5-1.5b-instruct": {
        "backend": "hf_api",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct:featherless-ai",
        "short_name": "hfapi-qwen2p5-1.5b",
    },
    "hfapi-qwen3-4b-instruct": {
        "backend": "hf_api",
        "model_id": "Qwen/Qwen3-4B-Instruct-2507:nscale",
        "short_name": "hfapi-qwen3-4b",
    },

    # Groq-hosted models (API)
    "groq-llama-3.1-8b-instant": {
        "backend": "groq",
        "model_id": "llama-3.1-8b-instant",
        "short_name": "groq-llama-8b",
    },
    "groq-llama-3.3-70b-versatile": {
        "backend": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "short_name": "groq-llama-70b",
    },
    "groq-gpt-oss-20b": {
        "backend": "groq",
        "model_id": "openai/gpt-oss-20b",
        "short_name": "groq-gpt-oss-20b",
    },
    "groq-gpt-oss-120b": {
        "backend": "groq",
        "model_id": "openai/gpt-oss-120b",
        "short_name": "groq-gpt-oss-120b",
    },
    # Preview models (evaluation / may change)
    "groq-llama-4-scout-17b-16e-instruct": {
        "backend": "groq",
        "model_id": "meta-llama/llama-4-scout-17b-16e-instruct",
        "short_name": "groq-llama4-scout",
    },
    "groq-llama-4-maverick-17b-128e-instruct": {
        "backend": "groq",
        "model_id": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "short_name": "groq-llama4-maverick",
    },
    "groq-kimi-k2-instruct-0905": {
        "backend": "groq",
        "model_id": "moonshotai/kimi-k2-instruct-0905",
        "short_name": "groq-kimi-k2",
    },
    "groq-qwen3-32b": {
        "backend": "groq",
        "model_id": "qwen/qwen3-32b",
        "short_name": "groq-qwen3-32b",
    },
    "groq-mixtral-8x7b": {
        "backend": "groq",
        "model_id": "mixtral-8x7b-32768",
        "short_name": "groq-mixtral",
    },
    
    # OpenAI models
    "gpt4o-mini": {
        "backend": "openai",
        "model_id": "gpt-4o-mini",
        "short_name": "gpt4o-mini",
    },
    "gpt4o": {
        "backend": "openai",
        "model_id": "gpt-4o",
        "short_name": "gpt4o",
    },
    "gpt35-turbo": {
        "backend": "openai",
        "model_id": "gpt-3.5-turbo",
        "short_name": "gpt35",
    },
    "gpt41-nano": {
        "backend": "openai",
        "model_id": "gpt-4.1-nano",
        "short_name": "gpt41-nano",
    },
    "gpt41-mini": {
        "backend": "openai",
        "model_id": "gpt-4.1-mini",
        "short_name": "gpt41-mini",
    },
    "gpt5-mini": {
        "backend": "openai",
        "model_id": "gpt-5-mini",
        "short_name": "gpt5-mini",
    },
    "o1-mini": {
        "backend": "openai",
        "model_id": "o1-mini",
        "short_name": "o1-mini",
    },
    "o3-mini": {
        "backend": "openai",
        "model_id": "o3-mini",
        "short_name": "o3-mini",
    },
    
    # Gemini models
    # NOTE: gemini-1.5-flash and gemini-1.5-pro have been removed from the
    # API (404 as of 2026-02). Presets now point to 2.0/2.5 equivalents.
    "gemini-2-flash": {
        "backend": "gemini",
        "model_id": "gemini-2.0-flash",
        "short_name": "gemini-2-flash",
    },
    "gemini-2-flash-lite": {
        "backend": "gemini",
        "model_id": "gemini-2.0-flash-lite",
        "short_name": "gemini-2-lite",
    },
}

# Backward-compatible aliases for old/duplicate preset names.
# Keep PRESETS canonical and unique; aliases resolve to canonical keys.
PRESET_ALIASES: Dict[str, str] = {
    # Duplicate HF aliases
    "qwen7b": "qwen2p5_7b_instruct",
    "yi15_6b_chat": "yi6b",
    "tinyllama_1b_chat": "tinyllama",
    # Existing short Groq aliases
    "groq-llama-8b": "groq-llama-3.1-8b-instant",
    "groq-llama-70b": "groq-llama-3.3-70b-versatile",
    # Hosted HF API aliases
    "hfapi-yi6b": "hfapi-yi-1.5-6b-chat",
    "hfapi-qwen7b": "hfapi-qwen2p5-7b-instruct",
    "hfapi-tinyllama": "hfapi-tinyllama-1.1b-chat",
    "hfapi-smollm3": "hfapi-smollm3-3b-hfi",
    "hfapi-arch-router": "hfapi-arch-router-1p5b-hfi",
    "hfapi-llama32-1b": "hfapi-llama-3.2-1b-instruct",
    "hfapi-llama32-3b": "hfapi-llama-3.2-3b-instruct",
    "hfapi-qwen2p5-1p5b": "hfapi-qwen2p5-1.5b-instruct",
    "hfapi-qwen3-4b": "hfapi-qwen3-4b-instruct",
}


def _resolve_preset_key(preset_name: str) -> str:
    """Resolve aliases to canonical preset key."""
    seen = set()
    key = preset_name
    while key in PRESET_ALIASES:
        if key in seen:
            raise ValueError(f"Alias cycle detected for preset: {preset_name}")
        seen.add(key)
        key = PRESET_ALIASES[key]
    return key


def resolve_preset(preset_name: str) -> Tuple[str, str]:
    """
    Resolve a preset name to (backend, model_id).
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        Tuple of (backend_name, model_id)
        
    Raises:
        ValueError: If preset not found
    """
    key = _resolve_preset_key(preset_name)
    if key not in PRESETS:
        available = get_available_presets(include_aliases=True)
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available: {available}"
        )
    
    preset = PRESETS[key]
    return preset["backend"], preset["model_id"]


def get_available_presets(include_aliases: bool = False) -> List[str]:
    """
    Get list of available preset names.

    Args:
        include_aliases: Include backward-compatible aliases.
    """
    names = list(PRESETS.keys())
    if include_aliases:
        names.extend(PRESET_ALIASES.keys())
    return sorted(names)


def get_preset_info(preset_name: str) -> Dict[str, Any]:
    """
    Get full info for a preset.
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        Dictionary with preset details
    """
    key = _resolve_preset_key(preset_name)
    if key not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}")
    info = PRESETS[key].copy()
    if key != preset_name:
        info["alias_of"] = key
    return info


def create_client(
    preset: str = None,
    backend: str = None,
    model: str = None,
    temperature: float = 1.0,
    max_tokens: int = 12,
    seed: int = None,
    **kwargs
):
    """
    Factory function to create an LLM client.
    
    Can use either a preset name or explicit backend/model specification.
    
    Args:
        preset: Preset name (optional)
        backend: Backend type (hf, hf_api, openai, groq, gemini, gguf)
        model: Model ID or path
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        seed: Random seed for reproducibility (see backend docs)
        **kwargs: Additional client parameters
        
    Returns:
        Configured LLM client instance
    
    Reproducibility:
        - OpenAI: Best-effort via API seed param (check system_fingerprint)
        - HuggingFace: Full reproducibility (set CUBLAS_WORKSPACE_CONFIG=:4096:8)
        - Gemini: No support (seed only affects mock mode)
        - GGUF: Full reproducibility via llama.cpp internal seeding
    """
    # Add seed to kwargs for passing to client
    if seed is not None:
        kwargs["seed"] = seed
    
    # Resolve from preset if provided
    if preset:
        backend, model = resolve_preset(preset)
    
    if not backend or not model:
        raise ValueError("Must provide either preset or both backend and model")
    
    # Import clients here to avoid circular imports
    if backend == "hf":
        from .huggingface_client import HFClient
        return HFClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    elif backend == "hf_api":
        from .hf_api_client import HFAPIClient
        return HFAPIClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    elif backend == "openai":
        from .openai_client import OpenAIClient
        return OpenAIClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    elif backend == "groq":
        from .groq_client import GroqClient
        return GroqClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    elif backend == "gemini":
        from .gemini_client import GeminiClient
        return GeminiClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    elif backend == "gguf":
        from .huggingface_client import GGUFClient
        return GGUFClient(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown backend: {backend}")
