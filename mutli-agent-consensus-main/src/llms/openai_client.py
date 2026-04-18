"""
OpenAI API client for naming game experiments.
"""

import os
import random
import json
import time
from typing import List, Optional, Dict, Any

from .base import BaseLLMClient, LLMResponse, ChoiceResult, CandidateScore


# Lazy import OpenAI
_openai_client = None
_openai_available = False

def _init_openai():
    global _openai_client, _openai_available
    if _openai_client is not None:
        return _openai_available
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            _openai_client = OpenAI(api_key=api_key)
            _openai_available = True
    except ImportError:
        pass
    
    return _openai_available


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client for GPT models.
    
    Supports:
    - Chat completions
    - JSON mode
    - Mock mode for testing
    - Seed for reproducibility (best-effort by OpenAI)
    
    Example:
        client = OpenAIClient(model="gpt-4o-mini", temperature=0.7, seed=42)
        response = client.chat("You are helpful.", "Pick a name: w0, w1, w2")
    
    Note on reproducibility:
        OpenAI's seed parameter provides "best effort" reproducibility.
        Most calls with same seed will return same result, but not guaranteed.
        Check response.system_fingerprint to verify determinism.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 10,
        top_p: float = 1.0,
        seed: int = None,
        timeout_s: float = 30.0,
        retries: int = 3,
        logprobs: bool = False,
        top_logprobs: int = 5,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, seed=seed, **kwargs)
        self.top_p = top_p
        self.timeout_s = timeout_s
        self.retries = retries
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        
        # Check for mock mode
        self.mock_mode = os.environ.get("MOCK_LLM", "0") == "1"
        self._mock_rng = random.Random(seed) if seed else random.Random()
        
        if not self.mock_mode:
            if not _init_openai():
                raise RuntimeError(
                    "OpenAI client not available. Either:\n"
                    "1. Set OPENAI_API_KEY in environment or .env file\n"
                    "2. Set MOCK_LLM=1 for testing without API calls"
                )
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        **kwargs
    ) -> str:
        """
        Send a chat completion request to OpenAI.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            json_mode: Request JSON formatted response
            **kwargs: Additional API parameters
            
        Returns:
            Response text string
        """
        if self.mock_mode:
            return self._mock_response(user_prompt)
        
        response_format = {"type": "json_object"} if json_mode else None
        
        try:
            # Build API call with optional seed for reproducibility
            api_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            }
            if response_format:
                api_kwargs["response_format"] = response_format
            if self.seed is not None:
                api_kwargs["seed"] = self.seed
            
            resp = _openai_client.chat.completions.create(**api_kwargs)
            return resp.choices[0].message.content.strip()
        
        except Exception as e:
            # Return fallback on error
            return ""
    
    def choose_from_allowed(
        self,
        system_prompt: str,
        user_prompt: str,
        allowed: List[str],
        temperature: Optional[float] = None,
    ) -> str:
        """
        Choose one token from allowed list.
        
        OpenAI doesn't support constrained generation directly,
        so we rely on the model following instructions.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            allowed: List of allowed response tokens
            temperature: Override temperature (optional)
            
        Returns:
            Selected token string
        """
        if self.mock_mode:
            return self._mock_rng.choice(allowed)
        
        temp = temperature if temperature is not None else self.temperature
        old_temp = self.temperature
        self.temperature = temp
        
        response = self.chat(system_prompt, user_prompt)
        
        self.temperature = old_temp
        
        # Parse response to find allowed token
        from ..core.parsing import extract_allowed_choice
        choice = extract_allowed_choice(response, allowed)
        
        return choice if choice else allowed[0]
    
    def ask(
        self,
        prompt: str,
        allowed: List[str],
        json_mode: bool = False
    ) -> str:
        """
        Simplified interface for naming game queries.
        
        Args:
            prompt: User prompt
            allowed: List of allowed responses
            json_mode: Request JSON format
            
        Returns:
            Response text
        """
        system = "Answer with exactly one token from the allowed list. Do not add punctuation or extra words."
        return self.chat(system, prompt, json_mode=json_mode)
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock response for testing (seeded for reproducibility)."""
        # Try to extract allowed names from prompt
        import re
        match = re.search(r"Allowed names?:\s*(.+?)(?:\n|$)", prompt)
        if match:
            names = [n.strip() for n in match.group(1).split(",")]
            return self._mock_rng.choice(names)
        return "w0"
    
    def generate_choice(
        self,
        prompt: str,
        allowed_labels: List[str],
        temperature: Optional[float] = None,
        seed: int = None,
        **kwargs
    ) -> ChoiceResult:
        """
        Generate a choice from allowed labels with retry logic.
        
        Args:
            prompt: Full prompt text
            allowed_labels: List of valid label choices
            temperature: Override temperature (optional)
            seed: Override seed for this call (optional)
            
        Returns:
            ChoiceResult with parsed choice and metadata
        """
        from ..core.parsing import extract_allowed_choice
        
        if self.mock_mode:
            choice = self._mock_rng.choice(allowed_labels)
            mock_meta = {"mock": True, "provider": "openai", "model": self.model}
            if self.logprobs:
                # Synthetic logprobs for pipeline testing
                mock_meta["first_token_top_logprobs"] = [
                    {"token": label, "logprob": -self._mock_rng.random() * 3}
                    for label in allowed_labels
                ]
            return ChoiceResult(
                text_raw=choice,
                choice=choice,
                valid=True,
                retries=0,
                meta=mock_meta
            )
        
        temp = temperature if temperature is not None else self.temperature
        use_seed = seed if seed is not None else self.seed
        
        system = "Answer with exactly one token from the allowed list. Do not add punctuation or extra words."
        response_text = ""
        latency_ms = 0
        meta = {}
        
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                api_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temp,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                }
                if use_seed is not None:
                    api_kwargs["seed"] = use_seed
                
                # Logprobs support (opt-in, backwards compatible)
                if self.logprobs:
                    api_kwargs["logprobs"] = True
                    api_kwargs["top_logprobs"] = self.top_logprobs
                
                resp = _openai_client.chat.completions.create(**api_kwargs)
                latency_ms = (time.time() - t0) * 1000
                
                response_text = resp.choices[0].message.content.strip()
                
                # Capture metadata
                meta = {
                    "provider": "openai",
                    "model": self.model,
                    "latency_ms": latency_ms,
                    "response_id": resp.id,
                    "system_fingerprint": getattr(resp, 'system_fingerprint', None),
                    "usage": {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    } if resp.usage else None,
                }
                
                # Extract first-token top_logprobs if available
                if self.logprobs:
                    meta["logprobs_requested"] = True
                    try:
                        lp_data = getattr(resp.choices[0], "logprobs", None)
                        meta["logprobs_present"] = (
                            lp_data is not None
                            and getattr(lp_data, "content", None) is not None
                        )
                        if lp_data and lp_data.content:
                            first_tok = lp_data.content[0]
                            meta["first_token_top_logprobs"] = [
                                {"token": tp.token, "logprob": tp.logprob}
                                for tp in first_tok.top_logprobs
                            ]
                    except (AttributeError, IndexError):
                        meta["logprobs_present"] = False
                
                choice = extract_allowed_choice(response_text, allowed_labels)
                
                if choice is not None:
                    return ChoiceResult(
                        text_raw=response_text,
                        choice=choice,
                        valid=True,
                        retries=attempt,
                        meta=meta
                    )
                    
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                meta["error"] = str(e)
        
        # All retries exhausted
        meta["error"] = "exhausted_retries"
        return ChoiceResult(
            text_raw=response_text,
            choice=None,
            valid=False,
            retries=self.retries,
            meta=meta
        )
    
    def set_seed(self, seed: int) -> None:
        """Update the seed for subsequent API calls."""
        self.seed = seed
        self._mock_rng = random.Random(seed)
    
    @property
    def supports_constrained_generation(self) -> bool:
        return False
    
    @property
    def supports_token_scores(self) -> bool:
        # OpenAI supports logprobs in response, but not teacher-forced scoring
        return False



# GeminiClient has been moved to gemini_client.py
# Import here for backwards compatibility
from .gemini_client import GeminiClient
