"""
Base classes and interfaces for LLM clients.

This module defines the standard interface for all LLM adapters used in
naming game experiments. All providers (OpenAI, Gemini, HuggingFace, GGUF)
must implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class LLMResponse:
    """
    Response from an LLM client (legacy interface).
    
    Attributes:
        text: Raw text response
        choice: Parsed choice (if applicable)
        logprobs: Log probabilities (if available)
        latency: Response time in seconds
        metadata: Additional response metadata
    """
    text: str
    choice: Optional[str] = None
    logprobs: Optional[Dict[str, float]] = None
    latency: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ChoiceResult:
    """
    Result from generate_choice() method.
    
    Standard result format for all choice generation across providers.
    
    Attributes:
        text_raw: Raw text response from provider
        choice: Parsed label (None if invalid/unparseable)
        valid: Whether a valid choice was extracted
        retries: Number of retry attempts made
        meta: Provider-specific metadata (response_id, token_usage, 
              system_fingerprint, latency_ms, etc.)
    """
    text_raw: str
    choice: Optional[str] = None
    valid: bool = False
    retries: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateScore:
    """
    Score for a candidate continuation.
    
    Used by score_candidates() for logprob-based scoring.
    
    Attributes:
        candidate: The candidate string scored
        logprob: Log-probability of the continuation
        meta: Additional metadata (tokenization length, token_ids, etc.)
    """
    candidate: str
    logprob: float
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    All LLM backends must implement this interface.
    
    Reproducibility:
        Pass seed parameter to enable reproducible sampling where supported.
        - OpenAI: Best-effort reproducibility via API seed parameter
        - HuggingFace: Full reproducibility via PyTorch generator seeding
        - Gemini: No API support; seed only affects mock mode
        - GGUF: Full reproducibility via llama.cpp seed
    """
    
    def __init__(
        self, 
        model: str, 
        temperature: float = 1.0,
        max_tokens: int = 10,
        seed: int = None,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.mock_mode = False
    
    @abstractmethod
    def chat(
        self, 
        system_prompt: str, 
        user_prompt: str,
        **kwargs
    ) -> str:
        """
        Send a chat completion request.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            **kwargs: Additional parameters
            
        Returns:
            Response text string
        """
        pass
    
    @abstractmethod
    def choose_from_allowed(
        self,
        system_prompt: str,
        user_prompt: str,
        allowed: List[str],
        temperature: Optional[float] = None,
    ) -> str:
        """
        Choose one token from an allowed list.
        
        Uses constrained generation when available.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            allowed: List of allowed response tokens
            temperature: Override temperature (optional)
            
        Returns:
            Selected token string
        """
        pass
    
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
        
        This is the preferred method for naming game experiments.
        Implementations should:
        1. Build appropriate instruction to constrain output
        2. Handle retries on invalid responses
        3. Capture provider metadata
        
        Args:
            prompt: Full prompt text
            allowed_labels: List of valid label choices
            temperature: Override temperature (optional)
            seed: Random seed if supported by provider
            **kwargs: Provider-specific arguments
            
        Returns:
            ChoiceResult with parsed choice and metadata
        """
        # Default implementation using legacy interface
        system = "Answer with exactly one token from the allowed list. No extra words."
        response = self.choose_from_allowed(system, prompt, allowed_labels, temperature)
        
        from ..core.parsing import extract_allowed_choice
        choice = extract_allowed_choice(response, allowed_labels)
        
        return ChoiceResult(
            text_raw=response,
            choice=choice,
            valid=choice is not None,
            retries=0,
            meta={"provider": "unknown", "model": self.model}
        )
    
    def score_candidates(
        self,
        prompt: str,
        candidates: List[str],
        **kwargs
    ) -> List[CandidateScore]:
        """
        Score candidate continuations by log-probability.
        
        Not all backends support this. Check supports_token_scores property.
        
        Args:
            prompt: Input prompt
            candidates: List of candidate strings to score
            **kwargs: Provider-specific arguments
            
        Returns:
            List of CandidateScore objects
            
        Raises:
            NotImplementedError: If backend doesn't support scoring
        """
        raise NotImplementedError("Token scoring not supported by this backend")
    
    def tokenize(self, text: str) -> List[Any]:
        """
        Tokenize text using the model's tokenizer.
        
        Optional method - not all backends expose tokenization.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of token IDs or token strings
            
        Raises:
            NotImplementedError: If tokenization not available
        """
        raise NotImplementedError("Tokenization not available for this backend")
    
    def ask(
        self, 
        prompt: str, 
        allowed: List[str],
        json_mode: bool = False
    ) -> str:
        """
        Convenience method for simple prompts (legacy interface).
        
        Prefer generate_choice() for new code.
        
        Args:
            prompt: User prompt
            allowed: List of allowed responses
            json_mode: Whether to request JSON format
            
        Returns:
            Response text
        """
        system = "Answer with exactly one token from the allowed list. No extra words."
        return self.chat(system, prompt)
    
    def get_token_scores(
        self,
        prompt: str,
        candidates: List[str],
    ) -> List[float]:
        """
        Get log-probability scores for candidate tokens (legacy interface).
        
        Prefer score_candidates() for new code.
        
        Args:
            prompt: Input prompt
            candidates: List of candidate tokens
            
        Returns:
            List of log-probabilities
        """
        results = self.score_candidates(prompt, candidates)
        return [r.logprob for r in results]
    
    def set_mock_mode(self, enabled: bool = True) -> None:
        """Enable or disable mock mode (for testing)."""
        self.mock_mode = enabled
    
    def set_seed(self, seed: int) -> None:
        """
        Update the seed for subsequent operations.
        
        Subclasses should override this to properly reset internal RNGs.
        
        Args:
            seed: New seed value (None to disable seeding)
        """
        self.seed = seed
    
    @property
    def supports_constrained_generation(self) -> bool:
        """Whether this backend supports constrained token generation."""
        return False
    
    @property
    def supports_token_scores(self) -> bool:
        """Whether this backend supports token log-probability scoring."""
        return False
