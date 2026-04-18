"""
Google Gemini API client for naming game experiments.
"""

import logging
import os
import random
import time
from typing import List, Optional, Dict, Any

from .base import BaseLLMClient, LLMResponse, ChoiceResult

logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """
    Google Gemini API client.
    
    Note: Requires google-generativeai package and GOOGLE_API_KEY.
    
    Reproducibility:
        Gemini API does not support seed parameters. Results will vary
        between runs even with identical inputs and temperature=0.
        For reproducible experiments, use local HuggingFace models instead.
    """
    
    # System instruction used for choice generation (set once on the model)
    _CHOICE_SYSTEM_INSTRUCTION = (
        "You are playing a coordination game. "
        "Answer with exactly one token from the allowed list. "
        "Do not add punctuation, explanation, or extra words."
    )
    
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        temperature: float = 1.0,
        max_tokens: int = 10,
        timeout_s: float = 30.0,
        retries: int = 3,
        seed: int = None,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, seed=seed, **kwargs)
        self.timeout_s = timeout_s
        self.retries = retries
        self.mock_mode = os.environ.get("MOCK_LLM", "0") == "1"
        self._mock_rng = random.Random(seed) if seed else random.Random()
        self._genai = None
        self._model_obj = None
        self._safety_settings = None
        
        if not self.mock_mode:
            self._init_gemini()
    
    def _init_gemini(self):
        """Initialize Gemini client with safety settings and system instruction."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        
        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("GOOGLE_API_KEY not set")
            genai.configure(api_key=api_key)
            self._genai = genai
            
            # Disable safety filters -- the naming game prompts are benign
            # research content but can trip Gemini's aggressive filters,
            # causing blocked responses that surface as empty strings.
            self._safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT:
                    genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH:
                    genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT:
                    genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT:
                    genai.types.HarmBlockThreshold.BLOCK_NONE,
            }
            
            # Use system_instruction so Gemini treats the game prompt as
            # a proper user turn instead of a single flat text blob.
            self._model_obj = genai.GenerativeModel(
                self.model,
                system_instruction=self._CHOICE_SYSTEM_INSTRUCTION,
                safety_settings=self._safety_settings,
            )
        except ImportError:
            raise RuntimeError("google-generativeai package not installed")
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """Send a chat request to Gemini."""
        if self.mock_mode:
            return self._mock_response(user_prompt)
        
        try:
            # The model already has system_instruction set; send only the
            # user content so Gemini sees proper role separation.
            response = self._model_obj.generate_content(
                user_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
            
            # Check for blocked response before accessing .text
            if not response.candidates:
                reason = getattr(response, "prompt_feedback", "unknown")
                logger.warning(
                    "Gemini returned no candidates (blocked). "
                    "prompt_feedback=%s model=%s", reason, self.model
                )
                return ""
            
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason"):
                # finish_reason == 1 means STOP (normal), others may indicate
                # safety blocks (3/4) or recitation (5) etc.
                fr = candidate.finish_reason
                if fr not in (0, 1):  # 0=UNSPECIFIED, 1=STOP
                    logger.warning(
                        "Gemini finish_reason=%s (expected STOP). "
                        "model=%s safety_ratings=%s",
                        fr, self.model,
                        getattr(candidate, "safety_ratings", None),
                    )
            
            return response.text.strip()
        
        except Exception as e:
            logger.warning(
                "Gemini chat() exception: %s (%s) model=%s",
                type(e).__name__, e, self.model,
            )
            return ""
    
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
            seed: Ignored (Gemini doesn't support seeds)
            
        Returns:
            ChoiceResult with parsed choice and metadata
        """
        from ..core.parsing import extract_allowed_choice
        
        if self.mock_mode:
            choice = self._mock_rng.choice(allowed_labels)
            return ChoiceResult(
                text_raw=choice,
                choice=choice,
                valid=True,
                retries=0,
                meta={"mock": True}
            )
        
        temp = temperature if temperature is not None else self.temperature
        old_temp = self.temperature
        self.temperature = temp
        
        response_text = ""
        latency_ms = 0.0
        last_error = None
        
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                # System instruction is already on the model object;
                # pass only the game prompt as the user turn.
                response_text = self.chat("", prompt)
                latency_ms = (time.time() - t0) * 1000
                
                choice = extract_allowed_choice(response_text, allowed_labels)
                
                if choice is not None:
                    self.temperature = old_temp
                    return ChoiceResult(
                        text_raw=response_text,
                        choice=choice,
                        valid=True,
                        retries=attempt,
                        meta={
                            "latency_ms": latency_ms,
                            "provider": "gemini",
                            "model": self.model,
                        }
                    )
                else:
                    # Parseable response but no valid token found
                    last_error = f"no_match_in_response: '{response_text[:120]}'"
            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                last_error = f"{type(e).__name__}: {e}"
        
        self.temperature = old_temp
        
        if last_error:
            logger.debug(
                "Gemini generate_choice exhausted %d retries. "
                "last_error=%s model=%s",
                self.retries, last_error, self.model,
            )
        
        # All retries exhausted
        return ChoiceResult(
            text_raw=response_text,
            choice=None,
            valid=False,
            retries=self.retries,
            meta={
                "latency_ms": latency_ms,
                "provider": "gemini",
                "model": self.model,
                "error": last_error or "exhausted_retries",
            }
        )
    
    def choose_from_allowed(
        self,
        system_prompt: str,
        user_prompt: str,
        allowed: List[str],
        temperature: Optional[float] = None,
    ) -> str:
        """Choose one token from allowed list (legacy interface)."""
        result = self.generate_choice(
            f"{system_prompt}\n\n{user_prompt}",
            allowed,
            temperature=temperature
        )
        return result.choice if result.valid else allowed[0]
    
    def score_candidates(
        self,
        prompt: str,
        candidates: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Score candidates by log-probability.
        
        Note: Gemini API does not currently expose logprobs.
        This raises NotImplementedError.
        """
        raise NotImplementedError(
            "Gemini API does not support logprob scoring. "
            "Use HuggingFace models for token scoring experiments."
        )
    
    def _mock_response(self, prompt: str) -> str:
        import re
        match = re.search(r"Allowed names?:\s*(.+?)(?:\n|$)", prompt)
        if match:
            names = [n.strip() for n in match.group(1).split(",")]
            return self._mock_rng.choice(names)
        return "w0"
    
    def set_seed(self, seed: int) -> None:
        """Update the seed (only affects mock mode for Gemini)."""
        self.seed = seed
        self._mock_rng = random.Random(seed)
    
    @property
    def supports_constrained_generation(self) -> bool:
        return False
    
    @property
    def supports_token_scores(self) -> bool:
        return False
