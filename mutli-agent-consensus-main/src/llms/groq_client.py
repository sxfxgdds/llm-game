"""
Groq API client for naming game experiments.

Implements the same BaseLLMClient interface as other providers.
Uses OpenAI-compatible API shape at https://api.groq.com/openai/v1.
"""

import os
import random
import time
import logging
from typing import List, Optional

from .base import BaseLLMClient, ChoiceResult

logger = logging.getLogger(__name__)


_groq_client = None
_groq_available = False


def _init_groq():
    global _groq_client, _groq_available
    if _groq_client is not None:
        return _groq_available

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        from openai import OpenAI
        api_key = os.getenv("GROQ_API_KEY")
        base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        if api_key:
            _groq_client = OpenAI(api_key=api_key, base_url=base_url)
            _groq_available = True
    except ImportError:
        pass

    return _groq_available


def _error_message(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    message = str(exc).strip() or exc.__class__.__name__
    if status_code is not None and f"status={status_code}" not in message:
        return f"status={status_code} {message}"
    return message


def _is_rate_limited(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "too many requests" in text or "429" in text


class GroqClient(BaseLLMClient):
    """
    Groq API client.

    Example model ids include:
    - llama-3.1-8b-instant
    - llama-3.3-70b-versatile
    """

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 10,
        top_p: float = 1.0,
        seed: int = None,
        retries: int = 3,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, seed=seed, **kwargs)
        self.top_p = top_p
        self.retries = retries

        self.mock_mode = os.environ.get("MOCK_LLM", "0") == "1"
        self._mock_rng = random.Random(seed) if seed else random.Random()

        if not self.mock_mode and not _init_groq():
            raise RuntimeError(
                "Groq client not available. Either:\n"
                "1. Set GROQ_API_KEY in environment or .env file\n"
                "2. Set MOCK_LLM=1 for testing without API calls"
            )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        **kwargs
    ) -> str:
        if self.mock_mode:
            return self._mock_response(user_prompt)

        response_format = {"type": "json_object"} if json_mode else None

        try:
            api_kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            }
            if response_format:
                api_kwargs["response_format"] = response_format
            # Some Groq models/routes may not support seed; only pass when provided.
            if self.seed is not None:
                api_kwargs["seed"] = self.seed

            resp = _groq_client.chat.completions.create(**api_kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            err = _error_message(exc)
            if _is_rate_limited(exc):
                logger.warning("Groq chat rate-limited (model=%s): %s", self.model, err)
            else:
                logger.error("Groq chat request failed (model=%s): %s", self.model, err)
            return ""

    def choose_from_allowed(
        self,
        system_prompt: str,
        user_prompt: str,
        allowed: List[str],
        temperature: Optional[float] = None,
    ) -> str:
        if self.mock_mode:
            return self._mock_rng.choice(allowed)

        temp = temperature if temperature is not None else self.temperature
        old_temp = self.temperature
        self.temperature = temp
        response = self.chat(system_prompt, user_prompt)
        self.temperature = old_temp

        from ..core.parsing import extract_allowed_choice
        choice = extract_allowed_choice(response, allowed)
        return choice if choice else allowed[0]

    def generate_choice(
        self,
        prompt: str,
        allowed_labels: List[str],
        temperature: Optional[float] = None,
        seed: int = None,
        **kwargs
    ) -> ChoiceResult:
        from ..core.parsing import extract_allowed_choice

        if self.mock_mode:
            choice = self._mock_rng.choice(allowed_labels)
            return ChoiceResult(
                text_raw=choice,
                choice=choice,
                valid=True,
                retries=0,
                meta={"mock": True, "provider": "groq", "model": self.model},
            )

        temp = temperature if temperature is not None else self.temperature
        use_seed = seed if seed is not None else self.seed

        system = "Answer with exactly one token from the allowed list. Do not add punctuation or extra words."
        response_text = ""
        latency_ms = 0.0
        meta = {}

        for attempt in range(self.retries):
            t0 = time.time()
            try:
                api_kwargs = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temp,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                }
                if use_seed is not None:
                    api_kwargs["seed"] = use_seed

                resp = _groq_client.chat.completions.create(**api_kwargs)
                latency_ms = (time.time() - t0) * 1000
                response_text = resp.choices[0].message.content.strip()

                meta = {
                    "provider": "groq",
                    "model": self.model,
                    "latency_ms": latency_ms,
                    "response_id": getattr(resp, "id", None),
                    "usage": {
                        "prompt_tokens": resp.usage.prompt_tokens,
                        "completion_tokens": resp.usage.completion_tokens,
                        "total_tokens": resp.usage.total_tokens,
                    } if getattr(resp, "usage", None) else None,
                }

                choice = extract_allowed_choice(response_text, allowed_labels)
                if choice is not None:
                    return ChoiceResult(
                        text_raw=response_text,
                        choice=choice,
                        valid=True,
                        retries=attempt,
                        meta=meta,
                    )
            except Exception as exc:
                err = _error_message(exc)
                rate_limited = _is_rate_limited(exc)
                meta = {
                    "provider": "groq",
                    "model": self.model,
                    "error": err,
                    "rate_limited": rate_limited,
                    "attempt": attempt + 1,
                    "max_attempts": self.retries,
                }
                if attempt < self.retries - 1:
                    if rate_limited:
                        logger.warning(
                            "Groq generate_choice rate-limited (model=%s, attempt=%d/%d). "
                            "Retrying. Error: %s",
                            self.model,
                            attempt + 1,
                            self.retries,
                            err,
                        )
                    else:
                        logger.warning(
                            "Groq generate_choice failed (model=%s, attempt=%d/%d). "
                            "Retrying. Error: %s",
                            self.model,
                            attempt + 1,
                            self.retries,
                            err,
                        )
                    time.sleep(min(2.0, 0.25 * (2 ** attempt)))
                else:
                    if rate_limited:
                        logger.error(
                            "Groq generate_choice failed after %d attempts due to rate limits "
                            "(model=%s): %s",
                            self.retries,
                            self.model,
                            err,
                        )
                    else:
                        logger.error(
                            "Groq generate_choice failed after %d attempts (model=%s): %s",
                            self.retries,
                            self.model,
                            err,
                        )

        return ChoiceResult(
            text_raw=response_text,
            choice=None,
            valid=False,
            retries=self.retries - 1,
            meta=meta,
        )

    def _mock_response(self, prompt: str) -> str:
        import re

        match = re.search(r"Allowed names?:\s*(.+?)(?:\n|$)", prompt)
        if match:
            names = [n.strip() for n in match.group(1).split(",")]
            if names:
                return self._mock_rng.choice(names)
        return "w0"
