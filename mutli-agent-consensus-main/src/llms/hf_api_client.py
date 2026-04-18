"""
Hugging Face Router API client for naming game experiments.

Uses the OpenAI-compatible endpoint at https://router.huggingface.co/v1.
This runs models remotely instead of loading them on the local machine.
"""

import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseLLMClient, ChoiceResult

logger = logging.getLogger(__name__)


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _normalize_router_base_url(raw_url: Optional[str]) -> str:
    """
    Normalize configured HF endpoint to a router-compatible base URL.
    """
    default_url = "https://router.huggingface.co/v1"
    if not raw_url:
        return default_url

    url = str(raw_url).strip().rstrip("/")
    if "api-inference.huggingface.co" in url:
        logger.warning(
            "Configured HF API URL '%s' is deprecated; using '%s' instead.",
            url,
            default_url,
        )
        return default_url

    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


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


def _is_model_not_supported(exc: Exception) -> bool:
    text = str(exc).lower()
    return "model_not_supported" in text or "not supported by any provider" in text


def _split_provider_suffix(model_id: str) -> Tuple[str, Optional[str]]:
    if ":" not in model_id:
        return model_id, None
    base, suffix = model_id.rsplit(":", 1)
    if "/" in suffix:
        return model_id, None
    return base, suffix


class HFAPIClient(BaseLLMClient):
    """
    Hugging Face hosted inference client via Router API.

    Requires `HF_TOKEN` (or `HUGGINGFACEHUB_API_TOKEN`) in environment.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 10,
        top_p: float = 1.0,
        seed: int = None,
        retries: int = 3,
        timeout_s: float = 60.0,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, seed=seed, **kwargs)
        _load_env()

        self.top_p = top_p
        self.retries = max(1, int(retries))
        self.timeout_s = timeout_s
        self.mock_mode = os.environ.get("MOCK_LLM", "0") == "1"
        self._mock_rng = random.Random(seed) if seed is not None else random.Random()

        self.token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or os.getenv("HUGGINGFACE_API_KEY")
        )
        self.base_url = _normalize_router_base_url(
            os.getenv("HF_ROUTER_BASE_URL")
            or os.getenv("HF_API_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        fallback_modes_env = os.getenv("HF_ROUTER_FALLBACK_MODES", "fastest,cheapest")
        self.fallback_modes = [
            m.strip()
            for m in fallback_modes_env.split(",")
            if m.strip() and m.strip() not in {"none", "off", "disabled"}
        ]

        self._client = None
        if not self.mock_mode:
            if not self.token:
                raise RuntimeError(
                    "HF API token missing. Set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN)."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "openai package is required for HF router client. "
                    "Install with: pip install openai"
                ) from exc
            self._client = OpenAI(
                api_key=self.token,
                base_url=self.base_url,
                timeout=self.timeout_s,
            )

    def _chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        seed: Optional[int],
        json_mode: bool = False,
    ) -> Tuple[str, Dict[str, Any]]:
        base_model, explicit_provider = _split_provider_suffix(self.model)

        candidate_models = [self.model]
        if explicit_provider is not None:
            for mode in self.fallback_modes:
                candidate = f"{base_model}:{mode}"
                if candidate not in candidate_models:
                    candidate_models.append(candidate)
            if base_model not in candidate_models:
                candidate_models.append(base_model)
        else:
            for mode in self.fallback_modes:
                candidate = f"{base_model}:{mode}"
                if candidate not in candidate_models:
                    candidate_models.append(candidate)

        last_exc: Optional[Exception] = None
        for idx, candidate_model in enumerate(candidate_models):
            api_kwargs = {
                "model": candidate_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
            }
            if seed is not None:
                api_kwargs["seed"] = seed
            if json_mode:
                api_kwargs["response_format"] = {"type": "json_object"}

            try:
                resp = self._client.chat.completions.create(**api_kwargs)
            except Exception as exc:
                last_exc = exc
                if _is_model_not_supported(exc) and idx < len(candidate_models) - 1:
                    logger.warning(
                        "HF router model route unsupported (requested=%s, tried=%s). "
                        "Trying fallback route.",
                        self.model,
                        candidate_model,
                    )
                    continue
                raise

            text = resp.choices[0].message.content or ""
            usage = None
            if getattr(resp, "usage", None) is not None:
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }

            meta = {
                "provider": "hf_api",
                "model": self.model,
                "model_resolved": candidate_model,
                "base_url": self.base_url,
                "response_id": getattr(resp, "id", None),
                "usage": usage,
            }
            return text.strip(), meta

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("HF router request failed before sending.")

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        json_mode: bool = False,
        **kwargs,
    ) -> str:
        if self.mock_mode:
            return self._mock_response(user_prompt)

        try:
            text, _ = self._chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=self.temperature,
                seed=self.seed,
                json_mode=json_mode,
            )
            return text
        except Exception as exc:
            err = _error_message(exc)
            if _is_rate_limited(exc):
                logger.warning("HF API chat rate-limited (model=%s): %s", self.model, err)
            else:
                logger.error("HF API chat request failed (model=%s): %s", self.model, err)
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

        temp = self.temperature if temperature is None else temperature
        try:
            response, _ = self._chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temp,
                seed=self.seed,
                json_mode=False,
            )
        except Exception as exc:
            err = _error_message(exc)
            if _is_rate_limited(exc):
                logger.warning(
                    "HF API choose_from_allowed rate-limited (model=%s): %s",
                    self.model,
                    err,
                )
            else:
                logger.error(
                    "HF API choose_from_allowed failed (model=%s): %s",
                    self.model,
                    err,
                )
            response = ""

        from ..core.parsing import extract_allowed_choice

        choice = extract_allowed_choice(response, allowed)
        return choice if choice else allowed[0]

    def generate_choice(
        self,
        prompt: str,
        allowed_labels: List[str],
        temperature: Optional[float] = None,
        seed: int = None,
        **kwargs,
    ) -> ChoiceResult:
        from ..core.parsing import extract_allowed_choice

        if self.mock_mode:
            choice = self._mock_rng.choice(allowed_labels)
            return ChoiceResult(
                text_raw=choice,
                choice=choice,
                valid=True,
                retries=0,
                meta={"mock": True, "provider": "hf_api", "model": self.model},
            )

        temp = self.temperature if temperature is None else temperature
        use_seed = self.seed if seed is None else seed
        system = "Answer with exactly one token from the allowed list. Do not add punctuation or extra words."

        response_text = ""
        meta: Dict[str, Any] = {}
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                response_text, meta = self._chat_completion(
                    system_prompt=system,
                    user_prompt=prompt,
                    temperature=temp,
                    seed=use_seed,
                    json_mode=False,
                )
                meta["latency_ms"] = (time.time() - t0) * 1000.0

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
                    "provider": "hf_api",
                    "model": self.model,
                    "base_url": self.base_url,
                    "error": err,
                    "rate_limited": rate_limited,
                    "attempt": attempt + 1,
                    "max_attempts": self.retries,
                }
                if attempt < self.retries - 1:
                    if rate_limited:
                        logger.warning(
                            "HF API generate_choice rate-limited (model=%s, attempt=%d/%d). "
                            "Retrying. Error: %s",
                            self.model,
                            attempt + 1,
                            self.retries,
                            err,
                        )
                    else:
                        logger.warning(
                            "HF API generate_choice failed (model=%s, attempt=%d/%d). "
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
                            "HF API generate_choice failed after %d attempts due to rate limits "
                            "(model=%s): %s",
                            self.retries,
                            self.model,
                            err,
                        )
                    else:
                        logger.error(
                            "HF API generate_choice failed after %d attempts (model=%s): %s",
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

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self._mock_rng = random.Random(seed) if seed is not None else random.Random()

    def _mock_response(self, prompt: str) -> str:
        import re

        match = re.search(r"Allowed names?:\s*(.+?)(?:\n|$)", prompt)
        if match:
            names = [n.strip() for n in match.group(1).split(",") if n.strip()]
            if names:
                return self._mock_rng.choice(names)
        return "w0"
