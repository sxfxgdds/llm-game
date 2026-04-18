"""
HuggingFace Transformers client for local LLM inference.

Supports 4-bit quantization for efficient GPU inference.
"""

import os
import random
from typing import List, Optional, Dict

from .base import BaseLLMClient


# Lazy imports for heavy dependencies
_torch = None
_transformers = None
_has_stack = False


def _init_hf_stack():
    """Lazily initialize HuggingFace stack."""
    global _torch, _transformers, _has_stack
    
    if _torch is not None:
        return _has_stack
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch.nn.functional as F
        
        _torch = torch
        _transformers = {
            "AutoModelForCausalLM": AutoModelForCausalLM,
            "AutoTokenizer": AutoTokenizer,
            "BitsAndBytesConfig": BitsAndBytesConfig,
            "F": F,
        }
        _has_stack = True
        
    except ImportError as e:
        print(f"Warning: HuggingFace stack not available: {e}")
        _has_stack = False
    
    return _has_stack


def apply_chat_template(tokenizer, system: str, user: str) -> str:
    """Apply model's chat template or fallback to simple format."""
    try:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        return tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
    except Exception:
        return f"<s>[SYSTEM]\n{system}\n[/SYSTEM]\n[USER]\n{user}\n[/USER]\n[ASSISTANT]\n"


class HFClient(BaseLLMClient):
    """
    HuggingFace Transformers client with 4-bit quantization.
    
    Uses bitsandbytes for memory-efficient inference on consumer GPUs.
    
    Example:
        client = HFClient(model="Qwen/Qwen2.5-7B-Instruct", temperature=0.7, seed=42)
        response = client.chat("You are helpful.", "Pick a name: w0, w1, w2")
    
    Reproducibility:
        Pass seed parameter to ensure reproducible sampling. This sets the
        PyTorch generator used for multinomial sampling. Note: Full determinism
        requires CUBLAS_WORKSPACE_CONFIG=:4096:8 environment variable on CUDA.
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 12,
        device: Optional[str] = None,
        use_4bit: bool = True,
        max_memory_per_gpu: str = "6.5GiB",
        seed: int = None,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.seed = seed
        self._generator = None  # Will be set after model load
        
        if not _init_hf_stack():
            raise RuntimeError(
                "HuggingFace stack not available. Install with:\n"
                "pip install torch transformers accelerate bitsandbytes"
            )
        
        self.device = device or ("cuda" if _torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        token = os.environ.get("HF_TOKEN")
        self.tokenizer = _transformers["AutoTokenizer"].from_pretrained(
            model, 
            use_fast=True, 
            token=token
        )
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configure quantization
        if use_4bit:
            bnb_config = _transformers["BitsAndBytesConfig"](
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=_torch.float16,
            )
        else:
            bnb_config = None
        
        # Memory mapping
        n_gpus = _torch.cuda.device_count()
        max_memory = {i: max_memory_per_gpu for i in range(n_gpus)} if n_gpus > 0 else None
        
        # Load model
        self.model_obj = _transformers["AutoModelForCausalLM"].from_pretrained(
            model,
            torch_dtype=_torch.float16,
            quantization_config=bnb_config,
            device_map="auto" if n_gpus > 0 else None,
            max_memory=max_memory,
            trust_remote_code=True,
            token=token,
        )
        self.model_obj.eval()
        
        # Initialize generator for reproducible sampling
        if self.seed is not None:
            self._generator = _torch.Generator(device=self.model_obj.device)
            self._generator.manual_seed(self.seed)
    
    def _tokenize(self, text: str) -> dict:
        """Tokenize text and move to model device."""
        tokens = self.tokenizer(text, return_tensors="pt")
        return {k: v.to(self.model_obj.device) for k, v in tokens.items()}
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """
        Generate a response using the model.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            
        Returns:
            Generated response text
        """
        text = apply_chat_template(self.tokenizer, system_prompt, user_prompt)
        inputs = self._tokenize(text)
        
        with _torch.no_grad():
            output = self.model_obj.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                max_new_tokens=self.max_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            output[0][input_len:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def choose_from_allowed(
        self,
        system_prompt: str,
        user_prompt: str,
        allowed: List[str],
        temperature: Optional[float] = None,
    ) -> str:
        """
        Choose from allowed tokens using constrained first-token sampling.
        
        Uses model logits to sample only from allowed token first-pieces.
        
        Args:
            system_prompt: System message content
            user_prompt: User message content
            allowed: List of allowed response tokens
            temperature: Override temperature (optional)
            
        Returns:
            Selected token string
        """
        temp = temperature if temperature is not None else self.temperature
        
        text = apply_chat_template(self.tokenizer, system_prompt, user_prompt)
        inputs = self._tokenize(text)
        
        with _torch.no_grad():
            output = self.model_obj.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        logits = output.scores[0][0]  # [vocab_size]
        
        # Map allowed strings to first-token IDs
        valid = []
        for s in allowed:
            ids = self.tokenizer(s, add_special_tokens=False)["input_ids"]
            if ids:
                valid.append((s, ids[0]))
        
        if not valid:
            # Fallback to free generation
            return self.chat(system_prompt, user_prompt)
        
        # Select from valid tokens only
        token_ids = _torch.tensor([tid for _, tid in valid], device=logits.device)
        sub_logits = logits.index_select(0, token_ids)
        
        probs = _torch.softmax(sub_logits / max(1e-6, temp), dim=-1)
        
        # Use seeded generator for reproducibility if available
        if self._generator is not None:
            idx = _torch.multinomial(probs, num_samples=1, generator=self._generator).item()
        else:
            idx = _torch.multinomial(probs, num_samples=1).item()
        
        return valid[idx][0]
    
    def set_seed(self, seed: int) -> None:
        """Update the seed for subsequent sampling operations."""
        self.seed = seed
        if seed is not None and _torch is not None:
            self._generator = _torch.Generator(device=self.model_obj.device)
            self._generator.manual_seed(seed)
        else:
            self._generator = None
    
    def score_first_token(self, prompt: str, candidates: List[str]) -> List[float]:
        """
        Get log-probability scores for candidate first tokens.
        
        Args:
            prompt: Input prompt
            candidates: List of candidate strings
            
        Returns:
            List of log-probabilities (one per candidate)
        """
        F = _transformers["F"]
        
        with _torch.inference_mode():
            inputs = self._tokenize(prompt)
            output = self.model_obj(**inputs)
            
            logits_next = output.logits[:, -1, :]  # [1, vocab]
            logp_next = F.log_softmax(logits_next, dim=-1)  # [1, vocab]
            
            scores = []
            for cand in candidates:
                cont = " " + cand
                ids = self.tokenizer(cont, add_special_tokens=False)["input_ids"]
                if not ids:
                    scores.append(float("-inf"))
                    continue
                scores.append(logp_next[0, ids[0]].item())
            
            return scores
    
    def score_full_string(self, prompt: str, candidates: List[str]) -> List[float]:
        """
        Get sum of log-probabilities for full candidate strings.
        
        Uses teacher-forcing to compute exact continuation probabilities.
        
        Args:
            prompt: Input prompt
            candidates: List of candidate strings
            
        Returns:
            List of total log-probabilities (one per candidate)
        """
        F = _transformers["F"]
        scores = []
        
        with _torch.inference_mode():
            base = self._tokenize(prompt)
            base_len = base["input_ids"].shape[-1]
            
            for cand in candidates:
                cont = " " + cand
                whole = self._tokenize(prompt + cont)
                input_ids = whole["input_ids"]
                
                output = self.model_obj(**whole)
                logits = output.logits  # [1, L, vocab]
                
                if input_ids.shape[1] <= base_len:
                    scores.append(float("-inf"))
                    continue
                
                lp = 0.0
                for k in range(base_len, input_ids.shape[1]):
                    prev_logits = logits[0, k - 1, :]
                    logp = F.log_softmax(prev_logits, dim=-1)
                    tok_id = input_ids[0, k]
                    lp += logp[tok_id].item()
                
                scores.append(lp)
        
        return scores
    
    @property
    def supports_constrained_generation(self) -> bool:
        return True
    
    @property
    def supports_token_scores(self) -> bool:
        return True


class GGUFClient(BaseLLMClient):
    """
    llama.cpp client for GGUF models.
    
    Uses llama-cpp-python for efficient CPU/GPU inference.
    
    Reproducibility:
        Pass seed parameter for reproducible sampling. llama.cpp supports
        internal seeding for its sampling operations.
    """
    
    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 12,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        seed: int = None,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.seed = seed if seed is not None else -1  # -1 = random in llama.cpp
        
        try:
            from llama_cpp import Llama, SplitMode
        except ImportError:
            raise RuntimeError(
                "llama-cpp-python not installed. Install with:\n"
                "pip install llama-cpp-python"
            )
        
        # GPU split for multi-GPU
        try:
            import torch
            n_gpu = torch.cuda.device_count()
            tensor_split = [1.0 / n_gpu] * n_gpu if n_gpu >= 2 else None
        except ImportError:
            tensor_split = None
            n_gpu = 0
        
        self.llm = Llama(
            model_path=model,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            tensor_split=tensor_split,
            split_mode=SplitMode.LAYER if tensor_split else None,
            logits_all=False,
            verbose=False,
        )
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs
    ) -> str:
        """Generate response using llama.cpp."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            top_p=0.95,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        
        return response["choices"][0]["message"]["content"].strip()
    
    def choose_from_allowed(
        self,
        system_prompt: str,
        user_prompt: str,
        allowed: List[str],
        temperature: Optional[float] = None,
    ) -> str:
        """Choose from allowed tokens (uses free generation + parsing)."""
        response = self.chat(system_prompt, user_prompt)
        
        from ..core.parsing import extract_allowed_choice
        choice = extract_allowed_choice(response, allowed)
        
        return choice if choice else allowed[0]
    
    def set_seed(self, seed: int) -> None:
        """Update the seed for subsequent API calls."""
        self.seed = seed if seed is not None else -1
