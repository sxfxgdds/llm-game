"""
Utility functions shared across the codebase.
"""

import os
import random
import datetime
from typing import List, Set

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    """Generate timestamp string for file naming."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def make_nonce_tokens(n: int, seed: int = 0) -> List[str]:
    """
    Generate unique pronounceable nonce tokens (e.g., 'Zib', 'Vex').
    
    These are used as unbiased label names in naming game experiments
    to avoid model priors on real words.
    
    Args:
        n: Number of tokens to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of unique nonce token strings
    """
    rng = random.Random(seed)
    vowels = "aeiou"
    consonants = "bcdfghjklmnpqrstvwxyz"
    tokens: Set[str] = set()
    
    while len(tokens) < n:
        form = rng.choice(["CVC", "CVCC"])
        if form == "CVC":
            t = rng.choice(consonants) + rng.choice(vowels) + rng.choice(consonants)
        else:
            t = (rng.choice(consonants) + rng.choice(vowels) + 
                 rng.choice(consonants) + rng.choice(consonants))
        tokens.add(t.capitalize())
    
    return list(tokens)


def make_simple_tokens(n: int) -> List[str]:
    """
    Generate simple w0, w1, ... token names.
    
    Args:
        n: Number of tokens to generate
        
    Returns:
        List of token strings like ['w0', 'w1', ...]
    """
    return [f"w{i}" for i in range(n)]


def create_param_filename(base_name: str, **params) -> str:
    """
    Create a filename with parameters embedded.
    
    Args:
        base_name: Base name for the file
        **params: Parameters to include in filename
        
    Returns:
        Filename string with parameters
    """
    param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
    return f"{base_name}_{param_str}"


def get_short_model_name(model_path_or_id: str) -> str:
    """
    Extract a short identifier from a model path or HuggingFace ID.
    
    Args:
        model_path_or_id: Full model path or HuggingFace model ID
        
    Returns:
        Short identifier string
    """
    s = str(model_path_or_id).lower()
    
    if "qwen" in s and "2.5" in s:
        return "qwen2p5"
    if "qwen" in s:
        return "qwen"
    if "yi" in s:
        return "yi"
    if "mistral" in s and "v0.3" in s:
        return "mistralv3"
    if "mistral" in s:
        return "mistral"
    if "phi" in s:
        return "phi3"
    if "tinyllama" in s:
        return "tinyllama"
    if "gpt-4" in s:
        return "gpt4"
    if "gpt-3.5" in s:
        return "gpt35"
    if "gemini" in s:
        return "gemini"
    
    # Fallback: extract from path
    if "/" in s:
        bits = s.split("/")
        return (bits[-1] or bits[0]).split(".")[0][:12]
    
    return os.path.basename(s).split(".")[0].lower()[:12]
