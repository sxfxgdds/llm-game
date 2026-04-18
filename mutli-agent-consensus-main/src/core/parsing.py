"""
Response parsing utilities for LLM outputs.
"""

import re
import json
from typing import List, Optional


# Regex patterns
ALLOWED_TOKEN_RE = re.compile(r'\b([A-Za-z0-9_\-]{2,})\b')
TOKEN_CHARS_RE = r"[A-Za-z0-9_\-]+"


def parse_choice(text: str, allowed: List[str]) -> Optional[str]:
    """
    Parse an allowed token from LLM response text.
    
    Accepts the first occurrence of any allowed token anywhere in the text.
    Uses word boundary matching to avoid partial matches.
    
    Args:
        text: Raw LLM response text
        allowed: List of allowed token strings
        
    Returns:
        Matched token string, or None if no match found
    """
    if not text:
        return None
    
    # Build pattern with word boundaries
    pattern = r"(?<!\w)(" + "|".join(map(re.escape, allowed)) + r")(?!\w)"
    match = re.search(pattern, text)
    
    if match:
        return match.group(1)
    
    return None


def extract_allowed_choice(text: str, allowed: List[str]) -> Optional[str]:
    """
    Multi-tier extraction of allowed choice from LLM output.
    
    Tries multiple parsing strategies in order:
    1. JSON object with "choice" key
    2. Exact match from allowed list
    3. Regex word boundary match
    4. First word extraction
    5. Case-insensitive match
    
    Args:
        text: Raw LLM response text
        allowed: List of allowed token strings
        
    Returns:
        Matched token string, or None if no match found
    """
    if not text:
        return None
    
    text = text.strip()
    
    # 1. Try JSON object
    try:
        # Look for JSON anywhere in text
        json_match = re.search(r'\{[^}]+\}', text, flags=re.S)
        if json_match:
            obj = json.loads(json_match.group(0))
            if isinstance(obj, dict) and "choice" in obj:
                choice = str(obj["choice"]).strip()
                if choice in allowed:
                    return choice
    except (json.JSONDecodeError, ValueError):
        pass
    
    # 2. Exact match (entire text is just the token)
    if text in allowed:
        return text
    
    # 3. Exact match on first word
    first_word = text.split()[0].strip(",.;:\"'") if text else ""
    if first_word in allowed:
        return first_word
    
    # 4. Word boundary regex match
    result = parse_choice(text, allowed)
    if result:
        return result
    
    # 5. Case-insensitive match
    text_lower = text.lower()
    for tok in allowed:
        if tok.lower() in text_lower:
            return tok
    
    return None


def extract_json_choice(text: str, allowed: List[str]) -> Optional[str]:
    """
    Extract choice from JSON formatted response.
    
    Expects format like: {"choice": "w3"}
    
    Args:
        text: Raw LLM response text
        allowed: List of allowed token strings
        
    Returns:
        Matched token string, or None if not found/invalid
    """
    if not text:
        return None
    
    try:
        # Find JSON object
        match = re.search(r'\{[^}]+\}', text, flags=re.S)
        if not match:
            return None
        
        obj = json.loads(match.group(0))
        if not isinstance(obj, dict):
            return None
        
        choice = obj.get("choice", "")
        if isinstance(choice, str) and choice.strip() in allowed:
            return choice.strip()
        
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    return None


def validate_choice(choice: Optional[str], allowed: List[str]) -> bool:
    """
    Check if a choice is valid (in allowed list).
    
    Args:
        choice: Choice string to validate
        allowed: List of allowed token strings
        
    Returns:
        True if choice is in allowed list
    """
    return choice is not None and choice in allowed


def parse_with_fallback(
    text: str, 
    allowed: List[str], 
    fallback_random: bool = False
) -> str:
    """
    Parse choice with fallback to first allowed token.
    
    Args:
        text: Raw LLM response text
        allowed: List of allowed token strings
        fallback_random: If True, fallback to random choice; otherwise first
        
    Returns:
        Token string (always returns a valid choice)
    """
    import random
    
    result = extract_allowed_choice(text, allowed)
    
    if result is None:
        if fallback_random:
            return random.choice(allowed)
        return allowed[0]
    
    return result
