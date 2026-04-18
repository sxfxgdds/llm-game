"""
Agent pairing strategies for naming game experiments.
"""

import random
from typing import List, Tuple, Optional


def pair_indices(n_agents: int, rng: Optional[random.Random] = None) -> List[Tuple[int, int]]:
    """
    Randomly pair agents for a round of the naming game.
    
    Args:
        n_agents: Number of agents (must be even)
        rng: Random number generator instance (optional)
        
    Returns:
        List of (agent_i, agent_j) pairs
        
    Raises:
        ValueError: If n_agents is odd
    """
    if n_agents % 2 != 0:
        raise ValueError(f"n_agents must be even, got {n_agents}")
    
    if rng is None:
        rng = random.Random()
    
    indices = list(range(n_agents))
    rng.shuffle(indices)
    
    pairs = [(indices[i], indices[i + 1]) for i in range(0, n_agents, 2)]
    return pairs


def chunk_pairs(indices: List[int], rng: Optional[random.Random] = None) -> List[Tuple[int, int]]:
    """
    Create pairs from a list of indices, shuffling first.
    
    If odd number of indices, the last agent is paired with itself.
    
    Args:
        indices: List of agent indices
        rng: Random number generator instance (optional)
        
    Returns:
        List of (agent_i, agent_j) pairs
    """
    if rng is None:
        rng = random.Random()
    
    shuffled = indices.copy()
    rng.shuffle(shuffled)
    
    pairs = []
    for i in range(0, len(shuffled) - 1, 2):
        pairs.append((shuffled[i], shuffled[i + 1]))
    
    # Handle odd case
    if len(shuffled) % 2 == 1:
        pairs.append((shuffled[-1], shuffled[-1]))
    
    return pairs


def shared_allowed_for_round(
    base_names: List[str], 
    rng: Optional[random.Random] = None
) -> List[str]:
    """
    Create a shuffled copy of allowed names for a round.
    
    This ensures both agents in a pair see the same ordering,
    which prevents permutation-driven mismatches.
    
    Args:
        base_names: List of allowed name tokens
        rng: Random number generator instance (optional)
        
    Returns:
        Shuffled copy of base_names
    """
    if rng is None:
        rng = random.Random()
    
    allowed = base_names.copy()
    rng.shuffle(allowed)
    return allowed


def uniform_model_assignment(
    n_agents: int, 
    n_models: int, 
    seed: int = 0
) -> Tuple[List[int], List[int]]:
    """
    Uniformly assign agents to models.
    
    Args:
        n_agents: Number of agents
        n_models: Number of model types
        seed: Random seed for shuffling
        
    Returns:
        Tuple of (assignment list, counts per model)
        - assignment[i] is the model index for agent i
        - counts[j] is the number of agents using model j
    """
    base = n_agents // n_models
    remainder = n_agents % n_models
    
    counts = [base] * n_models
    rng = random.Random(seed)
    
    # Distribute remainder
    for idx in rng.sample(range(n_models), remainder):
        counts[idx] += 1
    
    # Build assignment
    assignment = []
    for model_idx, count in enumerate(counts):
        assignment.extend([model_idx] * count)
    
    rng.shuffle(assignment)
    return assignment, counts
