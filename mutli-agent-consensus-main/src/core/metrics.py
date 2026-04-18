"""
Metrics and analysis utilities for naming game experiments.
"""

from typing import List, Tuple, Optional
import numpy as np


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling mean over non-overlapping windows.
    
    Args:
        x: Input array
        window: Window size
        
    Returns:
        Array of windowed means
    """
    if len(x) == 0 or window <= 0:
        return np.array([])
    
    vals = []
    for i in range(0, len(x), window):
        chunk = x[i:i + window]
        vals.append(np.mean(chunk))
    
    return np.array(vals)


def rolling_bin_mean(x: np.ndarray, bin_size: int) -> np.ndarray:
    """
    Compute binned mean (alias for rolling_mean).
    
    Args:
        x: Input array
        bin_size: Bin size
        
    Returns:
        Array of binned means
    """
    if bin_size <= 1:
        return x.copy()
    
    length = (len(x) // bin_size) * bin_size
    if length == 0:
        return np.array([])
    
    return x[:length].reshape(-1, bin_size).mean(axis=1)


def aggregate_curves(curves: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate multiple run curves into mean and std.
    
    Handles curves of different lengths by padding with NaN.
    
    Args:
        curves: List of 1D arrays from different runs
        
    Returns:
        Tuple of (x_values, mean_values, std_values)
    """
    if len(curves) == 0:
        return np.array([]), np.array([]), np.array([])
    
    max_len = max(len(c) for c in curves)
    
    # Create matrix with NaN padding
    mat = np.empty((len(curves), max_len))
    mat[:] = np.nan
    
    for i, curve in enumerate(curves):
        mat[i, :len(curve)] = curve
    
    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0, ddof=1)
    x = np.arange(max_len)
    
    return x, mean, std


def summarize_bins(
    all_runs: List[List[float]], 
    bin_size: int, 
    last_k: int = 5
) -> Tuple[float, float]:
    """
    Summarize success rates using binned means.
    
    Bins each run's per-round success into non-overlapping bins,
    takes the mean per bin, then averages the last k bins.
    
    Args:
        all_runs: List of per-round success lists from multiple runs
        bin_size: Size of each bin
        last_k: Number of final bins to average
        
    Returns:
        Tuple of (mean_across_runs, std_across_runs)
    """
    lastk_vals = []
    
    for run in all_runs:
        if bin_size <= 0:
            bin_size = 1
        
        n_bins = len(run) // bin_size
        
        if n_bins == 0:
            lastk_vals.append(np.mean(run) if run else 0.0)
        else:
            run_trunc = run[:n_bins * bin_size]
            bins = np.array(run_trunc).reshape(n_bins, bin_size).mean(axis=1)
            take = bins[-last_k:] if len(bins) >= last_k else bins
            lastk_vals.append(float(np.mean(take)))
    
    return float(np.mean(lastk_vals)), float(np.std(lastk_vals))


def time_to_consensus(
    success_series: np.ndarray, 
    threshold: float = 0.9,
    window: int = 1
) -> Optional[int]:
    """
    Calculate rounds to reach consensus threshold.
    
    Args:
        success_series: Array of success rates per round
        threshold: Success threshold to reach
        window: Smoothing window size
        
    Returns:
        Round index when threshold reached, or None if never reached
    """
    if window > 1:
        smoothed = rolling_mean(success_series, window)
    else:
        smoothed = success_series
    
    for t, val in enumerate(smoothed):
        if val >= threshold:
            return t * window if window > 1 else t
    
    return None


def time_to_recoord(
    success_series: np.ndarray,
    shock_round: int,
    threshold: float = 0.8,
    streak: int = 3,
    bin_size: int = 1
) -> Optional[int]:
    """
    Calculate re-coordination time after shock.
    
    Finds the first time after shock where binned success >= threshold
    for a given consecutive streak.
    
    Args:
        success_series: Array of reward-aligned success rates
        shock_round: Round index where shock occurred
        threshold: Success threshold
        streak: Number of consecutive windows required
        bin_size: Bin size for smoothing
        
    Returns:
        Rounds after shock to re-coordinate, or None if not achieved
    """
    x = success_series
    stride = 1
    
    if bin_size > 1:
        x = rolling_bin_mean(x, bin_size)
        shock_idx = shock_round // bin_size
        stride = bin_size
    else:
        shock_idx = shock_round
    
    if len(x) == 0:
        return None
    
    for t in range(shock_idx, len(x) - streak + 1):
        if np.all(x[t:t + streak] >= threshold):
            return (t - shock_idx + 1) * stride
    
    return None


def compute_match_rate(choices_a: List[str], choices_b: List[str]) -> float:
    """
    Compute match rate between two choice lists.
    
    Args:
        choices_a: List of choices from agent A
        choices_b: List of choices from agent B
        
    Returns:
        Fraction of matching choices
    """
    if len(choices_a) != len(choices_b) or len(choices_a) == 0:
        return 0.0
    
    matches = sum(1 for a, b in zip(choices_a, choices_b) if a == b)
    return matches / len(choices_a)


def compute_entropy(choices_or_counts) -> float:
    """
    Compute entropy of choice distribution.
    
    Args:
        choices_or_counts: Either a list of choices or a dict mapping choices to counts
        
    Returns:
        Entropy in bits
    """
    from collections import Counter
    
    if isinstance(choices_or_counts, dict):
        counts = choices_or_counts
    else:
        counts = Counter(choices_or_counts)
    
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    probs = [c / total for c in counts.values() if c > 0]
    return -sum(p * np.log2(p) for p in probs)


def compute_concentration(choices_or_counts) -> float:
    """
    Compute concentration (dominant share) of choices.
    
    Returns the fraction of choices that are the most common choice.
    Higher values indicate more concentrated/consensus choices.
    
    Args:
        choices_or_counts: Either a list of choices or a dict mapping choices to counts
        
    Returns:
        Concentration ratio in [0, 1] (dominant share)
    """
    from collections import Counter
    
    if isinstance(choices_or_counts, dict):
        counts = choices_or_counts
    else:
        counts = Counter(choices_or_counts)
    
    if not counts:
        return 0.0
    
    total = sum(counts.values())
    if total == 0:
        return 0.0
    
    max_count = max(counts.values())
    return max_count / total
