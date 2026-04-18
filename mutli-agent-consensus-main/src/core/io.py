"""
Standardized I/O for experiment outputs.

This module provides canonical implementations for:
- Run directory initialization
- Config dumping with digest
- Trial logging to CSV
- Run summary JSON
- Figure saving with consistent naming
- Raw response storage

All experiments should use these functions to ensure consistent output format.
"""

import os
import csv
import json
import hashlib
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, TextIO
from dataclasses import dataclass, asdict
import random
import string


# ============================================================================
# Standard Schema Definitions (B2, B3)
# ============================================================================

# Required columns for trials.csv (experiments may add more, but never remove these)
TRIAL_SCHEMA = [
    "run_id",
    "experiment", 
    "variant_id",
    "seed",
    "repeat_idx",
    "round",
    "phase",                 # pre_shock/post_shock/na
    "agent_id",
    "agent_type",            # model preset name or type id
    "partner_id",
    "partner_type",
    "provider",
    "model_name",
    "temperature",
    "H",                     # history length
    "N",                     # number of agents
    "W",                     # number of words/labels
    "prompt_hash",
    "allowed_set_id",
    "allowed_order_id",
    "prompt_variant",        # scored/structure_only/no_score etc
    "choice",
    "choice_valid",
    "partner_choice",
    "match",
    "reward",
    "cum_reward",
    "latency_ms",
    "retries",
    "raw_response_path",     # optional
]

# Required fields for run_summary.json
RUN_SUMMARY_SCHEMA = [
    "run_id",
    "experiment",
    "variant_id",
    "config_digest",
    "n_trials_total",
    "n_invalid",
    "n_retries_total",
    "final_match_rate",
    "time_to_consensus",     # null if not defined
    "dominant_share_final",
    "entropy_final",
    "within_type_match_final",  # null if homogeneous
    "cross_type_match_final",   # null if homogeneous
    "notes",                 # dict for extra derived stats
]


# ============================================================================
# Directory and File Management
# ============================================================================

def generate_run_id(
    experiment: str,
    model: Optional[str] = None,
    variant: Optional[str] = None,
    seed: Optional[int] = None,
    extra_tags: Optional[List[str]] = None,
    length: int = 4,
) -> str:
    """
    Generate unique run identifier with descriptive components.
    
    Format: {experiment}_{model}_{variant}_{timestamp}_{seed}_{rand}
    
    Examples:
        - naming_game_gpt4o-mini_scored_20240131_143052_s12345_a3f2
        - beta_calibration_gemini-flash_20240131_143052_s42_x9k1
        - mixed_models_gpt4o-mini+gemini-flash_20240131_143052_s0_b2c4
    
    Args:
        experiment: Experiment type name
        model: Model name or preset (cleaned for filesystem)
        variant: Condition/variant name (e.g., "scored", "structure_only")
        seed: Random seed used
        extra_tags: Additional identifier tags
        length: Length of random suffix
        
    Returns:
        Unique run identifier string
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    
    parts = [experiment]
    
    if model:
        # Clean model name for filesystem (replace slashes, colons)
        clean_model = model.replace("/", "-").replace(":", "-").replace(" ", "_")
        # Truncate very long model names
        if len(clean_model) > 30:
            clean_model = clean_model[:30]
        parts.append(clean_model)
    
    if variant:
        parts.append(variant)
    
    parts.append(timestamp)
    
    if seed is not None:
        parts.append(f"s{seed}")
    
    if extra_tags:
        parts.extend(extra_tags)
    
    parts.append(rand_suffix)
    
    return "_".join(parts)


def init_run_dir(
    out_dir: str,
    experiment: str,
    run_name: Optional[str] = None,
    model: Optional[str] = None,
    variant: Optional[str] = None,
    seed: Optional[int] = None,
    extra_tags: Optional[List[str]] = None,
) -> tuple:
    """
    Initialize run directory structure with descriptive naming.
    
    Creates:
        {out_dir}/{experiment}/{run_id}/
        {out_dir}/{experiment}/{run_id}/figures/
        {out_dir}/{experiment}/{run_id}/raw/
    
    The run_id includes model, variant, timestamp, and seed for easy identification.
    
    Args:
        out_dir: Base output directory
        experiment: Experiment type name
        run_name: Optional custom run name (overrides auto-generation)
        model: Model name/preset for identification
        variant: Condition/variant name
        seed: Random seed used
        extra_tags: Additional identifier tags
        
    Returns:
        Tuple of (run_id, run_path)
        
    Examples:
        >>> init_run_dir("results", "naming_game", model="gpt4o-mini", variant="scored", seed=12345)
        ('naming_game_gpt4o-mini_scored_20240131_143052_s12345_a3f2', 
         'results/naming_game/naming_game_gpt4o-mini_scored_20240131_143052_s12345_a3f2')
    """
    if run_name:
        run_id = run_name
    else:
        run_id = generate_run_id(
            experiment=experiment,
            model=model,
            variant=variant,
            seed=seed,
            extra_tags=extra_tags,
        )
    
    run_path = Path(out_dir) / experiment / run_id
    
    # Create directories
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "figures").mkdir(exist_ok=True)
    (run_path / "raw").mkdir(exist_ok=True)
    
    return run_id, str(run_path)


def compute_config_digest(config_dict: Dict[str, Any]) -> str:
    """Compute SHA256 digest of canonical config JSON."""
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def dump_resolved_config(
    run_path: str,
    config_dict: Dict[str, Any],
) -> str:
    """
    Write resolved config to run directory.
    
    Args:
        run_path: Path to run directory
        config_dict: Full resolved configuration
        
    Returns:
        Config digest string
    """
    config_path = Path(run_path) / "config.yaml"
    
    try:
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback to JSON if yaml not available
        config_path = Path(run_path) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    return compute_config_digest(config_dict)


# ============================================================================
# Trial Logging
# ============================================================================

@dataclass
class TrialRow:
    """
    A single trial row for logging.
    
    Use this dataclass to ensure consistent schema.
    """
    run_id: str
    experiment: str
    variant_id: str
    seed: int
    repeat_idx: int
    round: int
    phase: str = "na"
    agent_id: int = 0
    agent_type: str = ""
    partner_id: int = 0
    partner_type: str = ""
    provider: str = ""
    model_name: str = ""
    temperature: float = 1.0
    H: int = 0
    N: int = 0
    W: int = 0
    prompt_hash: str = ""
    allowed_set_id: str = ""
    allowed_order_id: str = ""
    prompt_variant: str = ""
    choice: str = ""
    choice_valid: bool = True
    partner_choice: str = ""
    match: bool = False
    reward: float = 0.0
    cum_reward: float = 0.0
    latency_ms: float = 0.0
    retries: int = 0
    raw_response_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TrialsWriter:
    """
    CSV writer for trial logs with schema enforcement.
    
    Usage:
        writer = TrialsWriter(run_path, extra_columns=["my_col"])
        writer.log(row_dict)
        writer.close()
    
    Or use as context manager:
        with open_trials_writer(run_path) as writer:
            writer.log(row_dict)
    """
    
    def __init__(
        self,
        run_path: str,
        extra_columns: Optional[List[str]] = None,
        filename: str = "trials.csv",
    ):
        self.run_path = Path(run_path)
        self.columns = TRIAL_SCHEMA.copy()
        if extra_columns:
            self.columns.extend(extra_columns)
        
        self.filepath = self.run_path / filename
        self.file: Optional[TextIO] = None
        self.writer: Optional[csv.DictWriter] = None
        self._open()
    
    def _open(self):
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.columns)
        self.writer.writeheader()
    
    def log(self, row: Dict[str, Any]) -> None:
        """
        Log a trial row.
        
        Missing columns are filled with empty string.
        Extra columns not in schema are ignored.
        """
        # Ensure all required columns exist
        sanitized = {}
        for col in self.columns:
            sanitized[col] = row.get(col, "")
        
        self.writer.writerow(sanitized)
        self.file.flush()  # Ensure data is written
    
    def log_trial_row(self, row: TrialRow) -> None:
        """Log a TrialRow dataclass instance."""
        self.log(row.to_dict())
    
    def close(self) -> None:
        if self.file:
            self.file.close()
            self.file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def open_trials_writer(
    run_path: str,
    extra_columns: Optional[List[str]] = None,
) -> TrialsWriter:
    """
    Open a trials CSV writer.
    
    Args:
        run_path: Path to run directory
        extra_columns: Additional columns beyond standard schema
        
    Returns:
        TrialsWriter instance
    """
    return TrialsWriter(run_path, extra_columns)


# ============================================================================
# Run Summary
# ============================================================================

def write_run_summary(
    run_path: str,
    summary: Dict[str, Any],
) -> None:
    """
    Write run summary JSON.
    
    Args:
        run_path: Path to run directory
        summary: Summary statistics dictionary
    """
    # Ensure required fields exist (fill with None if missing)
    for field in RUN_SUMMARY_SCHEMA:
        if field not in summary:
            summary[field] = None
    
    summary_path = Path(run_path) / "run_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)


def load_run_summary(run_path: str) -> Dict[str, Any]:
    """Load run summary from directory."""
    summary_path = Path(run_path) / "run_summary.json"
    with open(summary_path) as f:
        return json.load(f)


# ============================================================================
# Raw Response Storage
# ============================================================================

def save_raw_response(
    run_path: str,
    key: str,
    content: Any,
    format: str = "json",
) -> str:
    """
    Save raw API response for debugging/reproducibility.
    
    Args:
        run_path: Path to run directory
        key: Unique identifier for this response
        content: Response content (dict or string)
        format: "json" or "txt"
        
    Returns:
        Path to saved file (relative to run_path)
    """
    raw_dir = Path(run_path) / "raw"
    raw_dir.mkdir(exist_ok=True)
    
    if format == "json":
        filepath = raw_dir / f"{key}.json"
        with open(filepath, 'w') as f:
            json.dump(content, f, indent=2, default=str)
    else:
        filepath = raw_dir / f"{key}.txt"
        with open(filepath, 'w') as f:
            f.write(str(content))
    
    return f"raw/{key}.{format}"


# ============================================================================
# Figure Saving
# ============================================================================

def save_figure(
    run_path: str,
    fig,
    name: str,
    formats: List[str] = None,
    dpi: int = 300,
) -> List[str]:
    """
    Save figure to run's figures directory.
    
    Args:
        run_path: Path to run directory
        fig: Matplotlib figure object
        name: Base filename (without extension)
        formats: List of formats to save (default: ["png", "pdf"])
        dpi: Resolution for raster formats
        
    Returns:
        List of saved file paths
    """
    if formats is None:
        formats = ["png"]
    
    figures_dir = Path(run_path) / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    saved = []
    for fmt in formats:
        filepath = figures_dir / f"{name}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        saved.append(str(filepath))
    
    return saved


# ============================================================================
# Finalization
# ============================================================================

def finalize_run(
    run_path: str,
    summary: Dict[str, Any],
    trials_writer: Optional[TrialsWriter] = None,
) -> None:
    """
    Finalize a run: close files and write summary.
    
    Args:
        run_path: Path to run directory
        summary: Summary statistics dictionary
        trials_writer: Optional open trials writer to close
    """
    if trials_writer:
        trials_writer.close()
    
    write_run_summary(run_path, summary)


# ============================================================================
# Utility Functions
# ============================================================================

def hash_prompt(prompt: str, length: int = 8) -> str:
    """Generate short hash of prompt for logging."""
    return hashlib.md5(prompt.encode()).hexdigest()[:length]


def list_runs(out_dir: str, experiment: str = None) -> List[str]:
    """List all run directories."""
    base = Path(out_dir)
    if not base.exists():
        return []
    
    runs = []
    if experiment:
        exp_dir = base / experiment
        if exp_dir.exists():
            runs = [str(p) for p in exp_dir.iterdir() if p.is_dir()]
    else:
        for exp_dir in base.iterdir():
            if exp_dir.is_dir():
                runs.extend([str(p) for p in exp_dir.iterdir() if p.is_dir()])
    
    return sorted(runs)


def load_trials(run_path: str) -> "pd.DataFrame":
    """Load trials CSV as DataFrame."""
    import pandas as pd
    return pd.read_csv(Path(run_path) / "trials.csv")
