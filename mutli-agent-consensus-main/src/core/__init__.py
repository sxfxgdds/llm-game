"""
Core module: pairing, prompt templates, parsing, state updates, metrics, I/O, engine.
"""

from .pairing import pair_indices, chunk_pairs, uniform_model_assignment
from .prompts import PromptBuilder, ABLATION_CONDITIONS
from .parsing import parse_choice, extract_allowed_choice
from .state import AgentState, GameState, RunResult
from .metrics import (
    rolling_mean,
    aggregate_curves,
    summarize_bins,
    time_to_consensus,
    time_to_recoord,
    compute_entropy,
    compute_concentration,
)
from .utils import (
    set_global_seed,
    ensure_dir,
    make_nonce_tokens,
    make_simple_tokens,
    timestamp,
)
from .io import (
    init_run_dir,
    dump_resolved_config,
    open_trials_writer,
    TrialsWriter,
    TrialRow,
    write_run_summary,
    save_raw_response,
    save_figure,
    finalize_run,
    TRIAL_SCHEMA,
    RUN_SUMMARY_SCHEMA,
)
from .engine import (
    EngineConfig,
    run_population_game,
    compute_run_summary,
)

__all__ = [
    # Pairing
    "pair_indices",
    "chunk_pairs",
    "uniform_model_assignment",
    # Prompts
    "PromptBuilder",
    "ABLATION_CONDITIONS",
    # Parsing
    "parse_choice",
    "extract_allowed_choice",
    # State
    "AgentState",
    "GameState",
    "RunResult",
    # Metrics
    "rolling_mean",
    "aggregate_curves",
    "summarize_bins",
    "time_to_consensus",
    "time_to_recoord",
    "compute_entropy",
    "compute_concentration",
    # Utils
    "set_global_seed",
    "ensure_dir",
    "make_nonce_tokens",
    "make_simple_tokens",
    "timestamp",
    # I/O
    "init_run_dir",
    "dump_resolved_config",
    "open_trials_writer",
    "TrialsWriter",
    "TrialRow",
    "write_run_summary",
    "save_raw_response",
    "save_figure",
    "finalize_run",
    "TRIAL_SCHEMA",
    "RUN_SUMMARY_SCHEMA",
    # Engine
    "EngineConfig",
    "run_population_game",
    "compute_run_summary",
]
