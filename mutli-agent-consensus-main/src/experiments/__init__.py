"""
Runnable experiment definitions.

Provides high-level interfaces for:
- Naming game ablations
- Beta calibration
- Prior probes
- Shock experiments
- Mixed-model experiments (heterogeneous cohorts)
"""

from .naming_game import NamingGameExperiment, NamingGameConfig, run_naming_game
from .beta_calibration import BetaCalibrationExperiment, BetaCalibrationConfig, run_beta_calibration
from .prior_probe import PriorProbeExperiment, PriorProbeConfig, run_prior_probe
from .shock import ShockExperiment, ShockConfig, run_shock_experiment
from .mixed_models import (
    MixedModelsExperiment, 
    MixedModelsConfig,
    CohortComposition,
    run_mixed_models,
)

__all__ = [
    # Naming game
    "NamingGameExperiment",
    "NamingGameConfig",
    "run_naming_game",
    # Beta calibration
    "BetaCalibrationExperiment",
    "BetaCalibrationConfig",
    "run_beta_calibration",
    # Prior probe
    "PriorProbeExperiment",
    "PriorProbeConfig",
    "run_prior_probe",
    # Shock
    "ShockExperiment",
    "ShockConfig",
    "run_shock_experiment",
    # Mixed models
    "MixedModelsExperiment",
    "MixedModelsConfig",
    "CohortComposition",
    "run_mixed_models",
]
