"""
OS naming game experiment for Hugging Face hosted inference APIs.

This module reuses the OS experiment logic from naming_game_os_groq and is
paired with HF API clients in the runner.
"""

from .naming_game_os_groq import NamingGameOSConfig, NamingGameOSExperiment


class NamingGameOSHFExperiment(NamingGameOSExperiment):
    """Alias class for clearer naming in HF API runner scripts."""


__all__ = [
    "NamingGameOSConfig",
    "NamingGameOSHFExperiment",
]

