"""
Configuration management for experiments.

Provides YAML-based configuration loading and validation.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    if not _HAS_YAML:
        raise RuntimeError("PyYAML required for config loading. pip install pyyaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config or {}


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save YAML file
    """
    if not _HAS_YAML:
        raise RuntimeError("PyYAML required for config saving. pip install pyyaml")
    
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_config_dir() -> Path:
    """Get the config directory path."""
    return Path(__file__).parent


def list_configs() -> Dict[str, Path]:
    """
    List available configuration files.
    
    Returns:
        Dictionary mapping config names to file paths
    """
    config_dir = get_config_dir()
    configs = {}
    
    for yaml_file in config_dir.glob("**/*.yaml"):
        name = yaml_file.stem
        configs[name] = yaml_file
    
    return configs
