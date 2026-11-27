"""Small helpers: configuration loading and reproducibility."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

# Repository root (this file lives in <root>/src/human_or_ai/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Read a YAML config file into a dictionary."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Turn a path from the config into an absolute path inside the repository."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed: int) -> None:
    """Fix the Python, NumPy and TensorFlow random seeds so runs are repeatable."""
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    # Keras helper: seeds Python, NumPy and TensorFlow in one call.
    tf.keras.utils.set_random_seed(seed)
    # Make TensorFlow ops deterministic (same result on every run, on CPU).
    tf.config.experimental.enable_op_determinism()
