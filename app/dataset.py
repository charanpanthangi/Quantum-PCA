"""
Dataset utilities for simplified Quantum PCA demonstrations.
The functions here generate small synthetic datasets that are easy to
understand and can be encoded into quantum states.
"""
from __future__ import annotations

import numpy as np


def generate_correlated_gaussian(samples: int = 50, seed: int | None = 42) -> np.ndarray:
    """Generate a simple 2D correlated Gaussian dataset.

    Args:
        samples: Number of data points to draw.
        seed: Optional random seed for reproducibility.

    Returns:
        A NumPy array with shape ``(samples, 2)`` containing centered data.

    Notes:
        The correlation makes one direction carry more variance than the other,
        which is helpful when showing how PCA finds principal directions.
    """

    rng = np.random.default_rng(seed)
    mean = np.array([0.0, 0.0])
    covariance = np.array([[1.0, 0.8], [0.8, 0.6]])
    data = rng.multivariate_normal(mean, covariance, size=samples)

    # Center and normalize so that encoding behaves nicely.
    data -= np.mean(data, axis=0)
    max_val = np.max(np.abs(data)) or 1.0
    data /= max_val
    return data


def generate_binary_patterns(samples: int = 6) -> np.ndarray:
    """Create a tiny set of binary patterns that resemble classical bits.

    Args:
        samples: Number of patterns to generate. The function cycles through
            a small pattern bank so any number can be produced.

    Returns:
        A NumPy array with shape ``(samples, 2)`` containing values in ``{-1, 1}``.

    Notes:
        These patterns are intentionally simple so they can be translated into
        single-qubit rotations without complicated preprocessing.
    """

    base_patterns = np.array(
        [
            [1, 1],
            [1, -1],
            [-1, 1],
            [-1, -1],
            [1, 0.5],
            [0.5, 1],
        ]
    )
    tiled = np.tile(base_patterns, (int(np.ceil(samples / len(base_patterns))), 1))
    data = tiled[:samples]

    # Normalize to the range [-1, 1] for stable rotation angles.
    data = data.astype(float)
    max_val = np.max(np.abs(data)) or 1.0
    data /= max_val
    return data


def load_dataset(kind: str = "gaussian", samples: int = 50) -> np.ndarray:
    """Load a dataset according to the chosen kind.

    Args:
        kind: Either ``"gaussian"`` or ``"binary"`` to select the generator.
        samples: Number of data points to create.

    Returns:
        A 2D NumPy array ready for encoding.
    """

    if kind == "gaussian":
        return generate_correlated_gaussian(samples=samples)
    if kind == "binary":
        return generate_binary_patterns(samples=samples)
    raise ValueError("Unsupported dataset kind. Choose 'gaussian' or 'binary'.")
