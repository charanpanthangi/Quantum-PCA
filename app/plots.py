"""
Plotting helpers that only create SVG files to keep diffs clean on GitHub.
The functions produce simple vector graphics illustrating eigenvalues and states.
"""
from __future__ import annotations

import pathlib
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Force SVG backend for all plots generated here.
matplotlib.use("svg")

sns.set(style="whitegrid")


def _ensure_directory(path: str | pathlib.Path) -> None:
    """Create parent directories for a given path if they do not exist."""

    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_eigenvalues(evals_q: Iterable[float], evals_classical: Iterable[float], output_path: str) -> None:
    """Compare quantum-inspired and classical eigenvalues.

    Args:
        evals_q: Iterable of eigenvalues from qPCA.
        evals_classical: Eigenvalues from classical PCA.
        output_path: File path for the SVG output.
    """

    _ensure_directory(output_path)
    labels = [f"qPCA {i}" for i, _ in enumerate(evals_q)] + [
        f"Classical {i}" for i, _ in enumerate(evals_classical)
    ]
    values = list(evals_q) + list(evals_classical)

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=values, palette="viridis")
    plt.ylabel("Eigenvalue magnitude")
    plt.xticks(rotation=30)
    plt.title("Dominant eigenvalues: quantum vs classical")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_variance_explained(evals: Iterable[float], output_path: str) -> None:
    """Plot cumulative variance explained from a sequence of eigenvalues."""

    _ensure_directory(output_path)
    values = np.array(list(evals), dtype=float)
    total = np.sum(values) + 1e-9
    cumulative = np.cumsum(values) / total

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(values) + 1), cumulative, marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative variance explained")
    plt.ylim(0, 1.05)
    plt.title("Variance captured by leading components")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_state_visualization(states: list[np.ndarray], output_path: str) -> None:
    """Plot the Bloch sphere projection of single-qubit states.

    The goal is to show how encoded samples populate the state space.
    """

    _ensure_directory(output_path)
    bloch_points = []
    for state in states:
        # Convert statevector to Bloch sphere coordinates.
        rho = np.outer(state, np.conjugate(state))
        bloch_x = 2 * np.real(rho[0, 1])
        bloch_y = 2 * np.imag(rho[1, 0])
        bloch_z = np.real(rho[0, 0] - rho[1, 1])
        bloch_points.append((bloch_x, bloch_y, bloch_z))

    bloch_points = np.array(bloch_points)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(bloch_points[:, 0], bloch_points[:, 1], bloch_points[:, 2], c="teal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Encoded states on the Bloch sphere (projection)")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close(fig)
