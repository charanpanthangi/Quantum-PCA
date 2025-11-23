"""
Density matrix construction for Quantum PCA demonstrations.
Given a collection of encoded quantum states, this module builds the
empirical density matrix that captures average behavior.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml


def construct_density_matrix(encoded_states: list[np.ndarray]) -> np.ndarray:
    """Construct a density matrix from encoded quantum states.

    Args:
        encoded_states: List of statevectors representing |psi_i> states.

    Returns:
        Density matrix ``rho`` as a NumPy array with shape (2, 2).
    """

    if len(encoded_states) == 0:
        raise ValueError("At least one state is required to build a density matrix.")

    dimension = encoded_states[0].shape[0]
    rho = np.zeros((dimension, dimension), dtype=complex)
    for state in encoded_states:
        rho += qml.math.outer(state, np.conjugate(state))
    rho /= len(encoded_states)

    # Normalize trace to 1 for numerical stability.
    trace_val = np.trace(rho)
    if trace_val != 0:
        rho = rho / trace_val
    return rho
