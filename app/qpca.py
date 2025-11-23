"""
Simplified Quantum PCA routines built on PennyLane.
This module demonstrates how overlaps between quantum states can reveal
principal components of a density matrix.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml


def swap_test(state1: np.ndarray, state2: np.ndarray) -> float:
    """Estimate the overlap of two states using a SWAP test.

    Args:
        state1: First statevector.
        state2: Second statevector.

    Returns:
        Estimated probability of measuring ``|0>`` on the control qubit, which
        relates to the fidelity between the two states.
    """

    dev = qml.device("default.qubit", wires=3, shots=None)

    @qml.qnode(dev)
    def circuit():
        # Prepare control qubit in |+>
        qml.Hadamard(wires=0)

        # Load the two states on separate wires.
        qml.MottonenStatePreparation(state1, wires=1)
        qml.MottonenStatePreparation(state2, wires=2)

        # Controlled-SWAP exchanges target states when control is |1>.
        qml.CSWAP(wires=[0, 1, 2])

        # Final Hadamard reveals overlap information on the control.
        qml.Hadamard(wires=0)
        return qml.probs(wires=0)

    probs = circuit()
    # Probability of measuring 0 corresponds to (1 + |<psi|phi>|^2) / 2
    return float(probs[0])


def build_overlap_matrix(encoded_states: list[np.ndarray]) -> np.ndarray:
    """Compute a Gram matrix of overlaps using SWAP tests.

    Args:
        encoded_states: List of statevectors representing encoded samples.

    Returns:
        Symmetric matrix where entry (i, j) approximates |<psi_i|psi_j>|^2.
    """

    n = len(encoded_states)
    overlaps = np.zeros((n, n))
    for i in range(n):
        overlaps[i, i] = 1.0
        for j in range(i + 1, n):
            prob_zero = swap_test(encoded_states[i], encoded_states[j])
            fidelity = 2 * prob_zero - 1
            overlaps[i, j] = overlaps[j, i] = fidelity
    return overlaps


def qpca_eigenvalues(density_matrix: np.ndarray, k: int = 2) -> np.ndarray:
    """Estimate dominant eigenvalues from the density matrix.

    Args:
        density_matrix: Empirical density matrix derived from data.
        k: Number of leading eigenvalues to return.

    Returns:
        Array of the largest ``k`` eigenvalues sorted in descending order.

    Notes:
        In a full qPCA algorithm, phase estimation would reveal eigenvalues.
        Here we use classical diagonalization on the small density matrix while
        keeping the workflow close to what a quantum routine would produce.
    """

    evals, _ = np.linalg.eigh(density_matrix)
    sorted_evals = np.sort(np.real(evals))[::-1]
    return sorted_evals[:k]


def qpca_components(density_matrix: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return dominant eigenvalues and eigenvectors for interpretation.

    Args:
        density_matrix: Empirical density matrix.
        k: Number of principal components to keep.

    Returns:
        Tuple of (eigenvalues, eigenvectors) limited to the top ``k``.
    """

    evals, evecs = np.linalg.eigh(density_matrix)
    order = np.argsort(np.real(evals))[::-1]
    evals_sorted = np.real(evals[order])[:k]
    evecs_sorted = np.real(evecs[:, order])[:, :k]
    return evals_sorted, evecs_sorted
