"""
Encoding helpers for turning classical samples into quantum states.
We use PennyLane to create simple single-qubit and two-qubit state preparations.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml


def angle_encoding(sample: np.ndarray) -> np.ndarray:
    """Perform angle encoding for a 2D sample.

    Each feature controls a rotation so that the angles reflect the magnitude
    of the value. This keeps the encoding hardware-friendly.

    Args:
        sample: Array with two values.

    Returns:
        The statevector produced by the encoding circuit.
    """

    dev = qml.device("default.qubit", wires=1, shots=None)

    @qml.qnode(dev)
    def circuit():
        qml.RY(float(sample[0]) * np.pi / 2, wires=0)
        qml.RZ(float(sample[1]) * np.pi / 2, wires=0)
        return qml.state()

    return circuit()


def amplitude_encoding(sample: np.ndarray) -> np.ndarray:
    """Encode a 2D vector as amplitudes on a single qubit.

    Args:
        sample: Array with two values.

    Returns:
        A normalized statevector with two amplitudes.
    """

    normalized = sample / (np.linalg.norm(sample) + 1e-9)
    dev = qml.device("default.qubit", wires=1, shots=None)

    @qml.qnode(dev)
    def circuit():
        qml.MottonenStatePreparation(
            np.array([normalized[0], normalized[1]], dtype=float), wires=0
        )
        return qml.state()

    return circuit()


def encode_sample(sample: np.ndarray, method: str = "angle") -> np.ndarray:
    """Encode a sample using the requested method.

    Args:
        sample: A 1D array with two entries.
        method: ``"angle"`` or ``"amplitude"``.

    Returns:
        A statevector representing the encoded quantum state.
    """

    if method == "angle":
        return angle_encoding(sample)
    if method == "amplitude":
        return amplitude_encoding(sample)
    raise ValueError("Encoding method must be 'angle' or 'amplitude'.")
