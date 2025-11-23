"""Tests for encoding utilities."""
import numpy as np

from app import encoding


def test_angle_encoding_output_shape():
    state = encoding.angle_encoding(np.array([0.1, -0.2]))
    assert state.shape == (2,)
    assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-6)


def test_amplitude_encoding_output_shape():
    state = encoding.amplitude_encoding(np.array([1.0, 0.0]))
    assert state.shape == (2,)
    assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-6)
