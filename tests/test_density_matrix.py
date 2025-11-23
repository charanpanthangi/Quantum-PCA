"""Tests for density matrix construction."""
import numpy as np

from app import density_matrix


def test_density_matrix_properties():
    states = [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]
    rho = density_matrix.construct_density_matrix(states)
    assert rho.shape == (2, 2)
    # Hermitian: rho should equal its conjugate transpose
    assert np.allclose(rho, rho.conj().T)
    # Trace should be close to 1
    assert np.isclose(np.trace(rho), 1.0)
