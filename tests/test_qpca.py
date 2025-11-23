"""Tests for simplified qPCA routines."""
import numpy as np

from app import qpca


def test_qpca_eigenvalues_format():
    rho = np.array([[0.6, 0.0], [0.0, 0.4]])
    vals = qpca.qpca_eigenvalues(rho, k=2)
    assert len(vals) == 2
    assert np.all(vals <= 1.0)


def test_swap_test_overlap():
    state = np.array([1, 0], dtype=complex)
    prob = qpca.swap_test(state, state)
    # Identical states should yield probability near 1
    assert np.isclose(prob, 1.0)
