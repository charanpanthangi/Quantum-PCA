"""Tests for classical PCA baseline."""
import numpy as np

from app import classical_pca


def test_classical_pca_runs():
    data = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    vals, vecs = classical_pca.run_classical_pca(data, components=2)
    assert len(vals) == 2
    assert vecs.shape == (2, 2)
