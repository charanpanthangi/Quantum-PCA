"""
Classical PCA baseline using scikit-learn.
The baseline makes it easy to compare quantum-inspired estimates against
standard principal component analysis.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def run_classical_pca(data: np.ndarray, components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Run classical PCA on the provided data.

    Args:
        data: 2D array of shape (samples, features).
        components: Number of principal components to keep.

    Returns:
        Tuple of (eigenvalues, eigenvectors) similar to the qPCA outputs.
    """

    pca = PCA(n_components=components)
    pca.fit(data)

    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_.T
    return eigenvalues, eigenvectors
