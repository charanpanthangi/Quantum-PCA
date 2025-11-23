"""Tests for dataset generation functions."""
from app import dataset


def test_gaussian_shape():
    data = dataset.generate_correlated_gaussian(samples=10)
    assert data.shape == (10, 2)


def test_binary_shape():
    data = dataset.generate_binary_patterns(samples=5)
    assert data.shape == (5, 2)
