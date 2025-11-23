"""
Command-line interface for running the simplified Quantum PCA demo.
The script wires together dataset creation, encoding, density matrix
construction, qPCA estimation, classical baseline, and SVG plotting.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from app import classical_pca, dataset, density_matrix, encoding, plots, qpca


def run_pipeline(samples: int = 50, components: int = 2, dataset_kind: str = "gaussian") -> dict:
    """Execute the end-to-end qPCA workflow.

    Returns a dictionary with intermediate results for reuse in notebooks or tests.
    """

    raw_data = dataset.load_dataset(kind=dataset_kind, samples=samples)

    encoded_states = [encoding.encode_sample(sample, method="angle") for sample in raw_data]
    rho = density_matrix.construct_density_matrix(encoded_states)

    qpca_vals, qpca_vecs = qpca.qpca_components(rho, k=components)
    classical_vals, classical_vecs = classical_pca.run_classical_pca(
        raw_data, components=components
    )

    plots_dir = Path("examples")
    plots_dir.mkdir(exist_ok=True)

    plots.plot_eigenvalues(
        qpca_vals,
        classical_vals,
        output_path=plots_dir / "qpca_eigenvalues.svg",
    )
    plots.plot_variance_explained(qpca_vals, output_path=plots_dir / "qpca_variance_plot.svg")
    plots.plot_state_visualization(encoded_states[: min(20, len(encoded_states))], output_path=plots_dir / "qpca_state_visualization.svg")

    return {
        "data": raw_data,
        "encoded_states": encoded_states,
        "density_matrix": rho,
        "qpca_values": qpca_vals,
        "qpca_vectors": qpca_vecs,
        "classical_values": classical_vals,
        "classical_vectors": classical_vecs,
    }


def main() -> None:
    """Parse arguments and run the qPCA demo."""

    parser = argparse.ArgumentParser(description="Run a simplified qPCA demonstration.")
    parser.add_argument("--samples", type=int, default=50, help="Number of data samples")
    parser.add_argument(
        "--components", type=int, default=2, help="Number of principal components to keep"
    )
    parser.add_argument(
        "--dataset", type=str, default="gaussian", choices=["gaussian", "binary"],
        help="Dataset kind to generate",
    )
    args = parser.parse_args()

    results = run_pipeline(samples=args.samples, components=args.components, dataset_kind=args.dataset)

    print("qPCA eigenvalues:", np.round(results["qpca_values"], 3))
    print("Classical PCA eigenvalues:", np.round(results["classical_values"], 3))
    print("SVG plots saved in examples/ directory.")


if __name__ == "__main__":
    main()
