# Quantum PCA (qPCA)

## What This Project Does
- Classical PCA finds main directions in data using a covariance matrix.
- Quantum PCA works on density matrices built from quantum states.
- This demo encodes simple classical data into quantum states, builds a density matrix, and extracts dominant eigenvalues.
- The quantum-like outputs are compared with classical PCA.

## Why Quantum PCA Is Interesting
- Can analyze quantum data natively.
- Uses quantum linear algebra ideas such as the SWAP test.
- Builds intuition for quantum machine learning models.

## Why SVG Instead of PNG
> GitHub’s CODEX interface cannot preview PNG/JPG and shows
> “Binary files are not supported.”
> All images in this repository are stored as lightweight SVGs to avoid diff
> issues and render cleanly inside CODEX and GitHub.

## How It Works (Plain English)
1. Generate a tiny dataset (correlated Gaussian or binary patterns).
2. Encode each sample into a quantum state with PennyLane.
3. Build an empirical density matrix from the states.
4. Use sampling intuition (SWAP tests) and small matrix diagonalization to estimate eigenvalues.
5. Compare against classical PCA results.

## How to Run
```bash
pip install -r requirements.txt
python app/main.py --samples 50 --components 2
```

## Expected Output
- Printed qPCA and classical eigenvalues.
- SVG plots saved to `examples/` showing eigenvalues, variance explained, and state locations.

## Project Structure
- `app/` contains dataset generation, encoding, density matrix construction, qPCA logic, plotting, and CLI entrypoint.
- `examples/` holds generated SVG figures.
- `notebooks/` includes a tutorial notebook.
- `tests/` contains small pytest checks.

## Future Extensions
- Run qPCA on real hardware backends.
- Explore larger density matrices.
- Implement a full phase-estimation-based version.

## License
MIT License.
