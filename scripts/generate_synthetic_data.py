"""Generate a small synthetic seismic section for reproducibility tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from psin.wavelet import convolve_reflectivity, normalize_to_unit_range


def generate_sparse_reflectivity(n_samples: int, n_traces: int, spike_probability: float, seed: int) -> np.ndarray:
    """Generate a sparse reflectivity model with laterally varying amplitudes."""
    rng = np.random.default_rng(seed)
    reflectivity = np.zeros((n_samples, n_traces), dtype=np.float32)
    mask = rng.random((n_samples, n_traces)) < spike_probability
    reflectivity[mask] = rng.normal(loc=0.0, scale=1.0, size=np.count_nonzero(mask))

    # Add a few coherent reflection events so the example is visually meaningful.
    x = np.arange(n_traces)
    for center, amplitude, period in [(35, 0.8, 40), (70, -0.7, 55), (105, 0.6, 70)]:
        event = center + (6 * np.sin(2 * np.pi * x / period)).astype(int)
        valid = (event >= 0) & (event < n_samples)
        reflectivity[event[valid], x[valid]] += amplitude

    return reflectivity


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data for PSIN.")
    parser.add_argument("--output", default="data/sample/synthetic_section.npz", help="Output NPZ file.")
    parser.add_argument("--n-samples", type=int, default=128, help="Number of time samples.")
    parser.add_argument("--n-traces", type=int, default=64, help="Number of traces.")
    parser.add_argument("--spike-probability", type=float, default=0.015, help="Sparse spike probability.")
    parser.add_argument("--frequency", type=float, default=40.0, help="Dominant frequency in Hz.")
    parser.add_argument("--dt", type=float, default=0.002, help="Sampling interval in seconds.")
    parser.add_argument("--noise-std", type=float, default=0.03, help="Gaussian noise standard deviation.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    reflectivity = generate_sparse_reflectivity(args.n_samples, args.n_traces, args.spike_probability, args.seed)
    seismic = convolve_reflectivity(reflectivity, frequency=args.frequency, dt=args.dt)
    seismic = seismic + rng.normal(0.0, args.noise_std, size=seismic.shape).astype(np.float32)
    seismic = normalize_to_unit_range(seismic)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        seismic=seismic.astype(np.float32),
        reflectivity=reflectivity.astype(np.float32),
        frequency=np.array(args.frequency),
        dt=np.array(args.dt),
    )
    print(f"Synthetic dataset saved to {output_path}")


if __name__ == "__main__":
    main()
