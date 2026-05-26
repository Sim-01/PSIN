"""Run a quick PSIN demonstration on the bundled synthetic dataset."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from psin import PSIN, convolve_reflectivity, normalize_to_unit_range, toeplitz_wavelet


def ensure_synthetic_data(path: Path) -> None:
    """Create the synthetic dataset if it is missing."""
    if path.exists():
        return
    command = [sys.executable, "scripts/generate_synthetic_data.py", "--output", str(path)]
    subprocess.check_call(command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible PSIN demo.")
    parser.add_argument("--data", default="data/sample/synthetic_section.npz", help="Path to synthetic NPZ data.")
    parser.add_argument("--output-dir", default="outputs/demo", help="Directory for outputs.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs for the quick demo.")
    parser.add_argument("--n-traces", type=int, default=32, help="Number of traces used in the quick demo.")
    parser.add_argument("--n-stages", type=int, default=3, help="Number of unfolded stages.")
    parser.add_argument("--init-frequency", type=float, default=40.0, help="Ricker frequency for operator initialization.")
    parser.add_argument("--recon-frequency", type=float, default=50.0, help="Ricker frequency for displayed reconstruction.")
    parser.add_argument("--dt", type=float, default=0.002, help="Sampling interval.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    data_path = Path(args.data)
    ensure_synthetic_data(data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = np.load(data_path)
    seismic = dataset["seismic"].astype(np.float32)
    seismic = seismic[:, : args.n_traces]
    seismic = normalize_to_unit_range(seismic)

    init_operator = toeplitz_wavelet(seismic.shape[0], args.init_frequency, args.dt)
    initial_state = np.zeros_like(seismic, dtype=np.float32)

    model = PSIN(
        seismic_data=seismic,
        init_operator=init_operator,
        initial_sparse_state=initial_state,
        n_traces_train=seismic.shape[1],
        n_stages=args.n_stages,
    )
    history = model.fit(epochs=args.epochs, verbose=True)

    estimated_reflectivity = model.middle_state
    reconstructed = convolve_reflectivity(estimated_reflectivity, frequency=args.recon_frequency, dt=args.dt)
    reconstructed = normalize_to_unit_range(reconstructed)

    np.savez_compressed(
        output_dir / "demo_outputs.npz",
        input_seismic=seismic,
        estimated_reflectivity=estimated_reflectivity,
        reconstructed_seismic=reconstructed,
        epoch_losses=np.array(history.epoch_losses),
        learned_operator=model.get_operator(),
    )

    plt.figure(figsize=(10, 3))
    plt.imshow(seismic, cmap="gray", aspect="auto")
    plt.title("Input synthetic seismic section")
    plt.xlabel("Trace")
    plt.ylabel("Sample")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / "input_seismic.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.imshow(reconstructed, cmap="gray", aspect="auto")
    plt.title("PSIN reconstructed seismic section")
    plt.xlabel("Trace")
    plt.ylabel("Sample")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_dir / "reconstructed_seismic.png", dpi=200)
    plt.close()

    plt.figure(figsize=(5, 3))
    plt.plot(history.epoch_losses, marker="o")
    plt.title("PSIN training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean MSE loss")
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=200)
    plt.close()

    print(f"Demo outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
