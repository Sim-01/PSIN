"""Train PSIN on a local field seismic section.

This script can be used by reviewers with their own data or with datasets that
are legally available to them. Proprietary field data are not required for the
quick reproducibility demo; see ``scripts/run_demo.py`` for a fully bundled test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from psin import PSIN, convolve_reflectivity, load_seismic_section, normalize_to_unit_range, toeplitz_wavelet


def parse_shape(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return None
    values = text.lower().replace("x", ",").split(",")
    if len(values) != 2:
        raise ValueError("shape must be formatted as rows,cols or rowsxcols")
    return int(values[0]), int(values[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PSIN on a local seismic section.")
    parser.add_argument("--data", required=True, help="Path to .npy, .npz, .mat, or .dat data file.")
    parser.add_argument("--shape", default=None, help="Shape for raw .dat files, e.g., 608x400.")
    parser.add_argument("--mat-key", default=None, help="Variable name for MAT files.")
    parser.add_argument("--output-dir", default="outputs/field_run", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    parser.add_argument("--n-stages", type=int, default=3, help="Number of unfolded stages.")
    parser.add_argument("--init-frequency", type=float, default=40.0, help="Initial Ricker frequency.")
    parser.add_argument("--recon-frequency", type=float, default=50.0, help="Reconstruction Ricker frequency.")
    parser.add_argument("--dt", type=float, default=0.002, help="Sampling interval.")
    parser.add_argument("--n-traces", type=int, default=None, help="Optional number of traces to train.")
    args = parser.parse_args()

    shape = parse_shape(args.shape)
    seismic = load_seismic_section(args.data, shape=shape, mat_key=args.mat_key)
    seismic = normalize_to_unit_range(seismic)
    if args.n_traces is not None:
        seismic = seismic[:, : args.n_traces]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        output_dir / "field_outputs.npz",
        input_seismic=seismic,
        estimated_reflectivity=estimated_reflectivity,
        reconstructed_seismic=reconstructed,
        epoch_losses=np.array(history.epoch_losses),
        learned_operator=model.get_operator(),
    )

    plt.figure(figsize=(10, 4))
    plt.imshow(reconstructed, cmap="gray", aspect="auto")
    plt.title("PSIN reconstructed seismic section")
    plt.xlabel("Trace")
    plt.ylabel("Sample")
    plt.tight_layout()
    plt.savefig(output_dir / "reconstructed_seismic.png", dpi=200)
    plt.close()

    model.save_reflectivity_mat(str(output_dir / "output_reflectivity.mat"))
    print(f"Field run outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
