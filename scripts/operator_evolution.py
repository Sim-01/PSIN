"""Monitor adaptive changes of the PSIN operator."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from psin import PSIN, load_seismic_section, normalize_to_unit_range, toeplitz_wavelet
from psin.wavelet import nearest_toeplitz_projection


def parse_shape(text: str | None) -> tuple[int, int] | None:
    if text is None:
        return None
    values = text.lower().replace("x", ",").split(",")
    if len(values) != 2:
        raise ValueError("shape must be formatted as rows,cols or rowsxcols")
    return int(values[0]), int(values[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor adaptive evolution of W in PSIN.")
    parser.add_argument("--data", default="data/sample/synthetic_section.npz", help="Path to data file.")
    parser.add_argument("--shape", default=None, help="Shape for raw .dat files, e.g., 501x501.")
    parser.add_argument("--mat-key", default=None, help="Variable name for MAT files.")
    parser.add_argument("--output-dir", default="outputs/operator_evolution", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--n-traces", type=int, default=32, help="Number of traces used for monitoring.")
    parser.add_argument("--init-frequency", type=float, default=40.0, help="Initial Ricker frequency.")
    parser.add_argument("--dt", type=float, default=0.002, help="Sampling interval.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    if not data_path.exists() and data_path.suffix == ".npz":
        import subprocess, sys
        subprocess.check_call([sys.executable, "scripts/generate_synthetic_data.py", "--output", str(data_path)])

    seismic = load_seismic_section(str(data_path), shape=parse_shape(args.shape), mat_key=args.mat_key)
    seismic = normalize_to_unit_range(seismic)[:, : args.n_traces]

    init_operator = toeplitz_wavelet(seismic.shape[0], args.init_frequency, args.dt)
    initial_state = np.zeros_like(seismic, dtype=np.float32)
    model = PSIN(seismic, init_operator, initial_state, n_traces_train=seismic.shape[1])

    initial_w = init_operator.copy()
    delta_w = []
    toeplitz_deviation = []

    for _ in range(args.epochs):
        model.fit(epochs=1, verbose=False)
        current_w = model.get_operator()
        delta_w.append(np.linalg.norm(current_w - initial_w, ord="fro") / (np.linalg.norm(initial_w, ord="fro") + 1e-12))
        projected = nearest_toeplitz_projection(current_w)
        toeplitz_deviation.append(np.linalg.norm(current_w - projected, ord="fro") / (np.linalg.norm(current_w, ord="fro") + 1e-12))

    np.savez_compressed(output_dir / "operator_evolution.npz", delta_w=np.array(delta_w), toeplitz_deviation=np.array(toeplitz_deviation))

    plt.figure(figsize=(6, 4))
    plt.plot(delta_w, label="Relative change of W")
    plt.plot(toeplitz_deviation, label="Toeplitz deviation")
    plt.xlabel("Epoch")
    plt.ylabel("Relative value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "operator_evolution.png", dpi=200)
    plt.close()
    print(f"Operator-evolution outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
