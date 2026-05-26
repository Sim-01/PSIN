# PSIN: Physics-Informed Self-Supervised ISTA-Unrolled Network

This repository contains the reference implementation for the manuscript:

**A Physics-Informed Self-Supervised ISTA-Unrolled Network for Sparse Spike Deconvolution**

PSIN is a physics-informed, self-supervised, weakly unrolled network for sparse spike deconvolution and seismic-resolution enhancement. The method embeds an ISTA-inspired sparse-inference update into a trainable architecture, initializes the convolution operator with a Toeplitz-Ricker physical prior, and learns the effective operator and iterative coefficients through self-supervised seismic reconstruction.

## Repository contents

```text
PSIN_repository/
├── LICENSE
├── README.md
├── requirements.txt
├── environment.yml
├── CITATION.cff
├── src/psin/
│   ├── __init__.py
│   ├── io.py
│   ├── model.py
│   └── wavelet.py
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── run_demo.py
│   ├── train_field_data.py
│   └── operator_evolution.py
├── data/
│   ├── sample/
│   │   └── synthetic_section.npz
│   └── field/
│       └── README.md
├── docs/
│   ├── compute_requirements.md
│   ├── data_availability.md
│   ├── tutorial.md
│   └── user_guide.md
└── outputs/
```

## Installation

Clone the repository and install the dependencies:

```bash
git clone <YOUR_REPOSITORY_URL>.git
cd PSIN_repository
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
```

Alternatively, create a conda environment:

```bash
conda env create -f environment.yml
conda activate psin
pip install -e .
```

## Quick reproducibility demo

The repository includes a small synthetic example so reviewers can run PSIN without access to proprietary field data.

```bash
python scripts/run_demo.py --epochs 2 --n-traces 32
```

The demo generates or loads `data/sample/synthetic_section.npz`, trains PSIN for a small number of epochs, and writes results to:

```text
outputs/demo/
├── demo_outputs.npz
├── input_seismic.png
├── reconstructed_seismic.png
└── training_loss.png
```

The quick demo is intentionally small and is intended to verify the computational workflow. It is not meant to reproduce the full field-data figures in the manuscript.

## Run PSIN on a local seismic section

For a NumPy file:

```bash
python scripts/train_field_data.py \
  --data data/private/my_section.npy \
  --epochs 20 \
  --n-stages 3 \
  --init-frequency 40 \
  --recon-frequency 50 \
  --dt 0.002
```

For a raw `.dat` file, provide the shape:

```bash
python scripts/train_field_data.py \
  --data data/private/kumano2_608x400.dat \
  --shape 608x400 \
  --epochs 20
```

For a MATLAB file:

```bash
python scripts/train_field_data.py \
  --data data/private/Lu203.mat \
  --mat-key Lu203 \
  --epochs 20
```

Outputs are saved in `outputs/field_run/` by default.

## Operator-evolution experiment

To monitor the relative change of the learned operator and its deviation from the nearest Toeplitz-structured matrix, run:

```bash
python scripts/operator_evolution.py --epochs 5 --n-traces 32
```

For a local field dataset:

```bash
python scripts/operator_evolution.py \
  --data data/private/Lu203_crop.npy \
  --epochs 30 \
  --n-traces 501
```

The output file `operator_evolution.npz` contains the relative operator change and Toeplitz-deviation curves.

## Main options

| Option | Meaning | Default |
|---|---|---|
| `--epochs` | Number of training epochs | `20` for field script, `2` for quick demo |
| `--n-stages` | Number of weakly unrolled stages | `3` |
| `--init-frequency` | Ricker frequency for Toeplitz initialization | `40` Hz |
| `--recon-frequency` | Ricker frequency for displayed seismic reconstruction | `50` Hz |
| `--dt` | Sampling interval | `0.002` s |
| `--n-traces` | Number of traces used in a run | all traces or user-specified |
| `--shape` | Shape of raw `.dat` data | required for `.dat` |
| `--mat-key` | MATLAB variable name | inferred if omitted |

## Data availability

The repository includes a synthetic dataset and a generator script for reproducibility. Some field datasets used in the manuscript may be subject to data-use restrictions and are therefore not redistributed in this public repository. See `docs/data_availability.md` for details and instructions for using local field data.

## Expected input and output

Input seismic sections should be two-dimensional arrays with shape:

```text
(n_samples, n_traces)
```

The main outputs are:

- estimated sparse reflectivity-like representation;
- reconstructed seismic section synthesized with a Ricker wavelet;
- learned convolution operator;
- training loss history;
- optional operator-evolution curves.

Detailed usage information is provided in `docs/user_guide.md`.

## License

This repository is released under the MIT License. See `LICENSE` for details.

## Citation

If you use this code, please cite the manuscript and this repository. A preliminary citation file is provided in `CITATION.cff`.
