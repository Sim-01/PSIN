# User guide

## Overview

PSIN is a physics-informed self-supervised ISTA-unrolled network for sparse spike deconvolution. It takes a 2D seismic section as input and estimates a sparse reflectivity-like representation. The reconstructed seismic section is obtained by convolving the estimated sparse representation with a Ricker wavelet.

## Input format

The input seismic section must be a two-dimensional array with shape:

```text
(n_samples, n_traces)
```

Supported file formats:

- `.npy`: loaded directly with `numpy.load`;
- `.npz`: loaded from the key `seismic` if available, otherwise from the first key;
- `.mat`: loaded with `scipy.io.loadmat`; users may specify `--mat-key`;
- `.dat`: loaded as a raw array; users must specify `--shape rowsxcols`.

All input sections are normalized to `[-1, 1]` before training.

## Main model parameters

| Parameter | Description | Default |
|---|---|---|
| `n_stages` | Number of weakly unrolled sparse-refinement stages | `3` |
| `init_frequency` | Dominant frequency of the initial Ricker wavelet | `40` Hz |
| `recon_frequency` | Dominant frequency for visualization reconstruction | `50` Hz |
| `dt` | Sampling interval | `0.002` s |
| `tau` | Fixed soft-thresholding parameter | `0.001` |
| `alpha` | Trainable back-projection coefficient | initialized as `0.05` |
| `beta_raw` | Raw trainable feedback parameter | initialized as `0.1` |
| `beta` | Negative feedback coefficient | `-softplus(beta_raw)` |

## Expected behavior

During training, PSIN minimizes the MSE between the input seismic trace and its reconstruction from the estimated sparse representation. The trainable variables are the convolution operator `W`, the back-projection coefficient `alpha`, and the raw feedback parameter `beta_raw`. The threshold `tau` is fixed in the provided implementation.

The external memory state stores the estimated sparse representation after each trace update with gradients detached. This implements the outer memory-update mechanism described in the manuscript.

## Outputs

The scripts save the following outputs:

- `estimated_reflectivity`: estimated sparse reflectivity-like representation;
- `reconstructed_seismic`: seismic section synthesized from the sparse representation;
- `learned_operator`: learned operator after training;
- `epoch_losses`: training loss curve;
- `operator_evolution.npz`: relative operator change and Toeplitz-deviation curves, when using the operator-evolution script.

## Common issues

### Raw `.dat` file cannot be loaded

Provide the shape explicitly, for example:

```bash
python scripts/train_field_data.py --data data/private/file.dat --shape 608x400
```

### CUDA is not available

The code automatically falls back to CPU. The synthetic demo is CPU-friendly, but full field sections may be slow on CPU.

### Results differ slightly between machines

Small numerical differences may occur because of floating-point operations, hardware differences, or PyTorch backend behavior. Set random seeds and use the same library versions for closer reproducibility.
