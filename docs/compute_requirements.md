# Computational requirements

## Software

The code was prepared for Python 3.10 or later. The main Python dependencies are:

- NumPy
- SciPy
- Matplotlib
- PyTorch
- tqdm

Install them with:

```bash
pip install -r requirements.txt
```

## Hardware

The quick synthetic demo can be executed on a standard CPU laptop. A CUDA-capable GPU is recommended for larger field sections because PSIN trains trace by trace and the operator matrix has size `(n_samples, n_samples)`.

Approximate guidance:

| Dataset size | Suggested hardware |
|---|---|
| Synthetic demo, 128 x 64 | CPU is sufficient |
| Small field crop, about 501 x 501 | GPU recommended |
| Full field section larger than 600 x 400 | GPU recommended; runtime depends on epochs and trace count |

## Memory considerations

The operator matrix has size `(n_samples, n_samples)`. For a section with 501 samples, this is about 251,001 floating-point values. Larger sample lengths require more memory and longer training time.
