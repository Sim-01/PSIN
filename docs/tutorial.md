# Tutorial: typical PSIN workflow

This tutorial shows the typical workflow used by PSIN.

## 1. Install the package

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Run the synthetic demo

```bash
python scripts/run_demo.py --epochs 2 --n-traces 32
```

This command will:

1. generate a synthetic sparse-reflectivity model if needed;
2. convolve it with a Ricker wavelet to create a seismic section;
3. normalize the seismic section to `[-1, 1]`;
4. initialize PSIN with a Toeplitz-Ricker operator;
5. train the model with self-supervised MSE reconstruction;
6. save the estimated sparse representation and reconstructed section.

## 3. Run a field-data experiment

Place a local seismic section under `data/private/`, for example:

```text
data/private/my_section.npy
```

Run:

```bash
python scripts/train_field_data.py --data data/private/my_section.npy --epochs 20
```

For a raw `.dat` file, specify its shape:

```bash
python scripts/train_field_data.py --data data/private/my_section.dat --shape 608x400 --epochs 20
```

## 4. Monitor the operator evolution

```bash
python scripts/operator_evolution.py --epochs 5 --n-traces 32
```

This script outputs:

- relative change of the learned operator with respect to initialization;
- deviation from the nearest Toeplitz-structured matrix.

## 5. Interpret the outputs

The main outputs are saved as compressed NumPy files. Typical keys are:

- `input_seismic`: normalized input seismic section;
- `estimated_reflectivity`: sparse reflectivity-like representation estimated by PSIN;
- `reconstructed_seismic`: seismic section reconstructed from the estimated sparse representation;
- `learned_operator`: learned convolution operator;
- `epoch_losses`: mean MSE loss per epoch.
