"""Wavelet, Toeplitz-operator, and reconstruction utilities for PSIN."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def ricker(frequency: float, dt: float) -> np.ndarray:
    """Generate a zero-phase Ricker wavelet.

    Parameters
    ----------
    frequency:
        Dominant frequency in Hz.
    dt:
        Sampling interval in seconds.

    Returns
    -------
    numpy.ndarray
        One-dimensional Ricker wavelet.
    """
    if frequency <= 0:
        raise ValueError("frequency must be positive")
    if dt <= 0:
        raise ValueError("dt must be positive")

    nw = 2.2 / frequency / dt
    nw = 2 * math.floor(nw / 2) + 1
    nc = math.floor(nw / 2)
    k = np.arange(1, nw + 1)
    alpha = (nc - k + 1) * frequency * dt * np.pi
    beta = alpha**2
    return (1.0 - 2.0 * beta) * np.exp(-beta)


def toeplitz_wavelet(n_samples: int, frequency: float, dt: float, noise_scale: float = 0.0) -> np.ndarray:
    """Construct a Toeplitz-Ricker convolution matrix.

    The wavelet is arranged around the main diagonal. This is the physical
    initialization used by PSIN. A small random perturbation can optionally be
    added by setting ``noise_scale`` to a positive value.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    w = ricker(frequency, dt)
    half_width = int(np.ceil(len(w) / 2))

    if noise_scale > 0:
        matrix = np.random.uniform(low=-noise_scale, high=noise_scale, size=(n_samples, n_samples))
    else:
        matrix = np.zeros((n_samples, n_samples), dtype=np.float32)

    for i in range(n_samples):
        matrix[i, i] = w[half_width - 1]
        for j in range(1, half_width):
            if i - j >= 0:
                matrix[i - j, i] = w[half_width - j - 1]
            if i + j <= n_samples - 1 and half_width + j - 1 < len(w):
                matrix[i + j, i] = w[half_width + j - 1]

    return matrix.astype(np.float32)


def normalize_to_unit_range(section: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a seismic section to [-1, 1]."""
    section = np.asarray(section, dtype=np.float32)
    min_value = np.min(section)
    max_value = np.max(section)
    return 2.0 * (section - min_value) / (max_value - min_value + eps) - 1.0


def convolve_reflectivity(reflectivity: np.ndarray, frequency: float = 50.0, dt: float = 0.002) -> np.ndarray:
    """Synthesize a seismic section from reflectivity using a Ricker wavelet.

    Parameters
    ----------
    reflectivity:
        A two-dimensional array with shape ``(n_samples, n_traces)``.
    frequency:
        Dominant frequency of the reconstruction wavelet.
    dt:
        Sampling interval in seconds.
    """
    reflectivity = np.asarray(reflectivity, dtype=np.float32)
    if reflectivity.ndim != 2:
        raise ValueError("reflectivity must be a 2D array")

    wavelet = ricker(frequency, dt)
    n_samples, n_traces = reflectivity.shape
    full_length = n_samples + len(wavelet) - 1
    result = np.zeros((full_length, n_traces), dtype=np.float32)

    for trace_id in range(n_traces):
        result[:, trace_id] = np.convolve(reflectivity[:, trace_id], wavelet, mode="full")

    start = (len(wavelet) - 1) // 2
    return result[start : start + n_samples, :]


def nearest_toeplitz_projection(matrix: np.ndarray) -> np.ndarray:
    """Project a square matrix onto the nearest Toeplitz matrix in Frobenius norm.

    The projection is obtained by averaging values along each diagonal.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    n = matrix.shape[0]
    projected = np.zeros_like(matrix, dtype=np.float32)
    for offset in range(-(n - 1), n):
        diagonal = np.diag(matrix, k=offset)
        if diagonal.size == 0:
            continue
        value = float(np.mean(diagonal))
        if offset >= 0:
            i = np.arange(0, n - offset)
            projected[i, i + offset] = value
        else:
            i = np.arange(0, n + offset)
            projected[i - offset, i] = value
    return projected
