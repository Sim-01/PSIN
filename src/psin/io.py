"""Input-output helpers for seismic sections."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.io as sio


def load_seismic_section(
    path: str,
    shape: Optional[Tuple[int, int]] = None,
    mat_key: Optional[str] = None,
    dtype: str = "float32",
) -> np.ndarray:
    """Load a seismic section from ``.npy``, ``.npz``, ``.mat``, or raw ``.dat``.

    Raw ``.dat`` files require the ``shape`` argument because shape metadata is
    not stored in the file.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        data = np.load(file_path)
    elif suffix == ".npz":
        archive = np.load(file_path)
        if "seismic" in archive:
            data = archive["seismic"]
        else:
            first_key = list(archive.keys())[0]
            data = archive[first_key]
    elif suffix == ".mat":
        mat = sio.loadmat(file_path)
        if mat_key is None:
            candidates = [key for key in mat.keys() if not key.startswith("__")]
            if not candidates:
                raise ValueError("No data variable found in the MAT file")
            mat_key = candidates[0]
        data = mat[mat_key]
    elif suffix == ".dat":
        if shape is None:
            raise ValueError("shape must be provided for raw .dat files")
        data = np.fromfile(file_path, dtype=dtype).reshape(shape)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return np.asarray(data, dtype=np.float32)
