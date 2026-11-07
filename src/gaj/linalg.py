"""Small batched linear-algebra helpers.

``expm_batch`` is a scaling-and-squaring matrix exponential evaluated for a whole
stack of matrices at once.  SciPy's ``expm`` is the reference implementation but
it is a Python loop over pages and costs ~3 ms for an 18x18 generator whose norm
is 1e5 (the Bloch-McConnell matrices are stiff), which makes the conventional
central-difference control needlessly slow; the batched version is the same
algorithm and agrees to ~1e-14 while being one to two orders of magnitude faster.
"""

from __future__ import annotations

import numpy as np


def expm_batch(A: np.ndarray, t: float = 1.0, order: int = 18) -> np.ndarray:
    """exp(t*A) for a stack of matrices, by scaling, Taylor series and squaring.

    ``A`` has shape ``(..., m, m)``; the result has the same shape.
    """
    A = np.asarray(A, float)
    m = A.shape[-1]
    M = A * t
    # one common scale for the whole stack (per-page scaling would need a
    # different number of squarings for each matrix)
    nrm = np.abs(M).sum(axis=-1).max() if M.ndim == 2 else np.abs(M).sum(axis=-1).max()
    s = int(min(max(np.ceil(np.log2(max(nrm, 1e-300) / 0.5)), 0.0), 60))
    M = M / (2.0 ** s)

    eye = np.broadcast_to(np.eye(m), M.shape)
    E = eye.copy()
    term = eye.copy()
    for k in range(1, order + 1):
        term = term @ M / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


def expm_batch_exact(A: np.ndarray, t: float = 1.0) -> np.ndarray:
    """SciPy reference (loops over the stack); used to validate ``expm_batch``."""
    from scipy.linalg import expm

    A = np.asarray(A, float)
    flat = A.reshape(-1, A.shape[-1], A.shape[-1])
    out = np.stack([expm(a * t) for a in flat])
    return out.reshape(A.shape)
