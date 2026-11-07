"""Slow eigenpairs of the generator, computed as a batch (Procedure 1).

The slow eigenpairs are the dominant ones of ``B = A^{-1}``.  They are obtained
by a two-sided power iteration on ``Bp = B**pw`` (``pw`` a power of two reached
by squarings), with deflation after each mode and a guard that drops modes whose
left and right iterates are nearly orthogonal (Eq. 7 of the paper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

EPS = np.finfo(float).eps


@dataclass
class SlowModes:
    """Retained eigenpairs, stacked over pages.

    ``lam`` (n_pages, P) real eigenvalues; dropped modes are flagged in ``kept``
    and carry ``lam = 1`` and zeroed vectors so that every formula stays finite.
    """

    lam: np.ndarray        # (kf, P)
    V: np.ndarray          # (kf, m, P)  right eigenvectors
    W: np.ndarray          # (kf, m, P)  left eigenvectors (columns: w_s)
    kept: np.ndarray       # (kf, P) bool
    gamma: np.ndarray      # (kf, P) overlap <b, a>, diagnostics

    @property
    def n_modes(self) -> int:
        return self.lam.shape[1]

    @property
    def max_kept(self) -> int:
        """Largest number of retained modes on any page."""
        return int(self.kept.sum(axis=1).max()) if self.kept.size else 0

    def masked(self, arr: np.ndarray) -> np.ndarray:
        """Zero out the entries of a (kf, P) array that belong to dropped modes."""
        return np.where(self.kept, arr, 0.0)


def power_iteration_slow_modes(
    A: np.ndarray,
    B: np.ndarray,
    n_modes: int = 2,
    pw: int = 8,
    nit: int = 8,
    tau_g: float = 1e-8,
    stability_check: bool = False,
) -> SlowModes:
    """Procedure 1 of the paper: batched two-sided power iteration with deflation."""
    kf, m, _ = A.shape
    if pw < 1 or (pw & (pw - 1)) != 0:
        raise ValueError("pw must be a power of two")
    n_modes = min(n_modes, m)

    Bp = B.copy()
    for _ in range(int(round(np.log2(pw)))):
        Bp = Bp @ Bp

    lam = np.ones((kf, n_modes))
    V = np.zeros((kf, m, n_modes))
    Wm = np.zeros((kf, m, n_modes))
    kept = np.zeros((kf, n_modes), dtype=bool)
    gamma = np.zeros((kf, n_modes))

    for k in range(n_modes):
        # the left iteration must see the *deflated* operator of this pass
        BpT = np.swapaxes(Bp, 1, 2)
        a = np.ones((kf, m))
        b = np.ones((kf, m))
        for _ in range(nit):
            a = _normalise(np.einsum("kij,kj->ki", Bp, a))
            b = _normalise(np.einsum("kij,kj->ki", BpT, b))
        g = np.einsum("ki,ki->k", b, a)
        gamma[:, k] = g
        ok = np.ones(kf, dtype=bool)
        if k >= 1:
            ok &= np.abs(g) >= tau_g          # guard on the overlap
        safe = np.where(ok, g, 1.0)
        v = a / safe[:, None]
        w = b.copy()
        lam_k = np.einsum("ki,kij,kj->k", w, A, v)
        if stability_check:
            ok &= lam_k < 0.0                 # an unstable mode is an artefact
        v = np.where(ok[:, None], v, 0.0)
        w = np.where(ok[:, None], w, 0.0)
        lam_k = np.where(ok, lam_k, 1.0)
        V[:, :, k] = v
        Wm[:, :, k] = w
        lam[:, k] = lam_k
        kept[:, k] = ok
        # deflate with the (possibly zeroed) mode
        coef = lam_k ** (-pw)
        Bp = Bp - coef[:, None, None] * v[:, :, None] * w[:, None, :]

    return SlowModes(lam=lam, V=V, W=Wm, kept=kept, gamma=gamma)


def _normalise(x: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    nrm = np.linalg.norm(x, axis=1)
    return x / np.maximum(nrm, eps)[:, None]


def exact_slow_modes(A: np.ndarray, n_modes: int = 2) -> SlowModes:
    """Reference eigenpairs from LAPACK, ordered by modulus (may be complex).

    Used for diagnostics and for forward-error studies; the GAJ Jacobian assumes
    real retained modes, as in Proposition 1.
    """
    kf, m, _ = A.shape
    lam = np.zeros((kf, n_modes), dtype=complex)
    V = np.zeros((kf, m, n_modes), dtype=complex)
    Wm = np.zeros((kf, m, n_modes), dtype=complex)
    kept = np.zeros((kf, n_modes), dtype=bool)
    gamma = np.zeros((kf, n_modes))
    for p in range(kf):
        vals, vecs = np.linalg.eig(A[p])
        order = np.argsort(np.abs(vals))[:n_modes]
        inv = np.linalg.inv(vecs)
        for k, idx in enumerate(order):
            lam[p, k] = vals[idx]
            V[p, :, k] = vecs[:, idx]
            Wm[p, :, k] = inv[idx, :]
            kept[p, k] = True
            gamma[p, k] = 1.0
    # a real generator with a real slow spectrum: drop the complex dtype so the
    # downstream algebra stays real (complex pairs are flagged, not hidden)
    if np.abs(lam.imag).max() < 1e-9 * max(1.0, np.abs(lam.real).max()):
        lam = lam.real
        V = V.real
        Wm = Wm.real
    return SlowModes(lam=lam, V=V, W=Wm, kept=kept, gamma=gamma)


def spectral_diagnostics(A: np.ndarray, modes: SlowModes) -> dict:
    """H1-H3 style quantities, per page: stability, damping and the ratio rho."""
    eigs = np.linalg.eigvals(A)
    re_max = eigs.real.max(axis=1)
    lam = np.where(modes.kept, modes.lam, np.inf)
    slow = np.abs(lam)
    fast = np.abs(eigs)
    # rho: slowest retained modulus over the fastest discarded modulus
    rho = np.empty(A.shape[0])
    for p in range(A.shape[0]):
        kept_mod = np.abs(modes.lam[p][modes.kept[p]])
        slow_max = kept_mod.max() if kept_mod.size else np.nan
        mask = np.ones(fast.shape[1], dtype=bool)
        # discard the retained eigenvalues by value
        for val in modes.lam[p][modes.kept[p]]:
            mask &= np.abs(eigs[p] - val) > 1e-9 * max(1.0, abs(val))
        fast_min = np.abs(eigs[p][mask]).min() if mask.any() else np.nan
        rho[p] = slow_max / fast_min
    return {"max_real_eigenvalue": re_max, "rho": rho,
            "d_slowest": slow.min(axis=1) if np.isfinite(slow).any() else None}
