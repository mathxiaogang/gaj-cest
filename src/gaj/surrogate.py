"""Forward evaluation of the surrogate response (Section 4.2, Eq. 5).

    Z_S(x) = o^T Mss + sum_{s in S} beta_s alpha_s exp(lam_s Ts)

with ``Mss = -B c``, ``B = A^{-1}``, ``e0 = M0 - Mss``, ``alpha_s = w_s^T e0``
and ``beta_s = o^T v_s``.  Everything is evaluated for all pages -- and, when
needed, for a stack of parameter vectors -- as one batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import expm

from .eigen import SlowModes, power_iteration_slow_modes, exact_slow_modes
from .model import LinearDriveObserveModel, PageSet


@dataclass
class GAJConfig:
    """Settings of Table 2 of the paper (defaults are the paper's values)."""

    n_modes: int = 2            # P: candidate slow modes
    pw: int = 8                 # power of B in the power iteration (2**sw)
    nit: int = 8                # power-iteration steps per mode
    tau_g: float = 1e-8         # guard threshold on gamma
    qmax: int = 12              # maximum order of the coupling series
    tau: float = 1e-8           # gate tolerance
    sh: float = 1e-4            # relative step of the central-difference fallback
    stability_check: bool = False
    use_guard: bool = True
    eig: str = "power"          # "power" (Procedure 1) or "exact" (LAPACK)

    def with_(self, **kw) -> "GAJConfig":
        out = GAJConfig(**self.__dict__)
        for k, v in kw.items():
            setattr(out, k, v)
        return out


@dataclass
class ForwardState:
    """Everything one forward evaluation leaves behind for the Jacobian."""

    x: np.ndarray               # (nx, p)
    pages: PageSet
    B: np.ndarray               # (nx, kf, m, m)
    Mss: np.ndarray             # (nx, kf, m)
    e0: np.ndarray              # (nx, kf, m)
    modes: SlowModes            # flattened over (nx, kf)
    alpha: np.ndarray           # (nx, kf, P)
    beta: np.ndarray            # (nx, kf, P)
    decay: np.ndarray           # (nx, kf, P)  exp(lam_s Ts)
    Z: np.ndarray               # (nx, kf)
    n_pages: int
    shape_m: int

    def take(self, i: int) -> "ForwardState":
        """One parameter vector out of the stack (keeps the page structure)."""
        kf = self.n_pages
        sl = slice(i * kf, (i + 1) * kf)
        return ForwardState(
            x=self.x[i:i + 1], pages=self.pages,
            B=self.B[i:i + 1], Mss=self.Mss[i:i + 1], e0=self.e0[i:i + 1],
            modes=SlowModes(lam=self.modes.lam[sl], V=self.modes.V[sl],
                            W=self.modes.W[sl], kept=self.modes.kept[sl],
                            gamma=self.modes.gamma[sl]),
            alpha=self.alpha[i:i + 1], beta=self.beta[i:i + 1],
            decay=self.decay[i:i + 1], Z=self.Z[i:i + 1],
            n_pages=kf, shape_m=self.shape_m)


def compute_modes(A: np.ndarray, B: np.ndarray, cfg: GAJConfig) -> SlowModes:
    if cfg.eig == "exact":
        return exact_slow_modes(A, cfg.n_modes)
    return power_iteration_slow_modes(
        A, B, n_modes=cfg.n_modes, pw=cfg.pw, nit=cfg.nit,
        tau_g=cfg.tau_g, stability_check=cfg.stability_check)


def forward(model: LinearDriveObserveModel, X, cfg: GAJConfig | None = None) -> ForwardState:
    """Surrogate response for one parameter vector or a stack of them."""
    cfg = cfg or GAJConfig()
    X = np.atleast_2d(np.asarray(X, float))
    nx, p = X.shape
    system = model.build_many(X)
    A = system.A
    kf, m = A.shape[1], A.shape[2]

    Af = A.reshape(nx * kf, m, m)
    cf = system.c.reshape(nx * kf, m)
    M0f = system.M0.reshape(nx * kf, m)
    B = np.linalg.inv(Af)
    Mss = -np.einsum("kij,kj->ki", B, cf)
    e0 = M0f - Mss

    modes = compute_modes(Af, B, cfg)
    alpha = np.einsum("kip,ki->kp", modes.W, e0)          # w_s^T e0
    beta = np.einsum("i,kip->kp", system.o, modes.V)      # o^T v_s
    decay = np.exp(modes.lam * model.pages.ts)

    Z = Mss[:, model.water_z] + np.einsum("kp,kp,kp->k", alpha, beta, decay)
    if np.iscomplexobj(Z):
        Z = Z.real

    return ForwardState(
        x=X, pages=model.pages,
        B=B.reshape(nx, kf, m, m), Mss=Mss.reshape(nx, kf, m),
        e0=e0.reshape(nx, kf, m), modes=modes,
        alpha=alpha.reshape(nx, kf, -1), beta=beta.reshape(nx, kf, -1),
        decay=decay.reshape(nx, kf, -1), Z=Z.reshape(nx, kf),
        n_pages=kf, shape_m=m)


def surrogate_response(model, X, cfg: GAJConfig | None = None) -> np.ndarray:
    """Convenience wrapper: (n_x, p) or (p,) -> response of shape (n_x, kf) or (kf,)."""
    X = np.asarray(X, float)
    single = X.ndim == 1
    state = forward(model, np.atleast_2d(X), cfg)
    return state.Z[0] if single else state.Z


# --------------------------------------------------------------------------- #
# reference forward models
# --------------------------------------------------------------------------- #

def exact_response(model, X) -> np.ndarray:
    """Exact response Eq. (2) by a scaled-and-squared matrix exponential.

    Batched over pages (and over parameter vectors), which is the same algorithm
    as SciPy's ``expm`` but one to two orders of magnitude faster on the stiff
    Bloch-McConnell generators; it agrees with ``scipy.linalg.expm`` to ~1e-11.
    """
    from .linalg import expm_batch

    X = np.atleast_2d(np.asarray(X, float))
    nx = X.shape[0]
    system = model.build_many(X)
    A = system.A                                   # (nx, kf, m, m)
    c = system.c
    M0 = system.M0
    B = np.linalg.inv(A)
    Mss = -np.einsum("...ij,...j->...i", B, c)
    e0 = M0 - Mss
    E = expm_batch(A, model.pages.ts)              # (nx, kf, m, m)
    M = Mss + np.einsum("...ij,...j->...i", E, e0)
    return np.einsum("i,...i->...", system.o, M)


def truncation_error(model, truth, X, cfg: GAJConfig | None = None) -> np.ndarray:
    """|Z_S - Z_exact| per page, for the vectors in ``X``."""
    zs = surrogate_response(model, X, cfg)
    ze = exact_response(model, X)
    return np.abs(zs - ze)


def spectral_error(model, x: np.ndarray, truth: np.ndarray) -> float:
    """Eq. (3): RMS over pages of the difference of the exact responses."""
    d = exact_response(model, np.atleast_2d(x))[0] - exact_response(model, np.atleast_2d(truth))[0]
    return float(np.sqrt(np.mean(d ** 2)))
