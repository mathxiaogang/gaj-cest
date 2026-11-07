"""Gated analytic Jacobian (Section 4.3-4.5 of the paper).

The Jacobian of the surrogate is assembled from three groups of terms,

    J = J0 + sum_{q>=1} T_q,
    J0 = T_ss + T_e0 + T_SS            (Eq. 8, 12, 13)
    T_q = sum_{s in S} -lam_s^{q-1} e^{lam_s Ts}
          [ beta_s w_s^T dA_i u_q - alpha_s c_q^T dA_i v_s ]      (Eq. 16)

with the reduced-resolvent series for the coupling to the discarded modes,
``u_q = B^q Q_F e0`` and ``c_q = o^T B^q Q_F``.  A run-time gate accepts the
partial sum when its last terms are small, otherwise the Jacobian falls back to
batched central differences of the surrogate (Eq. 19).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import expm

from .model import LinearDriveObserveModel
from .surrogate import ForwardState, GAJConfig, forward, surrogate_response


@dataclass
class JacobianInfo:
    accepted_order: int = 0        # 0 -> fallback
    used_fallback: bool = False
    eta: float = np.inf            # gate statistic of the accepted order
    term_norms: list = None        # ||T_q|| for q = 1..qmax
    n_modes: int = 0

    def __post_init__(self):
        if self.term_norms is None:
            self.term_norms = []


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _bilinear(u: np.ndarray, v: np.ndarray, rows, cols, vals) -> np.ndarray:
    """u^T M v for the sparse matrix M given by (rows, cols, vals).

    ``u`` and ``v`` are (n_pages, m); the result is (n_pages,).
    """
    out = np.zeros(u.shape[0])
    for r, c, w in zip(rows, cols, vals):
        out += u[:, r] * w * v[:, c]
    return out


def _divided_differences(lam: np.ndarray, ts: float) -> np.ndarray:
    """Phi_rc = (e^{lam_c Ts} - e^{lam_r Ts}) / (lam_c - lam_r), stable form."""
    kf, p = lam.shape
    er = np.exp(lam * ts)
    a = lam[:, None, :] * ts          # a[k, r, c] = lam_c Ts
    b = lam[:, :, None] * ts          # b[k, r, c] = lam_r Ts
    diff = a - b
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = ts * er[:, :, None] * np.expm1(diff) / diff
    same = np.abs(diff) < 1e-3
    phi = np.where(same, ts * er[:, :, None], phi)
    # exact equality (same mode) must be the analytic limit
    eye = np.eye(p, dtype=bool)[None, :, :]
    phi = np.where(eye, ts * er[:, :, None], phi)
    return phi


def _base_terms(model, x: np.ndarray, state: ForwardState, cfg: GAJConfig):
    """J0 = T_ss + T_e0 + T_SS, plus the quantities the series needs."""
    st = state.take(0)
    B = st.B[0]
    Mss = st.Mss[0]
    e0 = st.e0[0]
    alpha = st.alpha[0]
    beta = st.beta[0]
    decay = st.decay[0]
    lam = st.modes.lam
    V = st.modes.V
    Wm = st.modes.W
    kept = st.modes.kept
    ts = st.pages.ts
    zw = model.water_z
    m = st.shape_m
    kf = st.n_pages
    p = x.size

    derivs = model.derivatives(x)
    phi = _divided_differences(lam, ts)

    J0 = np.zeros((kf, p))
    for i in range(p):
        rows, cols, vals = derivs.dA_sparse[i]
        # steady state:  dMss = -B (dA Mss + dc)
        rhs = -(np.einsum("ij,kj->ki", derivs.dA[i], Mss) + derivs.dc[i][None, :])
        dMss = np.einsum("kij,kj->ki", B, rhs)
        J0[:, i] = dMss[:, zw]
        # initial-state term
        de0 = derivs.dM0[i][None, :] - dMss
        J0[:, i] += np.einsum("kp,kp,kp->k", beta, decay,
                              np.einsum("kip,ki->kp", Wm, de0))
        # retained-retained block
        G = np.empty((kf, lam.shape[1], lam.shape[1]))
        for r in range(lam.shape[1]):
            for c in range(lam.shape[1]):
                G[:, r, c] = _bilinear(Wm[:, :, r], V[:, :, c], rows, cols, vals)
        J0[:, i] += np.einsum("kr,krc,krc,kc->k", beta, G, phi, alpha)

    return J0, phi, derivs, st


def _series_terms(model, x: np.ndarray, cfg: GAJConfig, derivs, st: ForwardState):
    """Yield T_q for q = 1..qmax, together with the u_q / c_q recursions."""
    B = st.B[0]
    e0 = st.e0[0]
    alpha = st.alpha[0]
    beta = st.beta[0]
    decay = st.decay[0]
    lam = st.modes.lam
    V = st.modes.V
    Wm = st.modes.W
    kf, m = e0.shape
    p = x.size
    qmax = cfg.qmax

    o = np.zeros(m)
    o[model.water_z] = 1.0

    U = np.empty((qmax + 1, kf, m))
    C = np.empty((qmax + 1, kf, m))
    # u_0 = Q_F e0 and c_0 = o^T Q_F: start in the discarded subspace, so the
    # recursion never carries an O(1) slow component that would have to be
    # subtracted off at every order (that subtraction is what limits the paper's
    # recursion to ~5 useful terms in double precision).  Mathematically
    # identical to u_q = B^q Q_F e0, see the note in docs/method.md.
    U[0] = e0 - np.einsum("kp,kip->ki", alpha, V)
    C[0] = np.broadcast_to(o, (kf, m)).copy() - np.einsum("kp,kip->ki", beta, Wm)

    for q in range(1, qmax + 1):
        U[q] = np.einsum("kij,kj->ki", B, U[q - 1])
        # remove the slow leakage that rounding injects (analytic value: zero)
        U[q] -= np.einsum("kp,kip->ki", np.einsum("kip,ki->kp", Wm, U[q]), V)
        C[q] = np.einsum("kji,kj->ki", B, C[q - 1])
        C[q] -= np.einsum("kp,kip->ki", np.einsum("kip,ki->kp", V, C[q]), Wm)

        coef = -(lam ** (q - 1)) * decay                       # (kf, P)
        T = np.zeros((kf, p))
        for i in range(p):
            rows, cols, vals = derivs.dA_sparse[i]
            for s in range(lam.shape[1]):
                wA = _bilinear(Wm[:, :, s], U[q], rows, cols, vals)
                cA = _bilinear(C[q], V[:, :, s], rows, cols, vals)
                T[:, i] += coef[:, s] * (beta[:, s] * wA + alpha[:, s] * cA)
        yield q, T


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def jacobian_gaj(model: LinearDriveObserveModel, x: np.ndarray,
                 cfg: Optional[GAJConfig] = None,
                 state: Optional[ForwardState] = None):
    """Gated analytic Jacobian of the surrogate, shape (n_pages, p)."""
    cfg = cfg or GAJConfig()
    x = np.asarray(x, float)
    state = state if state is not None else forward(model, x[None, :], cfg)
    J0, phi, derivs, st = _base_terms(model, x, state, cfg)

    info = JacobianInfo(n_modes=int(st.modes.kept[0].sum()))
    J = J0.copy()
    prev_norm = None
    for q, T in _series_terms(model, x, cfg, derivs, st):
        tnorm = float(np.abs(T).max())
        info.term_norms.append(tnorm)
        J = J + T
        if q >= 2 and prev_norm is not None:
            eta = max(tnorm, prev_norm) / max(float(np.abs(J).max()), np.finfo(float).eps)
            if tnorm <= prev_norm and eta <= cfg.tau:
                info.accepted_order = q
                info.eta = eta
                return J, info
        prev_norm = tnorm

    # no order qualified -> batched central differences of the surrogate
    J = jacobian_central_surrogate(model, x, cfg)
    info.used_fallback = True
    info.accepted_order = 0
    return J, info


def jacobian_central_surrogate(model, x: np.ndarray, cfg: GAJConfig) -> np.ndarray:
    """Eq. (19): central differences of the surrogate, all perturbations batched."""
    x = np.asarray(x, float)
    p = x.size
    h = cfg.sh * np.maximum(np.abs(x), 1e-12)
    X = np.repeat(x[None, :], 2 * p, axis=0)
    for i in range(p):
        X[2 * i, i] += h[i]
        X[2 * i + 1, i] -= h[i]
    Z = surrogate_response(model, X, cfg)                     # (2p, kf)
    return ((Z[0::2] - Z[1::2]) / (2.0 * h)[:, None]).T


def jacobian_surrogate_direct(model: LinearDriveObserveModel, x: np.ndarray,
                              cfg: Optional[GAJConfig] = None,
                              state: Optional[ForwardState] = None) -> np.ndarray:
    """Exact Jacobian of the surrogate with the resolvent computed directly.

    Same object the series converges to (Proposition 1) but without truncation;
    used to validate the series.  The reduced resolvent is evaluated on the full
    LAPACK eigenbasis of ``A``, restricted to the discarded subspace,

        chi_s = sum_{j in F} alpha_j v_j / (lam_s - lam_j),
        psi_s = sum_{j in F} beta_j  w_j / (lam_s - lam_j),

    which avoids the (numerically singular) direct solve of ``lam_s I - A``.
    """
    cfg = cfg or GAJConfig()
    x = np.asarray(x, float)
    state = state if state is not None else forward(model, x[None, :], cfg)
    J0, phi, derivs, st = _base_terms(model, x, state, cfg)

    B = st.B[0]
    A = np.linalg.inv(B)
    e0 = st.e0[0]
    alpha = st.alpha[0]
    beta = st.beta[0]
    decay = st.decay[0]
    lam = st.modes.lam
    V = st.modes.V
    Wm = st.modes.W
    kf, m = e0.shape
    p = x.size
    o = np.zeros(m)
    o[model.water_z] = 1.0

    J = J0.copy()
    for k in range(kf):
        vals, vecs = np.linalg.eig(A[k])
        left = np.linalg.inv(vecs)                        # rows: w_j^T
        # the P eigenvalues closest to the retained ones belong to S
        slow_idx = set()
        for s in range(st.modes.n_modes):
            if not st.modes.kept[k, s]:
                continue
            slow_idx.add(int(np.argmin(np.abs(vals - lam[k, s]))))
        fast = np.array([j for j in range(m) if j not in slow_idx])

        a_full = left @ e0[k]
        b_full = vecs.conj().T @ o                       # o^T v_j
        chi = np.zeros((st.modes.n_modes, m), dtype=complex)
        psi = np.zeros((st.modes.n_modes, m), dtype=complex)
        for s in range(st.modes.n_modes):
            if not st.modes.kept[k, s]:
                continue
            denom = lam[k, s] - vals[fast]
            chi[s] = vecs[:, fast] @ (a_full[fast] / denom)
            # psi_s = sum_j beta_j w_j / (lam_s - lam_j): left eigenvectors
            psi[s] = left[fast, :].conj().T @ (b_full[fast] / denom)
        for i in range(p):
            rows, cols, vals_i = derivs.dA_sparse[i]
            for s in range(st.modes.n_modes):
                if not st.modes.kept[k, s]:
                    continue
                wA = np.sum(Wm[k, rows, s] * vals_i * chi[s][cols])
                cA = np.sum(psi[s][rows] * vals_i * V[k, cols, s])
                J[k, i] += decay[k, s] * (beta[k, s] * wA.real + alpha[k, s] * cA.real)
    return J


def jacobian_central_exact(model: LinearDriveObserveModel, x: np.ndarray,
                           sh: float = 1e-4) -> np.ndarray:
    """Central differences of the *exact* response (the Trad-C control)."""
    from .surrogate import exact_response

    x = np.asarray(x, float)
    p = x.size
    h = sh * np.maximum(np.abs(x), 1e-12)
    X = np.repeat(x[None, :], 2 * p, axis=0)
    for i in range(p):
        X[2 * i, i] += h[i]
        X[2 * i + 1, i] -= h[i]
    Z = exact_response(model, X)
    return ((Z[0::2] - Z[1::2]) / (2.0 * h)[:, None]).T


def jacobian_vanloan(model: LinearDriveObserveModel, x: np.ndarray,
                     pages: Optional[np.ndarray] = None) -> np.ndarray:
    """Exact derivative of the exact response by Van Loan's block formula.

    Used as the reference Jacobian in the tests and for the Exact-AJ control.
    """
    x = np.asarray(x, float)
    system = model.build(x)
    A = system.A
    c = system.c
    M0 = system.M0
    ts = model.pages.ts
    m = model.m
    p = x.size
    zw = model.water_z
    derivs = model.derivatives(x)
    idx = list(range(A.shape[0])) if pages is None else list(pages)
    J = np.zeros((len(idx), p))
    for row, k in enumerate(idx):
        Bk = np.linalg.inv(A[k])
        Mss = -Bk @ c[k]
        e0 = M0[k] - Mss
        expA = expm(A[k] * ts)
        for i in range(p):
            dA = derivs.dA[i]
            rhs = -(dA @ Mss + derivs.dc[i])
            dMss = Bk @ rhs              # dMss = -A^{-1} (dA Mss + dc)
            block = np.zeros((2 * m, 2 * m))
            block[:m, :m] = A[k]
            block[m:, m:] = A[k]
            block[:m, m:] = dA
            dExp = expm(block * ts)[:m, m:]
            de0 = derivs.dM0[i] - dMss
            dM = dMss + dExp @ e0 + expA @ de0
            J[row, i] = dM[zw]
    return J
