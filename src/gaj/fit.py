"""Least-squares fitting with GAJ or with reference Jacobians.

All configurations use the same optimiser (SciPy's trust-region reflective
``least_squares``), the same bounds and the same starting points.  The
function-tolerance test of MATLAB's ``lsqnonlin``,
``|SSE_k-1 - SSE_k| < t (1 + SSE_k)``, is emulated by appending a constant
residual of one, as in the paper's reimplementation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import least_squares

from .jacobian import (jacobian_central_exact, jacobian_central_surrogate,
                       jacobian_gaj, jacobian_vanloan)
from .model import LinearDriveObserveModel
from .surrogate import GAJConfig, exact_response, forward, surrogate_response

JACOBIANS = ("gaj", "central_surrogate", "central_exact", "vanloan", "forward_exact")


@dataclass
class FitResult:
    x: np.ndarray
    cost: float
    nfev: int
    njev: int
    success: bool
    message: str
    seconds: float
    tol: float
    jacobian: str
    accepted_orders: list = field(default_factory=list)
    fallbacks: int = 0
    n_iter: int = 0
    trace: list = field(default_factory=list)

    @property
    def fallback_fraction(self) -> float:
        return self.fallbacks / self.njev if self.njev else 0.0


def _pack(model, z: np.ndarray, cfg: GAJConfig):
    """Residual with the appended constant term (see module docstring)."""

    def residual(x):
        return np.append(surrogate_response(model, x, cfg) - z, 1.0)

    return residual


def fit(model: LinearDriveObserveModel,
        z: np.ndarray,
        x0: np.ndarray,
        jacobian: str = "gaj",
        tol: float = 1e-8,
        max_nfev: int = 250,
        cfg: Optional[GAJConfig] = None,
        verbose: bool = False) -> FitResult:
    """Fit one start.  ``jacobian`` selects the configuration of Table 3."""
    if jacobian not in JACOBIANS:
        raise ValueError(f"jacobian must be one of {JACOBIANS}")
    cfg = cfg or GAJConfig()
    x0 = np.asarray(x0, float)
    lo, hi = model.bounds()
    residual = _pack(model, z, cfg)

    orders: list[int] = []
    fallbacks = [0]
    njev = [0]

    def jac(x):
        njev[0] += 1
        if jacobian == "gaj":
            J, info = jacobian_gaj(model, x, cfg)
            orders.append(info.accepted_order)
            if info.used_fallback:
                fallbacks[0] += 1
        elif jacobian == "central_surrogate":
            J = jacobian_central_surrogate(model, x, cfg)
        elif jacobian == "central_exact":
            J = jacobian_central_exact(model, x, cfg.sh)
        elif jacobian == "vanloan":
            J = jacobian_vanloan(model, x)
        else:                       # forward differences of the exact model
            h = cfg.sh * np.maximum(np.abs(x), 1e-12)
            Z0 = exact_response(model, x)
            J = np.empty((Z0.size, x.size))
            for i in range(x.size):
                xp = x.copy()
                xp[i] += h[i]
                J[:, i] = (exact_response(model, xp) - Z0) / h[i]
        return np.vstack([J, np.zeros((1, x.size))])

    t0 = time.perf_counter()
    result = least_squares(
        residual, x0, jac=jac, bounds=(lo, hi), method="trf",
        ftol=tol, xtol=tol, gtol=1e-6, max_nfev=max_nfev, verbose=0)
    seconds = time.perf_counter() - t0

    return FitResult(
        x=result.x, cost=float(result.cost), nfev=int(result.nfev),
        njev=njev[0], success=bool(result.success), message=str(result.message),
        seconds=seconds, tol=tol, jacobian=jacobian,
        accepted_orders=orders, fallbacks=fallbacks[0],
        n_iter=int(result.get('nit', 0) or 0),
        trace=list(result.fun))


def fit_multi_start(model, z, starts, jacobian="gaj", tol=1e-8,
                    max_nfev=250, cfg=None, keep_all=False):
    """Run every start and keep the solution with the smallest sum of squares."""
    results = []
    for k, x0 in enumerate(starts):
        res = fit(model, z, x0, jacobian=jacobian, tol=tol,
                  max_nfev=max_nfev, cfg=cfg)
        results.append(res)
    best = min(results, key=lambda r: r.cost)
    return (best, results) if keep_all else best
