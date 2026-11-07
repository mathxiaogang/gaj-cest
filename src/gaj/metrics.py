"""Accuracy metrics of Section 3.2: the monitored relative errors and e_spec."""

from __future__ import annotations

import numpy as np

from .model import LinearDriveObserveModel
from .surrogate import exact_response


def relative_errors(model: LinearDriveObserveModel, x: np.ndarray,
                    truth: np.ndarray) -> np.ndarray:
    """|q_j - q_j*| / |q_j*| for the monitored quantities of the model."""
    q = model.monitored(np.asarray(x, float))
    qs = model.monitored(np.asarray(truth, float))
    return np.abs(q - qs) / np.abs(qs)


def spectral_error(model: LinearDriveObserveModel, x: np.ndarray,
                   truth: np.ndarray) -> float:
    """Eq. (3): RMS over pages of the exact responses at ``x`` and at the truth."""
    d = (exact_response(model, x) - exact_response(model, truth))[0]
    return float(np.sqrt(np.mean(d ** 2)))


def meets_requirement(model, x, truth, eps: float = 0.05,
                      eps_spec: float = 1e-5) -> dict:
    """Apply the accuracy requirement of Section 5.1 to one fitted vector."""
    err = relative_errors(model, x, truth)
    spec = spectral_error(model, x, truth)
    names = model.parameter_names() + [f"ksw*M0s{i + 1}" for i in range(model.n_pools)]
    worst = int(np.argmax(err))
    return {
        "ok": bool(err.max() <= eps and spec <= eps_spec),
        "max_error": float(err.max()),
        "worst_quantity": names[worst],
        "spectral_error": spec,
        "errors": err,
        "names": names,
    }


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a pass rate (used for the pass-rate table)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
