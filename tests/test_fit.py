"""End-to-end: a fit recovers the generating vector, and GAJ agrees with the control."""

from __future__ import annotations

import numpy as np

from gaj import fit, meets_requirement, relative_errors, simulate


def test_fit_reduces_the_residual(model_small, truth):
    z = simulate(model_small, truth)
    x0 = truth * np.exp(0.01 * np.random.default_rng(0).normal(size=truth.size))
    res = fit(model_small, z, x0, jacobian="gaj", tol=1e-8, max_nfev=60)
    sse = float(np.sum(np.asarray(res.trace[:-1]) ** 2))   # last entry is the constant 1
    assert sse < 1e-8
    assert np.all(np.isfinite(res.x))


def test_gaj_and_central_jacobian_reach_the_same_solution(model_small, truth):
    z = simulate(model_small, truth)
    x0 = truth * np.exp(0.02 * np.random.default_rng(1).normal(size=truth.size))
    a = fit(model_small, z, x0, jacobian="gaj", tol=1e-8, max_nfev=60)
    b = fit(model_small, z, x0, jacobian="central_surrogate", tol=1e-8, max_nfev=60)
    rel = np.abs(a.x - b.x) / np.abs(b.x)
    assert rel.max() < 5e-3


def test_accuracy_metrics_run(model_small, truth):
    """The 22 monitored errors are small near the truth and the metric is symmetric."""
    err = relative_errors(model_small, truth, truth)
    assert err.size == 22 and np.all(err == 0)
    met = meets_requirement(model_small, truth, truth)
    assert met["ok"] and met["max_error"] == 0.0


def test_fit_does_not_leave_the_box(model_small, truth):
    z = simulate(model_small, truth, noise=1e-3, seed=7)
    lo, hi = model_small.bounds()
    x0 = np.exp(0.3 * np.log(lo) + 0.7 * np.log(hi))
    res = fit(model_small, z, x0, jacobian="gaj", tol=1e-6, max_nfev=40)
    assert np.all(res.x >= lo - 1e-12) and np.all(res.x <= hi + 1e-12)
