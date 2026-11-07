"""Sections 4.3-4.4: the analytic Jacobian against finite differences and the
direct reduced resolvent, and the exact derivative of the exact response."""

from __future__ import annotations

import numpy as np

from gaj import (GAJConfig, FivePoolBM, PageSet, exact_response, jacobian_central_exact,
                 jacobian_gaj, jacobian_surrogate_direct, jacobian_vanloan,
                 surrogate_response)


def _directional_error(model, x, cfg, n_trials=3, s=1e-5, seed=0):
    """Compare J with a central difference along random log-space directions.

    A directional derivative avoids the cancellation that per-parameter finite
    differences suffer on the 1.5e-5 s relaxation time.
    """
    J, _ = jacobian_gaj(model, x, cfg)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n_trials):
        d = rng.normal(size=x.size)
        d /= np.linalg.norm(d)
        zp = surrogate_response(model, x * np.exp(s * d), cfg)[0]
        zm = surrogate_response(model, x * np.exp(-s * d), cfg)[0]
        fd = (zp - zm) / (2 * s)
        an = float((J[0] * x * d).sum())
        worst = max(worst, abs(an - fd) / max(abs(fd), 1e-12))
    return worst


def test_jacobian_matches_directional_finite_differences(model_small, truth):
    for eig in ("power", "exact"):
        err = _directional_error(model_small, truth, GAJConfig(eig=eig))
        assert err < 1e-4, f"{eig}: relative directional error {err:.2e}"


def test_series_matches_direct_resolvent(model_small, truth):
    """Both routes must agree when the retained pairs are the exact ones.

    With the power-iteration pairs the two differ at ~1e-4 relative, because the
    direct route resolves the discarded subspace in the LAPACK basis while Q_F is
    built from the power-iteration vectors; the finite-difference test above is
    the reference for that configuration.
    """
    cfg = GAJConfig(eig="exact")
    J, info = jacobian_gaj(model_small, truth, cfg)
    Jd = jacobian_surrogate_direct(model_small, truth, cfg)
    assert not info.used_fallback
    assert np.abs(J - Jd).max() / np.abs(Jd).max() < 1e-6


def test_vanloan_matches_central_differences_of_exact_response(model_small, truth):
    pages = [0, 3, 7]
    Jv = jacobian_vanloan(model_small, truth, pages=pages)
    Jc = jacobian_central_exact(model_small, truth)[pages]
    assert np.abs(Jv - Jc).max() / np.abs(Jc).max() < 1e-5


def test_surrogate_jacobian_is_close_to_exact_response_jacobian(model_small, truth):
    """Proposition 1: JS - J is the (small) derivative of the discarded part."""
    pages = [0, 3, 7]
    Jv = jacobian_vanloan(model_small, truth, pages=pages)
    Jd = jacobian_surrogate_direct(model_small, truth)[pages]
    rel = np.abs(Jv - Jd).max() / np.abs(Jv).max()
    assert rel < 1e-6


def test_jacobian_is_finite_everywhere(model_small, truth):
    lo, hi = model_small.bounds()
    rng = np.random.default_rng(3)
    for _ in range(3):
        x = np.exp(rng.uniform(np.log(lo), np.log(hi)))
        J, info = jacobian_gaj(model_small, x, GAJConfig())
        assert np.all(np.isfinite(J))
        assert J.shape == (model_small.pages.n_pages, model_small.p)
