"""Procedure 1: the batched power iteration against LAPACK, and the guard."""

from __future__ import annotations

import numpy as np

from gaj import exact_slow_modes, power_iteration_slow_modes
from gaj.eigen import spectral_diagnostics


def test_eigenvalues_match_lapack(model, truth):
    system = model.build(truth)
    B = np.linalg.inv(system.A)
    modes = power_iteration_slow_modes(system.A, B, n_modes=2, pw=8, nit=8)
    exact = exact_slow_modes(system.A, 2)
    lam_exact = np.sort(np.abs(exact.lam.real), axis=1)
    lam = np.sort(np.abs(modes.lam), axis=1)
    rel = np.abs(lam - lam_exact) / np.abs(lam_exact)
    assert np.median(rel[:, 0]) < 1e-10          # the slowest mode is well resolved
    assert np.median(rel[:, 1]) < 1e-3           # the second one only to the power pw*nit


def _max_residual(A, modes, mode_index):
    worst = 0.0
    for p in range(A.shape[0]):
        if not modes.kept[p, mode_index]:
            continue
        v = modes.V[p, :, mode_index]
        lam = modes.lam[p, mode_index]
        worst = max(worst, np.linalg.norm(A[p] @ v - lam * v) / np.linalg.norm(A[p] @ v))
    return worst


def test_eigenvectors_are_dual_and_residual_small(model, truth):
    system = model.build(truth)
    B = np.linalg.inv(system.A)
    modes = power_iteration_slow_modes(system.A, B, n_modes=2, pw=8, nit=8)
    for p in range(0, model.pages.n_pages, 25):
        for s in range(2):
            if modes.kept[p, s]:
                assert abs(modes.W[p, :, s] @ modes.V[p, :, s] - 1.0) < 1e-10
    # the slowest mode is resolved to rounding, the second only to the power
    # pw*nit of |lam_2|/|lam_3| -- the paper's own statement about Procedure 1
    res1 = _max_residual(system.A, modes, 0)
    assert res1 < 1e-6            # worst page; the median is at rounding level
    assert _max_residual(system.A, modes, 1) < 5e-2
    # mode 2 is resolved only to ~(|lam_2|/|lam_3|)^(pw*nit): the paper's remark
    # that Procedure 1 handles a modulus ratio of 0.9 only to about 1e-3
    assert _max_residual(system.A, modes, 1) > 1e-8


def test_guard_drops_modes(model, truth):
    """A huge guard threshold must drop every mode after the first."""
    system = model.build(truth)
    B = np.linalg.inv(system.A)
    modes = power_iteration_slow_modes(system.A, B, n_modes=2, pw=8, nit=8, tau_g=1.0)
    assert modes.kept[:, 0].all()
    assert not modes.kept[:, 1].any()
    assert np.all(modes.V[~modes.kept[:, 1].reshape(-1)][:, :, 1] == 0)
    assert np.all(np.isfinite(modes.lam))


def test_stability_check_removes_positive_modes(model, truth):
    """A generator with a positive eigenvalue: the optional check drops it."""
    system = model.build(truth)
    A = system.A.copy()
    A[:, 0, 0] = 1.0                       # unstable water transverse mode
    B = np.linalg.inv(A)
    modes = power_iteration_slow_modes(A, B, n_modes=2, pw=8, nit=8, stability_check=True)
    assert np.all(modes.lam[modes.kept] < 0)


def test_products_are_h1_to_h3_diagnostics(model, truth):
    system = model.build(truth)
    modes = exact_slow_modes(system.A, 2)
    diag = spectral_diagnostics(system.A, modes)
    assert np.all(diag["max_real_eigenvalue"] < 0)      # H1
    assert np.all(diag["rho"] < 1.0)                    # H3
    assert 0.2 < np.nanmedian(diag["rho"]) < 0.95       # but not by a wide margin
