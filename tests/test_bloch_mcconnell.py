"""The Bloch-McConnell instance: dimensions, physics and closed-form derivatives."""

from __future__ import annotations

import numpy as np

from gaj import FivePoolBM, PageSet


def test_dimensions(model, truth):
    assert model.m == 18 and model.p == 17
    assert model.pages.n_pages == 147
    assert truth.size == model.p
    lo, hi = model.bounds()
    assert np.all(truth > lo) and np.all(truth < hi)


def test_monitored_quantities(model, truth):
    q = model.monitored(truth)
    assert q.size == 22                     # 17 parameters + 5 products
    n = model.n_pools
    ksw, m0s = truth[2 + n:2 + 2 * n], truth[2 + 2 * n:]
    assert np.allclose(q[17:], ksw * m0s)


def test_equilibrium_without_rf(model, truth):
    """With w1 = 0 the thermal equilibrium M0 must be the steady state."""
    static = FivePoolBM(PageSet(b1_uT=[0.0], offsets_ppm=[0.0], ts=1.0))
    system = static.build(truth)
    residual = system.A[0] @ system.M0[0] + system.c[0]
    assert np.abs(residual).max() < 1e-10


def test_steady_state_is_reached(model, truth):
    """Beyond ~20 T1 the exact response must coincide with the steady state."""
    long_ts = FivePoolBM(PageSet(b1_uT=[0.9], offsets_ppm=model.pages.offsets_ppm, ts=200.0))
    system = long_ts.build(truth)
    B = np.linalg.inv(system.A)
    mss = -np.einsum("kij,kj->ki", B, system.c)
    from gaj import exact_response

    z = exact_response(long_ts, truth)[0]
    assert np.abs(z - mss[:, long_ts.water_z]).max() < 1e-10


def test_derivatives_match_finite_differences(model, truth):
    """dA, dc and dM0 against central differences of the assembled system."""
    derivs = model.derivatives(truth)
    for i in [0, 1, 2, 6, 7, 11, 12, 16]:
        h = 1e-6 * abs(truth[i])
        xp, xm = truth.copy(), truth.copy()
        xp[i] += h
        xm[i] -= h
        sp, sm = model.build(xp), model.build(xm)
        dA_fd = (sp.A - sm.A) / (2 * h)
        dc_fd = (sp.c - sm.c) / (2 * h)
        dM0_fd = (sp.M0 - sm.M0) / (2 * h)
        assert np.abs(dA_fd - derivs.dA[i]).max() < 1e-6 * max(1.0, np.abs(derivs.dA[i]).max())
        assert np.abs(dc_fd - derivs.dc[i]).max() < 1e-6 * max(1.0, np.abs(derivs.dc[i]).max())
        assert np.abs(dM0_fd - derivs.dM0[i]).max() < 1e-6


def test_derivatives_are_page_independent(model, truth):
    """M3 in its strongest form: dA, dc and dM0 do not depend on Delta or B1."""
    derivs = model.derivatives(truth)
    one = FivePoolBM(PageSet(b1_uT=[0.5], offsets_ppm=[-6.0], ts=2.0))
    other = FivePoolBM(PageSet(b1_uT=[1.6], offsets_ppm=[3.5], ts=2.0))
    for i in [0, 3, 8, 13]:
        h = 1e-6 * abs(truth[i])
        xp, xm = truth.copy(), truth.copy()
        xp[i] += h
        xm[i] -= h
        d1 = (one.build(xp).A - one.build(xm).A) / (2 * h)
        d2 = (other.build(xp).A - other.build(xm).A) / (2 * h)
        assert np.abs(d1 - d2).max() < 1e-9
        assert np.abs(d1 - derivs.dA[i]).max() < 1e-6 * max(1.0, np.abs(derivs.dA[i]).max())


def test_zspectrum_shape(model, truth):
    """A CEST Z-spectrum: saturated at 0 ppm, near 1 far away, dips at the pools."""
    from gaj import surrogate_response

    z = surrogate_response(model, truth)
    b1, off = model.pages.broadcast()
    block = z[b1 == 0.5]
    assert block[np.argmin(np.abs(off[b1 == 0.5]))] < 0.05        # direct saturation
    assert 0.3 < block[np.argmin(np.abs(off[b1 == 0.5] - 3.5))] < 1.0  # amide CEST dip
    assert np.all(z > -1e-6) and np.all(z <= 1.0 + 1e-9)
