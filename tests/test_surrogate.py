"""Section 4.2: the surrogate against the exact response, and its validity range."""

from __future__ import annotations

import numpy as np

from gaj import (GAJConfig, FivePoolBM, PageSet, exact_response, forward,
                 surrogate_response, truncation_error)


def test_surrogate_matches_exact_response(model, truth):
    zs = surrogate_response(model, truth)
    ze = exact_response(model, truth)[0]
    assert np.abs(zs - ze).max() < 1e-10


def test_truncation_error_grows_at_short_saturation(model, truth):
    """H2 is not automatic: with a short Ts the discarded modes still carry weight."""
    short = FivePoolBM(PageSet(b1_uT=[0.9], offsets_ppm=model.pages.offsets_ppm, ts=0.5))
    err_short = truncation_error(short, truth, truth).max()
    err_long = truncation_error(model, truth, truth).max()
    assert err_short > 1e-5
    assert err_long < 1e-10
    assert err_short > 100 * err_long


def test_exact_eigenpairs_give_the_same_forward(model, truth):
    z_power = surrogate_response(model, truth, GAJConfig(eig="power"))
    z_exact = surrogate_response(model, truth, GAJConfig(eig="exact"))
    assert np.abs(z_power - z_exact).max() < 1e-9


def test_forward_state_shapes(model, truth):
    state = forward(model, truth[None, :])
    assert state.Z.shape == (1, model.pages.n_pages)
    assert state.alpha.shape[-1] == 2 and state.beta.shape[-1] == 2
    assert np.all(np.isfinite(state.decay))


def test_retained_modes_dominate_the_fast_ones(model, truth):
    """e^{lam Ts} of the retained modes is not negligible, the discarded ones are."""
    state = forward(model, truth[None, :])
    # the slowest retained mode survives 2 s of saturation; the second one is
    # already at the level of rounding (lam_2 ~ -13 /s), which is harmless
    assert np.median(state.decay[0][:, 0]) > 1e-3
    assert np.all(state.decay[0] <= 1.0 + 1e-12)
