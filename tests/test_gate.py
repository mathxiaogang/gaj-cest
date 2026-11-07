"""Section 4.5: the gate accepts a converged series and falls back otherwise."""

from __future__ import annotations

import numpy as np

from gaj import GAJConfig, jacobian_central_surrogate, jacobian_gaj


def test_series_terms_decay_and_are_accepted(model_small, truth):
    J, info = jacobian_gaj(model_small, truth, GAJConfig())
    assert not info.used_fallback
    assert 2 <= info.accepted_order <= 12
    assert info.eta <= 1e-8
    terms = np.array(info.term_norms)
    assert terms[0] > terms[1] > terms[2]           # decreasing before acceptance


def test_stricter_gate_asks_for_more_terms_or_falls_back(model_small, truth):
    loose, info_loose = jacobian_gaj(model_small, truth, GAJConfig(tau=1e-4))
    strict, info_strict = jacobian_gaj(model_small, truth, GAJConfig(tau=1e-14))
    assert info_loose.accepted_order <= info_strict.accepted_order or info_strict.used_fallback


def test_rejected_series_returns_the_central_difference_fallback(model_small, truth):
    cfg = GAJConfig(tau=1e-30)                       # unreachable in double precision
    J, info = jacobian_gaj(model_small, truth, cfg)
    assert info.used_fallback
    assert info.accepted_order == 0
    Jc = jacobian_central_surrogate(model_small, truth, cfg)
    assert np.allclose(J, Jc)


def test_qmax_one_forces_the_fallback(model_small, truth):
    J, info = jacobian_gaj(model_small, truth, GAJConfig(qmax=1))
    assert info.used_fallback


def test_fallback_and_series_agree_when_both_are_accurate(model_small, truth):
    """The fallback differentiates the same surrogate, so both routes coincide."""
    _, info = jacobian_gaj(model_small, truth, GAJConfig())
    J_series, _ = jacobian_gaj(model_small, truth, GAJConfig())
    J_fb, _ = jacobian_gaj(model_small, truth, GAJConfig(qmax=1))
    rel = np.abs(J_series - J_fb).max() / np.abs(J_series).max()
    assert rel < 1e-4
