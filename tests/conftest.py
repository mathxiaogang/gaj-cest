"""Shared fixtures: a small but representative Bloch-McConnell configuration."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from gaj import FivePoolBM, PageSet, GAJConfig  # noqa: E402


@pytest.fixture(scope="session")
def pages_full():
    return PageSet(b1_uT=[0.5, 0.9, 1.6], offsets_ppm=np.arange(-6.0, 6.001, 0.25), ts=2.0)


@pytest.fixture(scope="session")
def pages_small():
    """Two amplitudes, 25 offsets (50 pages) -- quick but over-determined."""
    return PageSet(b1_uT=[0.9, 1.6], offsets_ppm=np.arange(-6.0, 6.001, 0.5), ts=2.0)


@pytest.fixture(scope="session")
def model(pages_full):
    return FivePoolBM(pages_full)


@pytest.fixture(scope="session")
def model_small(pages_small):
    return FivePoolBM(pages_small)


@pytest.fixture(scope="session")
def truth(model):
    return model.truth()


@pytest.fixture(scope="session")
def cfg():
    return GAJConfig()
