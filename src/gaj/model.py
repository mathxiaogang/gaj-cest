"""Model abstractions for linear drive-and-observe systems.

A *page* is one combination of the drive conditions (saturation amplitude,
frequency offset).  On every page the state obeys

    dM/dt = A(x, xi) M + c(x),     M(0) = M0(x),     Z = o^T M(Ts)

with constant ``A`` and ``c`` over the drive phase of duration ``Ts``
(assumption M1), a linear observation ``o`` (M2) and parameter derivatives of
``A``, ``c`` and ``M0`` that are cheap and page independent (M3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class PageSet:
    """The drive conditions: ``K`` amplitudes x ``F`` offsets, all at one ``Ts``."""

    b1_uT: np.ndarray
    offsets_ppm: np.ndarray
    ts: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "b1_uT", np.atleast_1d(np.asarray(self.b1_uT, float)))
        object.__setattr__(self, "offsets_ppm", np.atleast_1d(np.asarray(self.offsets_ppm, float)))
        if self.ts <= 0:
            raise ValueError("saturation time must be positive")

    @property
    def n_amplitudes(self) -> int:
        return self.b1_uT.size

    @property
    def n_offsets(self) -> int:
        return self.offsets_ppm.size

    @property
    def n_pages(self) -> int:
        return self.n_amplitudes * self.n_offsets

    def broadcast(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-page amplitudes and offsets, amplitude-major (K blocks of F)."""
        b1 = np.repeat(self.b1_uT, self.n_offsets)
        off = np.tile(self.offsets_ppm, self.n_amplitudes)
        return b1, off


@dataclass
class System:
    """The matrices of one parameter vector, stacked over pages."""

    A: np.ndarray            # (n_pages, m, m)
    c: np.ndarray            # (n_pages, m)
    M0: np.ndarray           # (n_pages, m)
    o: np.ndarray            # (m,)  observation vector

    @property
    def n_pages(self) -> int:
        return self.A.shape[0]

    @property
    def m(self) -> int:
        return self.A.shape[-1]


@dataclass
class Derivatives:
    """Parameter derivatives; page independent by M3 (strong form)."""

    dA: np.ndarray           # (p, m, m)
    dc: np.ndarray           # (p, m)
    dM0: np.ndarray          # (p, m)
    dA_sparse: list = field(default_factory=list)   # optional (rows, cols, vals)


class LinearDriveObserveModel(Protocol):
    """What the GAJ machinery needs from a model."""

    m: int
    p: int
    pages: PageSet

    def build(self, x: np.ndarray) -> System: ...

    def derivatives(self, x: np.ndarray) -> Derivatives: ...

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]: ...

    def monitored(self, x: np.ndarray) -> np.ndarray: ...

    def parameter_names(self) -> list[str]: ...
