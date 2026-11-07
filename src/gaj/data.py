"""Data: simulated Z-spectra, starting points, and measured-spectrum I/O.

The simulated path reproduces the paper's phantom.  The measured-data path is a
strict reader for a tidy CSV of Z-spectrum values, so that the same fitting code
runs on scanner data without changes -- see ``data/README.md`` for the format and
for public sources of measured CEST spectra.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .model import LinearDriveObserveModel, PageSet
from .surrogate import exact_response


# --------------------------------------------------------------------------- #
# simulated data
# --------------------------------------------------------------------------- #

def simulate(model: LinearDriveObserveModel,
             x_true: np.ndarray,
             noise: float = 0.0,
             b0_shift_ppm: float = 0.0,
             b1_scale: float = 1.0,
             rician: bool = False,
             seed: Optional[int] = None,
             exact: bool = True) -> np.ndarray:
    """Generate a Z-spectrum from ``x_true``.

    ``b0_shift_ppm`` and ``b1_scale`` model the two imperfections that dominate
    measured CEST data: a wrong water frequency (the spectrum slides along the
    offset axis) and a wrong saturation amplitude.  The data are generated with
    the *true* conditions and fitted with the nominal ones, which is what makes
    the demonstration non-trivial.  With ``rician=True`` the magnitude noise of
    an MR image is applied instead of Gaussian noise.
    """
    pages = model.pages
    if b0_shift_ppm or b1_scale != 1.0:
        pages_true = PageSet(b1_uT=pages.b1_uT * b1_scale,
                             offsets_ppm=pages.offsets_ppm + b0_shift_ppm,
                             ts=pages.ts)
        model_true = _with_pages(model, pages_true)
    else:
        model_true = model

    z = exact_response(model_true, x_true)[0] if exact else None
    if z is None:
        from .surrogate import surrogate_response
        z = surrogate_response(model_true, x_true)

    if noise > 0:
        rng = np.random.default_rng(seed)
        if rician:
            z = np.sqrt((z + rng.normal(0, noise, z.shape)) ** 2
                        + (rng.normal(0, noise, z.shape)) ** 2)
        else:
            z = z + rng.normal(0, noise, z.shape)
    return z


def _with_pages(model: LinearDriveObserveModel, pages: PageSet):
    """A shallow copy of the model on different drive conditions."""
    import copy

    clone = copy.copy(model)
    clone.pages = pages
    return clone


# --------------------------------------------------------------------------- #
# starting points (Section 5.2)
# --------------------------------------------------------------------------- #

def log_bounds(model, zero_fraction: float = 0.03):
    """Bounds used for log-uniform sampling; a zero lower bound is replaced.

    Table B.1: a lower bound of zero is replaced by 0.03 times the upper bound.
    """
    lo, hi = model.bounds()
    lo_log = np.where(lo <= 0.0, zero_fraction * hi, lo)
    return lo_log, hi


def make_starts(model, truth: np.ndarray, n_random: int = 13,
                perturbation: float = 0.05, seed: int = 5000,
                n_total: Optional[int] = None) -> np.ndarray:
    """The 15 starting points of the paper: perturbation, centre, log-uniform draws."""
    rng = np.random.default_rng(seed)
    lo, hi = model.bounds()
    lo_log, hi_log = log_bounds(model)
    truth = np.asarray(truth, float)

    starts = []
    # 1: a perturbation of the truth (log-normal, sd 0.05 in the logarithm)
    starts.append(np.clip(truth * np.exp(rng.normal(0.0, perturbation, truth.size)), lo, hi))
    # 2: the geometric centre of the box
    starts.append(np.sqrt(lo_log * hi_log))
    # 3..: log-uniform draws in the box
    n_random = int(n_random if n_total is None else max(n_total - 2, 0))
    for _ in range(n_random):
        u = rng.random(truth.size)
        starts.append(np.exp(np.log(lo_log) + u * (np.log(hi_log) - np.log(lo_log))))
    return np.clip(np.array(starts), lo, hi)


# --------------------------------------------------------------------------- #
# measured spectra
# --------------------------------------------------------------------------- #

@dataclass
class Spectrum:
    """A measured Z-spectrum on a (B1, offset) grid."""

    pages: PageSet
    z: np.ndarray
    sd: Optional[np.ndarray] = None
    meta: Optional[dict] = None


def load_zspectrum(path: str | Path) -> Spectrum:
    """Read a Z-spectrum from CSV.

    Tidy format (one row per page)::

        b1_uT,offset_ppm,z[,sd]
        0.5,-6.00,0.9812[,0.004]
        ...

    Wide format (one row per offset, one column per amplitude)::

        offset_ppm,0.5,0.9,1.6
        -6.00,0.9812,0.9703,0.9520
        ...
    """
    path = Path(path)
    rows = [r for r in csv.reader(path.open(newline="", encoding="utf-8"))
            if r and not r[0].lstrip().startswith("#")]
    if not rows:
        raise ValueError(f"{path}: empty file")
    header = [h.strip() for h in rows[0]]
    lower = [h.lower() for h in header]

    if "z" in lower:
        # tidy
        i_b1 = lower.index("b1_ut") if "b1_ut" in lower else lower.index("b1")
        i_off = lower.index("offset_ppm") if "offset_ppm" in lower else lower.index("offset")
        i_z = lower.index("z")
        i_sd = lower.index("sd") if "sd" in lower else None
        data = np.array([[float(v) for v in r] for r in rows[1:] if r], float)
        b1 = np.unique(data[:, i_b1])
        off = np.unique(data[:, i_off])
        z = np.full((b1.size, off.size), np.nan)
        sd = np.full_like(z, np.nan) if i_sd is not None else None
        for r in data:
            k = int(np.argmin(np.abs(b1 - r[i_b1])))
            j = int(np.argmin(np.abs(off - r[i_off])))
            z[k, j] = r[i_z]
            if sd is not None:
                sd[k, j] = r[i_sd]
    else:
        # wide
        off = np.array([float(r[0]) for r in rows[1:] if r])
        b1 = np.array([float(h) for h in header[1:]])
        z = np.array([[float(v) for v in r[1:]] for r in rows[1:] if r], float).T
        sd = None

    if np.isnan(z).any():
        raise ValueError("the (B1, offset) grid is incomplete: every page needs a value")
    order = np.argsort(off)
    z = z[:, order]
    if sd is not None:
        sd = sd[:, order]
    pages = PageSet(b1_uT=np.sort(b1), offsets_ppm=off[order], ts=float(
        _read_meta(path).get("ts", 2.0)))
    return Spectrum(pages=pages, z=z.ravel(order="C"), sd=None if sd is None else sd.ravel(),
                    meta=_read_meta(path))


def _read_meta(path: Path) -> dict:
    """Metadata from leading ``# key: value`` comment lines."""
    meta = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            body = line.lstrip("#").strip()
            if ":" in body:
                k, v = body.split(":", 1)
                try:
                    meta[k.strip()] = float(v.strip())
                except ValueError:
                    meta[k.strip()] = v.strip()
    return meta


def save_zspectrum(path: str | Path, pages: PageSet, z: np.ndarray,
                   meta: Optional[dict] = None) -> None:
    """Write a Z-spectrum in the tidy format read by :func:`load_zspectrum`."""
    path = Path(path)
    b1, off = pages.broadcast()
    z = np.asarray(z, float).ravel()
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write(f"# ts: {pages.ts}\n")
        for k, v in (meta or {}).items():
            fh.write(f"# {k}: {v}\n")
        w = csv.writer(fh)
        w.writerow(["b1_uT", "offset_ppm", "z"])
        for a, b, c in zip(b1, off, z):
            w.writerow([f"{a:.6g}", f"{b:.6g}", f"{c:.8g}"])
