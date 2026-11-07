"""Fit a measured Z-spectrum with GAJ and report what the fit achieved.

This is the entry point a reviewer or a collaborator should use with their own
data: the reader, the model, the fit and the diagnostics are the ones used for
the paper, and nothing in the pipeline knows whether the file came from a
scanner or from a simulator.

    python experiments/real_data_demo.py --csv data/example_zspectrum_SIMULATED.csv
    python experiments/real_data_demo.py --csv my_spectrum.csv --starts 6 --tol 1e-9

The CSV format is described in ``data/README.md``.  If the file carries
``# truth_<parameter>: value`` comments (as the bundled simulated example does)
the script also prints the relative errors; with measured data it prints the
residual, the fitted parameters and their uncertainties instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaj import (FivePoolBM, GAJConfig, fit_multi_start, jacobian_gaj, load_zspectrum,
                 make_starts, relative_errors, surrogate_response)


def parameter_table(model, x, x0=None, sd=None):
    names = model.parameter_names()
    lo, hi = model.bounds()
    lines = [f"    {'parameter':10s} {'fitted':>12s} {'bound range':>22s}"
             + (f" {'1 sd':>10s}" if sd is not None else "")
             + ("   (truth)" if x0 is not None else "")]
    for i, name in enumerate(names):
        row = f"    {name:10s} {x[i]:12.4g} [{lo[i]:.3g}, {hi[i]:.3g}]".ljust(46)
        if sd is not None:
            row += f" {sd[i]:10.3g}"
        if x0 is not None:
            row += f"   ({x0[i]:.4g})"
        lines.append(row)
    return "\n".join(lines)


def parameter_sd(model, x, z, cfg):
    """One-sigma uncertainties from the Gauss-Newton approximation.

    With the surrogate residual r(x) = Z_S(x) - z and the Jacobian J, the
    linearised covariance is (J^T J)^{-1} * SSE / (n - p); the square roots give
    the standard errors of the parameters.  Reported for measured data, where the
    truth is unknown.
    """
    J, _ = jacobian_gaj(model, x, cfg)
    r = surrogate_response(model, x, cfg) - z
    n, p = J.shape
    sse = float(r @ r)
    try:
        cov = np.linalg.inv(J.T @ J) * sse / max(n - p, 1)
        return np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(ROOT / "data" / "example_zspectrum_SIMULATED.csv"))
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--max-nfev", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--figure", default=str(ROOT / "figures" / "real_data_fit.png"))
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    spec = load_zspectrum(args.csv)
    model = FivePoolBM(spec.pages)
    print(f"file      : {args.csv}")
    print(f"pages     : {spec.pages.n_pages} "
          f"({spec.pages.n_amplitudes} amplitudes x {spec.pages.n_offsets} offsets), "
          f"Ts = {spec.pages.ts:g} s")
    if spec.meta.get("source"):
        print(f"source    : {spec.meta['source']}")

    # starts: around the literature values (the bounds of Table B.1)
    x_ref = model.truth()
    starts = make_starts(model, x_ref, n_total=args.starts, seed=args.seed)
    cfg = GAJConfig()
    best, runs = fit_multi_start(model, spec.z, starts, jacobian="gaj",
                                 tol=args.tol, max_nfev=args.max_nfev, cfg=cfg,
                                 keep_all=True)

    z_fit = surrogate_response(model, best.x, cfg)
    resid = z_fit - spec.z
    n_pages = spec.pages.n_pages
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    r2 = 1.0 - np.sum(resid ** 2) / np.sum((spec.z - spec.z.mean()) ** 2)

    print(f"\nfit       : {len(runs)} starts at t = {args.tol:g}, "
          f"{best.seconds:.1f} s (best start), f-evals {best.nfev}, J-evals {best.njev}")
    print(f"            gate orders {sorted(set(best.accepted_orders))}, "
          f"fallbacks {best.fallbacks}/{best.njev}")
    print(f"residual  : RMSE {rmse:.4f} Z-units over {n_pages} pages, R^2 {r2:.5f}")
    print(f"            max |residual| {np.abs(resid).max():.4f} "
          f"at offset {spec.pages.broadcast()[1][np.argmax(np.abs(resid))]:+.2f} ppm")

    sd = parameter_sd(model, best.x, spec.z, cfg)
    truth = None
    truth_values = [spec.meta.get(f"truth_{n}") for n in model.parameter_names()]
    if all(v is not None for v in truth_values):
        truth = np.array(truth_values, float)
    print("\n" + parameter_table(model, best.x, truth, sd))
    if truth is not None:
        err = relative_errors(model, best.x, truth) * 100
        n = model.n_pools
        print("\nrelative error of the 22 monitored quantities (%):")
        print("    parameters  " + " ".join(f"{e:5.2f}" for e in err[:17]))
        print("    ksw*M0s     " + " ".join(f"{e:5.2f}" for e in err[17:]))
        print(f"    worst {err.max():.2f}%  (requirement 5%)")

    if not args.no_figure:
        make_figure(spec, z_fit, Path(args.figure))
        print(f"\nfigure    : {args.figure}")
    return 0


def make_figure(spec, z_fit, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b1, off = spec.pages.broadcast()
    levels = np.unique(b1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    colours = plt.cm.viridis(np.linspace(0.15, 0.8, levels.size))
    for c, level in zip(colours, levels):
        m = b1 == level
        order = np.argsort(off[m])
        axes[0].plot(off[m][order], spec.z[m][order], "o", ms=2.6, color=c,
                     alpha=0.75, label=f"data, B1 = {level:g} uT")
        axes[0].plot(off[m][order], z_fit[m][order], "-", lw=1.2, color=c)
    axes[0].set_xlabel("offset (ppm)")
    axes[0].set_ylabel("Z")
    axes[0].set_title("measured (dots) and fitted (lines)")
    axes[0].invert_xaxis()
    axes[0].legend(fontsize=7, frameon=False)
    for c, level in zip(colours, levels):
        m = b1 == level
        axes[1].plot(off[m], (z_fit - spec.z)[m], ".", ms=3, color=c)
    axes[1].axhline(0, color="0.3", lw=0.8)
    axes[1].set_xlabel("offset (ppm)")
    axes[1].set_ylabel("residual (Z)")
    axes[1].set_title("fit residual")
    axes[1].invert_xaxis()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
