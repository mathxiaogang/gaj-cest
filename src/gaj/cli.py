"""Command-line entry point: ``gaj info | simulate | fit | bench``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import (FivePoolBM, GAJConfig, PageSet, exact_response, fit_multi_start, forward,
               jacobian_gaj, load_zspectrum, make_starts, meets_requirement, save_zspectrum,
               simulate, spectral_diagnostics, surrogate_response)
from .eigen import exact_slow_modes

def _pages(args) -> PageSet:
    off = np.arange(-6.0, 6.001, 0.25)
    return PageSet(b1_uT=[0.5, 0.9, 1.6], offsets_ppm=off, ts=args.ts)


def cmd_info(args) -> int:
    pages = _pages(args)
    model = FivePoolBM(pages)
    x = model.truth()
    cfg = GAJConfig(stability_check=args.stability_check)

    state = forward(model, x[None, :], cfg)
    modes_exact = exact_slow_modes(model.build(x).A, cfg.n_modes)
    diag = spectral_diagnostics(model.build(x).A, modes_exact)
    lam_star = (cfg.tau_g ** (1.0 / (2 * cfg.nit)) * np.finfo(float).eps ** (-1.0 / cfg.pw))

    print(f"model        : five-pool Bloch-McConnell, m = {model.m}, p = {model.p}")
    print(f"pages        : {pages.n_amplitudes} amplitudes x {pages.n_offsets} offsets "
          f"= {pages.n_pages}, Ts = {pages.ts:g} s")
    print(f"settings     : P = {cfg.n_modes}, pw = {cfg.pw}, nit = {cfg.nit}, "
          f"qmax = {cfg.qmax}, tau = {cfg.tau:g}, sh = {cfg.sh:g}")
    print(f"H1           : max Re(lambda) = {diag['max_real_eigenvalue'].max():.3f} "
          f"(negative on every page: {bool(np.all(diag['max_real_eigenvalue'] < 0))})")
    print(f"H3           : rho (slowest/fastest) median {np.nanmedian(diag['rho']):.3f}, "
          f"max {np.nanmax(diag['rho']):.3f} (must be < 1)")
    print(f"guard        : Eq. (7) threshold |lambda*| ~ {lam_star:.0f} 1/s "
          f"(the paper's Table 2 quotes ~11 1/s for the same settings; Eq. (7) "
          f"drops factors of order one)")
    print(f"retained     : {int(state.modes.kept.sum(1).min())}-"
          f"{int(state.modes.kept.sum(1).max())} modes per page")

    z_s = surrogate_response(model, x, cfg)
    z_e = exact_response(model, x)[0]
    print(f"surrogate    : max |Z_S - Z_exact| = {np.abs(z_s - z_e).max():.2e}")

    J, info = jacobian_gaj(model, x, cfg)
    print(f"Jacobian     : gate order {info.accepted_order}, eta {info.eta:.2e}, "
          f"fallback {info.used_fallback}")
    print(f"               |J|max = {np.abs(J).max():.3e}")
    return 0


def cmd_simulate(args) -> int:
    pages = _pages(args)
    model = FivePoolBM(pages)
    z = simulate(model, model.truth(), noise=args.noise, b0_shift_ppm=args.b0,
                 b1_scale=args.b1, rician=args.rician, seed=args.seed)
    save_zspectrum(args.out, pages, z, meta={"source": "gaj simulate", "noise_sd": args.noise})
    print(f"wrote {args.out} ({pages.n_pages} pages, noise {args.noise:g})")
    return 0


def cmd_fit(args) -> int:
    if args.csv:
        spec = load_zspectrum(args.csv)
        pages, z, meta = spec.pages, spec.z, spec.meta
    else:
        pages = _pages(args)
        model0 = FivePoolBM(pages)
        z = simulate(model0, model0.truth(), noise=args.noise, seed=args.seed)
        meta = {}
    model = FivePoolBM(pages)
    cfg = GAJConfig()
    starts = make_starts(model, model.truth(), n_total=args.starts, seed=args.seed)
    best, runs = fit_multi_start(model, z, starts, jacobian=args.jacobian, tol=args.tol,
                                 max_nfev=args.max_nfev, cfg=cfg, keep_all=True)
    resid = surrogate_response(model, best.x, cfg) - z
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    print(f"fit: {len(runs)} starts, t = {args.tol:g}, jacobian = {args.jacobian}")
    print(f"     {best.seconds:.2f} s for the best start, nfev {best.nfev}, "
          f"njev {best.njev}, RMSE {rmse:.4f}")
    truth_values = [meta.get(f"truth_{n}") for n in model.parameter_names()]
    if all(v is not None for v in truth_values):
        met = meets_requirement(model, best.x, np.array(truth_values, float))
        print(f"     max monitored error {100 * met['max_error']:.2f}% "
              f"(requirement 5%) -> {'PASS' if met['ok'] else 'FAIL'}")
    for name, value in zip(model.parameter_names(), best.x):
        print(f"     {name:8s} {value:12.5g}")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"x": best.x.tolist(), "names": model.parameter_names(),
             "rmse": rmse, "seconds": best.seconds, "nfev": best.nfev}, indent=2),
            encoding="utf-8")
    return 0


def cmd_bench(args) -> int:
    from pathlib import Path as _P
    import runpy

    script = _P(__file__).resolve().parents[2] / "experiments" / "benchmark.py"
    sys.argv = [str(script)] + args.rest
    runpy.run_path(str(script), run_name="__main__")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gaj", description=__doc__)
    ap.add_argument("--ts", type=float, default=2.0)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="model size, H1-H3 diagnostics, gate behaviour")
    p.add_argument("--stability-check", action="store_true")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("simulate", help="write a simulated Z-spectrum")
    p.add_argument("--out", default="zspectrum.csv")
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--b0", type=float, default=0.0)
    p.add_argument("--b1", type=float, default=1.0)
    p.add_argument("--rician", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("fit", help="fit a Z-spectrum (measured CSV or simulated)")
    p.add_argument("--csv", default=None)
    p.add_argument("--starts", type=int, default=4)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--max-nfev", type=int, default=250)
    p.add_argument("--jacobian", default="gaj",
                   choices=["gaj", "central_surrogate", "central_exact", "vanloan"])
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("bench", help="run experiments/benchmark.py")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_bench)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
