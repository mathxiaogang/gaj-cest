"""Cost of one Jacobian and cost of a complete fit, for the configurations of
Table 3 of the paper.

Two stages:

  A  cost of a single forward evaluation and of one Jacobian of each kind, on the
     full 147-page grid -- this isolates the mechanism (nu = Jacobian / forward);
  B  complete multi-start fits at a grid of tolerances, scored against the
     accuracy requirement of Section 3.2.

The default is a reduced protocol (a few starts, one truth point) so that the
script finishes in minutes on a laptop; ``--full`` runs the paper's protocol
(15 starts, 10 cases, three timing repetitions) and takes hours.

Usage
-----
    python experiments/benchmark.py                      # reduced, ~2-5 min
    python experiments/benchmark.py --starts 5 --tols 1e-7 1e-8
    python experiments/benchmark.py --full --out results/full.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaj import (FivePoolBM, GAJConfig, PageSet, exact_response, fit, fit_multi_start,
                 jacobian_central_exact,
                 jacobian_central_surrogate, jacobian_gaj, jacobian_vanloan, make_starts,
                 meets_requirement, simulate, surrogate_response, wilson_interval)

CONFIGS = {
    "GAJ": "gaj",
    "Batch-C": "central_surrogate",
    "Trad-C": "central_exact",
}


def timed(fn, repeat: int = 3):
    best = np.inf
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def stage_a(model, truth):
    """Cost of one forward evaluation and of one Jacobian, per configuration."""
    cfg = GAJConfig()
    t_fwd, _ = timed(lambda: surrogate_response(model, truth, cfg))
    t_exact, _ = timed(lambda: exact_response(model, truth))
    t_gaj, (_, info) = timed(lambda: jacobian_gaj(model, truth, cfg), repeat=2)
    t_batch, _ = timed(lambda: jacobian_central_surrogate(model, truth, cfg), repeat=1)
    t_trad, _ = timed(lambda: jacobian_central_exact(model, truth, cfg.sh), repeat=1)
    rows = [
        ("forward, surrogate", t_fwd, 1.0),
        ("forward, exact (expm)", t_exact, t_exact / t_fwd),
        ("Jacobian, GAJ (gated series)", t_gaj, t_gaj / t_fwd),
        ("Jacobian, Batch-C (central diff. of surrogate)", t_batch, t_batch / t_fwd),
        ("Jacobian, Trad-C (central diff. of exact model)", t_trad, t_trad / t_fwd),
    ]
    return {
        "t_forward_ms": 1e3 * t_fwd,
        "rows": [(n, 1e3 * t, nu) for n, t, nu in rows],
        "gaj_accepted_order": info.accepted_order,
        "gaj_eta": info.eta,
        "gaj_used_fallback": info.used_fallback,
        "gaj_term_norms": info.term_norms,
        "speedup_vs_batchC": t_batch / t_gaj,
        "speedup_vs_tradC": t_trad / t_gaj,
    }


def stage_b(model, n_starts, tol, configs, cases, seed):
    """Complete multi-start fits; the best of the starts is scored.

    This is the unit of the paper's protocol: one "timed fit" is a whole start set
    (15 starts there), scored against the accuracy requirement, and the speed-ups
    are ratios of the times of complete fits.
    """
    results = []
    for case in cases:
        x_truth = case["truth"]
        z = case["z"]
        starts = make_starts(model, x_truth, n_total=n_starts, seed=seed)
        for name in configs:
            t0 = time.perf_counter()
            best, runs = fit_multi_start(model, z, starts, jacobian=CONFIGS[name],
                                         tol=tol, max_nfev=250, keep_all=True)
            wall = time.perf_counter() - t0
            met = meets_requirement(model, best.x, x_truth)
            results.append({
                "case": case["name"], "config": name,
                "seconds": wall,
                "nfev": int(np.sum([r.nfev for r in runs])),
                "njev": int(np.sum([r.njev for r in runs])),
                "sse": float(np.sum(np.asarray(best.trace[:-1]) ** 2)),
                "max_error": met["max_error"], "ok": met["ok"],
                "spectral_error": met["spectral_error"],
                "accepted_orders": best.accepted_orders,
                "fallbacks": int(np.sum([r.fallbacks for r in runs])),
            })
    return results


def summarise(results, tol):
    """Median time, pass rate and speed-ups per configuration."""
    names = sorted({r["config"] for r in results}, key=lambda n: list(CONFIGS).index(n))
    by_case = {}
    for r in results:
        by_case.setdefault(r["case"], {}).setdefault(r["config"], []).append(r)

    out = {"tol": tol, "configs": {}}
    for name in names:
        ok = sum(1 for r in results if r["config"] == name and r["ok"])
        n = sum(1 for r in results if r["config"] == name)
        secs = [r["seconds"] for r in results if r["config"] == name] or [float("nan")]
        out["configs"][name] = {
            "pass_rate": ok / n if n else float("nan"),
            "wilson": wilson_interval(ok, n),
            "median_seconds": float(np.median(secs)),
            "median_nfev": float(np.median([r["nfev"] for r in results if r["config"] == name])),
            "median_max_error": float(np.median([r["max_error"] for r in results
                                                 if r["config"] == name])),
            "fallback_fraction": float(np.mean([r["fallbacks"] / max(r["njev"], 1)
                                                for r in results if r["config"] == name])),
        }
    # speed-ups per case (paired), then median over cases
    for other in names:
        if other == "GAJ":
            continue
        ratios = []
        for case, per in by_case.items():
            if "GAJ" in per and other in per:
                ga = [r["seconds"] for r in per["GAJ"] if r["ok"]] or [float("nan")]
                ot = [r["seconds"] for r in per[other] if r["ok"]] or [float("nan")]
                a, b = float(np.median(ga)), float(np.median(ot))
                if np.isfinite(a) and np.isfinite(b) and a > 0:
                    ratios.append(b / a)
        if ratios:
            out["configs"]["GAJ"][f"speedup_vs_{other}"] = {
                "median": float(np.median(ratios)), "min": float(np.min(ratios)),
                "max": float(np.max(ratios)), "n_cases": len(ratios)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--starts", type=int, default=3, help="starting points per case")
    ap.add_argument("--tols", type=float, nargs="+", default=[1e-8])
    ap.add_argument("--cases", type=int, default=1, help="number of truth points")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--full", action="store_true", help="paper protocol (slow)")
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "benchmark.json"))
    args = ap.parse_args(argv)

    if args.full:
        args.starts, args.cases = 15, 10
        args.tols = [1e-7, 1e-8, 1e-9]

    pages = PageSet(b1_uT=[0.5, 0.9, 1.6], offsets_ppm=np.arange(-6.0, 6.001, 0.25), ts=2.0)
    model = FivePoolBM(pages)
    truth = model.truth()

    print(f"grid: {pages.n_amplitudes} amplitudes x {pages.n_offsets} offsets "
          f"= {pages.n_pages} pages; p = {model.p} parameters")
    a = stage_a(model, truth)
    print("\n[A] cost of one evaluation (best of 3)")
    for name, ms, nu in a["rows"]:
        print(f"    {name:52s} {ms:8.2f} ms   x{nu:5.2f} forward")
    print(f"    GAJ gate: order {a['gaj_accepted_order']}, eta {a['gaj_eta']:.2e}, "
          f"fallback {a['gaj_used_fallback']}")
    print(f"    Jacobian speed-up: vs Batch-C x{a['speedup_vs_batchC']:.1f}, "
          f"vs Trad-C x{a['speedup_vs_tradC']:.1f}")

    # cases: the primary phantom plus perturbations of it (log-normal, 0.35)
    rng = np.random.default_rng(424242)
    cases = []
    for j in range(args.cases):
        xt = truth if j == 0 else truth * np.exp(rng.normal(0.0, 0.2, truth.size))
        lo, hi = model.bounds()
        xt = np.clip(xt, lo, hi)
        cases.append({"name": f"case{j + 1}", "truth": xt,
                      "z": simulate(model, xt, noise=0.0)})
    all_results = []
    summary = []
    for tol in args.tols:
        res = stage_b(model, args.starts, tol, args.configs, cases, args.seed)
        all_results.extend(res)
        s = summarise(res, tol)
        summary.append(s)
        print(f"\n[B] fits at t = {tol:g}  ({len(cases)} case(s) x {args.starts} starts)")
        for name, st in s["configs"].items():
            extra = ""
            if "speedup_vs_Batch-C" in st:
                extra += f"  x{st['speedup_vs_Batch-C']['median']:.1f} vs Batch-C"
            if "speedup_vs_Trad-C" in st:
                extra += f"  x{st['speedup_vs_Trad-C']['median']:.1f} vs Trad-C"
            print(f"    {name:8s} pass {100 * st['pass_rate']:5.0f}%  "
                  f"median {st['median_seconds']:6.2f} s  "
                  f"nfev {st['median_nfev']:5.0f}  "
                  f"max err {100 * st['median_max_error']:5.2f}%  "
                  f"fallback {100 * st['fallback_fraction']:4.0f}%{extra}")

    payload = {"stage_a": a, "summary": summary, "runs": all_results,
               "protocol": {"starts": args.starts, "cases": len(cases), "tols": args.tols,
                            "pages": pages.n_pages, "p": model.p,
                            "python": sys.version.split()[0],
                            "numpy": np.__version__}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
