"""Create a Z-spectrum file that exercises the measured-data path.

The file is **simulated** -- no measured CEST data are redistributed with this
repository -- but it is generated the way a measurement differs from the ideal
phantom: Rician noise, a water-frequency error and a saturation-amplitude error.
The fitting code cannot tell the difference between this file and a scanner
export, which is the point: it is the same reader and the same fit.

    python experiments/make_example_data.py --out data/example_zspectrum_SIMULATED.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaj import FivePoolBM, PageSet, save_zspectrum, simulate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data" / "example_zspectrum_SIMULATED.csv"))
    ap.add_argument("--kind", choices=["phantom", "measured"], default="measured",
                    help="phantom: the paper's noiseless data; measured: noise + B0/B1 errors")
    ap.add_argument("--noise", type=float, default=0.01, help="Rician noise sd of Z")
    ap.add_argument("--b0", type=float, default=0.05, help="water-frequency error, ppm")
    ap.add_argument("--b1", type=float, default=1.05, help="B1 scaling error")
    ap.add_argument("--ts", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args(argv)

    pages = PageSet(b1_uT=[0.5, 0.9, 1.6],
                    offsets_ppm=np.arange(-6.0, 6.001, 0.25), ts=args.ts)
    model = FivePoolBM(pages)
    truth = model.truth()
    if args.kind == "phantom":
        args.noise, args.b0, args.b1 = 0.0, 0.0, 1.0
    z = simulate(model, truth, noise=args.noise, b0_shift_ppm=args.b0,
                 b1_scale=args.b1, rician=args.noise > 0, seed=args.seed)

    source = ("SIMULATED phantom of the paper, noiseless"
              if args.kind == "phantom" else
              "SIMULATED to mimic a measurement (Rician noise, B0 and B1 errors); "
              "no measured data are redistributed")
    meta = {"source": source,
            "noise_sd": args.noise, "b0_error_ppm": args.b0, "b1_error": args.b1}
    for name, value in zip(model.parameter_names(), truth):
        meta[f"truth_{name}"] = float(value)
    save_zspectrum(args.out, pages, z, meta=meta)
    print(f"wrote {args.out}  ({pages.n_pages} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
