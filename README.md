# GAJ — gated analytic Jacobian for linear drive-and-observe models

Reference implementation of

[![tests](https://github.com/mathxiaogang/gaj-cest/actions/workflows/ci.yml/badge.svg)](https://github.com/mathxiaogang/gaj-cest/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> G. Xiao, *GAJ: A Gated Analytic Jacobian for Fast Fitting of Linear Dynamical
> Models with Time-Scale Separation, with Application to Multi-Pool CEST*.

GAJ supplies a Gauss–Newton-type solver with the residual and the Jacobian of a
multi-pool Bloch–McConnell model at a few forward evaluations per iteration
instead of `2p + 1`. The residual comes from a **surrogate** of the response (the
steady state plus the two slowest modes of the generator); the Jacobian comes
from the **analytic derivative of that surrogate** (divided differences for the
retained modes, a reduced-resolvent series for the coupling to the discarded
ones); a **run-time gate** accepts the series when its last terms are small and
otherwise falls back to batched central differences of the surrogate.

Everything below is reproducible on a laptop; no data are hidden behind a
download.

---

## Quick start

```bash
git clone https://github.com/mathxiaogang/gaj-cest
cd gaj-cest
python -m pip install -r requirements.txt      # numpy, scipy, matplotlib
export PYTHONPATH=src                          # or: pip install -e .

python -m gaj.cli info                         # model size, H1-H3, gate behaviour
python -m pytest tests -q                      # 31 tests, ~3 s

# fit the paper's phantom (noiseless, 147 pages, 17 parameters)
python experiments/real_data_demo.py --csv data/phantom_zspectrum.csv --starts 3

# cost of one Jacobian and of complete fits: GAJ vs conventional vs control
python experiments/benchmark.py --starts 3 --tols 1e-8 1e-9
```

## What the code reproduces

**1. The algorithm itself.** Forward error of the surrogate against the exact
response (`scipy`/batched matrix exponential) is `4.5e-12` at `Ts = 2 s`; the
gated Jacobian agrees with directional finite differences of the surrogate to
`1e-8 … 1e-6`; the series agrees with the exact reduced resolvent to `6e-7`.
These are asserted in `tests/`.

**2. Accuracy and identifiability.** The paper's phantom is recovered from three
starts at `t = 1e-9`:

| quantity | result |
|---|---|
| worst of the 22 monitored relative errors | **0.42 %** (requirement 5 %) |
| `e_spec` | `6.7e-7` (requirement `1e-5`) |
| residual `RMSE` on 147 pages | `6.6e-7` Z-units (noiseless data) |

**3. Speed at equal accuracy.** One complete multi-start fit on this machine
(Python 3.12, NumPy 2.2, single thread, `t = 1e-9`, 3 starts, 147 pages):

| configuration | Jacobian per call | complete fit | speed-up | pass rate |
|---|---|---|---|---|
| **GAJ** (gated analytic) | 12.1 ms (3.5 forward) | **2.2 s** | 1× | 3/3 |
| Batch-C (central diff. of surrogate) | 153.5 ms | 12.9 s | 5.9× | 3/3 |
| Trad-C (central diff. of exact model) | 533.9 ms | 73.3 s | 33.5× | 3/3 |

At `t = 1e-8` none of the three passes on this start set (the paper reports 46 %
of *cases* at that tolerance and 97 % at `1e-9`); at `1e-9` all three pass, which
is the property GAJ is built to preserve — the speed must not cost accuracy.

The paper's numbers come from a MATLAB implementation with 8 threads and 15
starts over 10 start sets (18.8× vs forward differences, 35.4× vs central
differences, ≈8× vs the consistent-Jacobian control). The ratios above are
measured within one run, in one implementation, as the paper insists; absolute
times are not comparable across implementations. `experiments/benchmark.py
--full` runs the paper's full protocol (hours).

## Using it on measured data

The reader, the model, the fit and the diagnostics are the same as for the
simulated phantom — nothing in the pipeline knows where the numbers came from.

```bash
# your spectrum, tidy CSV: b1_uT, offset_ppm, z  (see data/README.md)
python -m gaj.cli fit --csv my_zspectrum.csv --starts 5 --tol 1e-9

# or with the full diagnostic report + figure
python experiments/real_data_demo.py --csv my_zspectrum.csv --figure figures/my_fit.png
```

What the report contains: the residual (`RMSE`, `R²`, worst offset), the fitted
parameters with their bounds and one-sigma standard errors from the
Gauss–Newton covariance, and — when the file carries `# truth_<name>:` comments —
the 22 monitored relative errors.

`data/example_zspectrum_SIMULATED.csv` is *simulated to look like a measurement*
(Rician noise `sd = 0.01`, a 0.05 ppm water-frequency error, a 1.05 B1 scaling).
It exercises the measured-data path end to end: the fit reproduces the spectrum
(`R² = 0.998`, residual at the noise level) while the parameters are only weakly
identified — the sd column reports exactly that. This is a property of the
inverse problem at 1 % noise with 17 parameters, not of the Jacobian, and it is
the reason the paper's accuracy requirement is stated for noiseless data. No
measured CEST data are redistributed here; `data/README.md` lists public sources
and the preprocessing they need.

## Repository layout

```
src/gaj/
  model.py             linear drive-and-observe class (M1-M3), pages, systems
  bloch_mcconnell.py   five-pool BM phantom, closed-form dA/dc/dM0 (Table B.1)
  eigen.py             Procedure 1: batched two-sided power iteration + guard
  surrogate.py         Section 4.2: Z_S, H2 truncation error, exact response
  jacobian.py          Sections 4.3-4.5: J0, reduced-resolvent series, gate,
                       central-difference fallback, Van Loan reference
  fit.py               least_squares wrapper (all configurations of Table 3)
  metrics.py           22 monitored errors, e_spec, Wilson intervals
  data.py              Z-spectrum I/O, simulated/measured imperfections, starts
  linalg.py            batched scaling-and-squaring expm
  cli.py               gaj info | simulate | fit | bench
experiments/
  benchmark.py         Jacobian costs and complete fits (Table 3 configurations)
  real_data_demo.py    fit a CSV, report fit quality, parameters and sd
  make_example_data.py regenerate the bundled spectra
tests/                 31 tests: derivatives, eigenpairs, surrogate, Jacobian, gate, fit
data/                  Z-spectra (simulated) and the format specification
docs/method.md         equation -> code map, and the three deviations from the paper
docs/reproducibility.md  what each script measures, what was and was not reproduced
```

## Three implementation notes a reader should know

1. **The `u_q` recursion is written in a numerically equivalent form.** The paper
   subtracts the slow components `sum_s alpha_s lambda_s^{-q} v_s` from `B^q e0`
   at every order; in double precision that is an O(1) cancellation and the terms
   stop decreasing after about five orders (`eta` bottoms out near `3e-8`, above
   `tau = 1e-8`, so the gate rejects a series that is in fact converged). Starting
   from `Q_F e0` and projecting out only the *computed* leakage keeps the same
   mathematics and reaches `eta = 7e-10` at order 6 — the gate then accepts.
   `docs/method.md` has the two-line derivation.
2. **The retained eigenpairs are only as accurate as Procedure 1 makes them.** With
   `pw = 8`, `nit = 8` and a modulus ratio `|lambda_2|/|lambda_3| ~ 0.85`, the
   second pair is resolved to about `1e-3`; this is the paper's own remark about
   the iteration, and it is why the gate's estimate is not an error bound.
3. **Complex retained pairs are not supported by the analytic path.** The paper's
   Proposition 1 assumes simple real modes; the phantom's slow spectrum is real on
   every page. `GAJConfig(eig="exact")` uses LAPACK and is used only for reference
   and diagnostics.

## Citing

`CITATION.cff` is provided. If you use this code, please cite the paper; if you
use the code itself, cite

> Xiao, G. (2026). *gaj-cest* (v0.1.0) [Computer software].
> https://github.com/mathxiaogang/gaj-cest

Archived releases will be deposited on Zenodo; the DOI goes here once the first
release is tagged.

## Licence

MIT (see `LICENSE`).
