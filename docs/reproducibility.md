# Reproducibility: what each script measures, and what was reproduced

All numbers below were produced on one Windows 11 desktop (Python 3.12.10,
NumPy 2.2.6, SciPy 1.13.1, OpenBLAS, single thread for the timings). Ratios are
read within one run, as the paper insists; absolute times are not comparable with
the paper's MATLAB implementation.

## Commands

| command | what it does | time |
|---|---|---|
| `python -m pytest tests -q` | 31 checks of the model, eigenpairs, surrogate, Jacobian, gate and fit | ~3 s |
| `python -m gaj.cli info` | model size, H1–H3 diagnostics, gate statistic and accepted order | ~1 s |
| `python experiments/real_data_demo.py --csv data/phantom_zspectrum.csv` | fits the paper's phantom, prints the 22 monitored errors | ~5 s |
| `python experiments/benchmark.py --starts 3 --tols 1e-8 1e-9` | Jacobian costs and complete fits for the three configurations of Table 3 | ~3 min |
| `python experiments/benchmark.py --full` | the paper's protocol: 15 starts, 10 cases, three tolerances | hours |

`results/benchmark_quick.json` is the output of the fourth command; `--full`
writes the same schema.

## Stage A — cost of one evaluation (147 pages)

| | time | × forward |
|---|---|---|
| forward, surrogate | 3.42 ms | 1.00 |
| forward, exact (`expm`) | 7.24 ms | 2.12 |
| **Jacobian, GAJ** | **12.09 ms** | **3.54** |
| Jacobian, Batch-C | 153.54 ms | 44.9 |
| Jacobian, Trad-C | 533.86 ms | 156.2 |

Gate: order 6, `eta = 6.97e-10`, no fallback. The Jacobian costs 3.5 forward
evaluations, i.e. `nu = 3.5` (the paper's Table 1 quotes `nu ~ 2`; the difference
is bookkeeping — this implementation rebuilds the system and the power iteration
on every Jacobian rather than caching them from the residual evaluation).

## Stage B — complete fits at equal accuracy (3 starts, 1 case)

| tolerance | configuration | pass rate | median time | median max error | speed-up |
|---|---|---|---|---|---|
| `1e-8` | GAJ | 0/1 | 1.52 s | 7.80 % | — |
| `1e-8` | Batch-C | 0/1 | 8.37 s | 7.79 % | 5.5× \* |
| `1e-8` | Trad-C | 0/1 | 46.14 s | 7.79 % | 30.4× \* |
| `1e-9` | GAJ | 1/1 | 2.19 s | 4.65 % | — |
| `1e-9` | Batch-C | 1/1 | 12.91 s | 4.64 % | 5.9× |
| `1e-9` | Trad-C | 1/1 | 73.32 s | 4.64 % | 33.5× |

\* At `t = 1e-8` no fit passes, so no speed-up at equal accuracy can be formed;
the entries are the plain ratio of the median times of the same run and are shown
only to indicate the cost ordering at that tolerance.

Reading: at `t = 1e-9` all three configurations meet the requirement and GAJ is
5.9× faster than the consistent-Jacobian control and 33.5× faster than
conventional central differences — the same ordering and a similar magnitude as
the paper (18.8× and 35.4×, ≈8× for the controls). At `t = 1e-8` none of them
passes on this start set; the paper reports 46 % of *cases* at that tolerance over
12 truth points, so a single case is not informative about the rate.

The paper's 15-start protocol matters here: the fit that is scored is the one with
the smallest sum of squares among the starts, and the 5 % requirement is hard
enough that the best-of-15 is what makes it pass at `t = 1e-9` with probability
97 %.

## What was *not* reproduced

* **The MATLAB timings.** Absolute seconds and the 8-thread ratios require the
  original implementation, which is not redistributed here.
* **The 12 truth points of the Latin hypercube.** `--cases N` draws perturbations
  of the primary phantom (log-normal, 0.2); they do not coincide with the paper's
  hypercube.
* **Figures 2 and 3 of the paper** (pass rate vs tolerance, forward error vs
  saturation time). The scripts to produce them are a `for` loop over `ts` and
  over tolerances using the same functions; they are not included to keep the
  runtime of the repository under a few minutes.
* **The gate calibration against reference Jacobians (Section 7.2).** The pieces
  exist (`jacobian_vanloan` for the reference, `jacobian_surrogate_direct` for
  the exact derivative of the surrogate) but the sweep is not scripted.

## Determinism

All random draws use `numpy.random.default_rng` with the seed given on the command
line (`--seed`, default 5000 for starts and 424242 for the benchmark cases). The
optimiser and the model are deterministic; timing is not, which is why every
timing is the best of three repetitions in stage A and the median over repeats
would be needed for a full protocol.
