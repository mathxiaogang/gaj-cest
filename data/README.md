# Data

## What is bundled

| file | what it is |
|---|---|
| `phantom_zspectrum.csv` | the paper's phantom, **noiseless**, 3 amplitudes x 49 offsets, `Ts = 2 s` |
| `example_zspectrum_SIMULATED.csv` | **simulated to mimic a measurement**: Rician noise `sd = 0.01`, 0.05 ppm water-frequency error, 1.05 B1 scaling |

Neither file is measured data. The second one exists so that the measured-data
path (noisy input, wrong B0/B1, unknown truth) is exercised by the test suite and
by the demo; it is produced by `experiments/make_example_data.py --kind measured`
and regenerated deterministically from seed 20261024.

Both files carry the generating parameters as `# truth_<name>: value` comments, so
the demo can print the relative errors. **Delete those comments when preparing a
real file** — or leave them out; the reader ignores everything it does not know.

## Format

Tidy CSV, one row per page, with a header:

```
# ts: 2.0                     optional metadata (key: value, other keys are ignored)
# source: Siemens 3T, volunteer 3, voxel (12, 31), 2026-05-04
b1_uT,offset_ppm,z
0.5,-6,0.98123456
0.5,-5.75,0.97890123
...
```

* `b1_uT` — saturation amplitude in microtesla (any monotone label works: the
  code only groups pages by it).
* `offset_ppm` — saturation offset in ppm, relative to water.
* `z` — the Z-spectrum value, `Z = Mz(Ts) / Mz(0)`, dimensionless, one row per
  (amplitude, offset) pair.
* `sd` — optional per-point standard deviation; it is read and carried, and
  currently not used to weight the fit (see *Limitations*).

A wide layout is also accepted (first column `offset_ppm`, one column per
amplitude, the header giving the amplitudes):

```
offset_ppm,0.5,0.9,1.6
-6.00,0.9812,0.9703,0.9520
```

The grid must be complete: every (amplitude, offset) combination needs a value,
because the residual is formed against a rectangular page set. Interpolate or
drop offsets before exporting — the reader raises rather than filling silently.

## Preparing measured data

The model assumes continuous-wave saturation of duration `Ts` and a single
homogeneous voxel. Before the fit:

1. **Normalise** each offset by the unsaturated signal (`Mz(0)`) acquired with the
   same readout, to obtain `Z`.
2. **Correct B0** (WASSR or the water shift) to the water frequency used by the
   model, or accept the residual shift as a model error — the demo's simulated
   example has 0.05 ppm for exactly that reason.
3. **B1** — either map it and rescale the amplitudes, or pass the nominal values
   and accept the bias; the model is sensitive to B1 through `w1 = 2 pi gamma B1`.
4. **Coil/transmit inhomogeneity** cancels in `Z` only if the reference image is
   acquired with the same coil combination.
5. **Average** repeats before exporting, or export one column per repeat and fit
   them separately; the residual is unweighted, so a single averaged curve is the
   closest to the model's assumptions.

`Ts` is read from the `# ts:` header; if it is absent the default is 2 s, which is
wrong for most acquisitions — set it.

## Public sources of measured CEST data

The repository does not redistribute measured data. If you need a public
Z-spectrum to test the pipeline, the following are commonly used; check each
licence before use and cite the original publication:

* **Zenodo / figshare CEST phantom and in-vivo datasets** — search "CEST
  Z-spectrum" and prefer records that ship the acquisition parameters (`B1`, `Ts`,
  offsets) alongside the spectra; those are exactly the columns the reader needs.
* **ISMRM & MRM data-sharing initiatives** — several CEST papers deposit the
  spectra behind their figures; the offset grid and `B1` levels are usually in the
  methods section.
* **Synthetic-but-physical phantoms** — when no measured spectrum is at hand,
  `experiments/make_example_data.py --kind measured` produces one with the two
  imperfections that dominate real data; state clearly in a paper that it is
  simulated.

Whatever the source, the check that matters is the same: the fitted curve must
reproduce the data to within the noise, and the parameter standard errors must be
small compared with the effect being claimed. The demo prints both.

## Limitations

* Unweighted least squares; `sd` is carried but not used.
* One voxel per file. For an ROI, loop over voxels and fit each (embarrassingly
  parallel, ~0.5 s per voxel with three starts).
* B0 and B1 are not fitted; a wrong value biases the parameters, which is visible
  as structure in the residual rather than as a large residual.
