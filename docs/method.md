# Equation → code map

Every numbered equation of the paper has one place in the code. The list is
meant to be read with the paper open.

| Paper | Meaning | Code |
|---|---|---|
| (1) `dM/dt = A M + c`, `M(0)=M0` | drive phase | `bloch_mcconnell.FivePoolBM.build` |
| (2) `M(Ts) = Mss + e^{A Ts} e0` | exact response | `surrogate.exact_response` |
| (3) `e_spec` | spectral error | `metrics.spectral_error` |
| (4) `Z = o^T Mss + sum_j beta_j alpha_j e^{lam_j Ts}` | full spectral form | `surrogate.forward` |
| (5) `Z_S` | surrogate (truncated to S) | `surrogate.forward` |
| (6) `|delta| <= ...` | truncation bound, H2 | `surrogate.truncation_error` (measured, not bounded) |
| (7) `lambda*` | guard threshold implied by the guard | `cli.cmd_info` (reported) |
| (8) `dMss = -B(dA Mss + dc)` | steady-state term | `jacobian._base_terms` |
| (9)-(10) divided differences `Phi_rc` | `T_SS` | `jacobian._divided_differences` |
| (11)-(12) exponential and initial-state terms | `T_e0` | `jacobian._base_terms` |
| (13) `T_SS` | retained–retained block | `jacobian._base_terms` |
| (14) `T_SF` | coupling through the reduced resolvent | `jacobian.jacobian_surrogate_direct` (direct) |
| (15) Neumann series of the resolvent | series | `jacobian._series_terms` |
| (16) `T_q` | q-th series term | `jacobian._series_terms` |
| (17) `eta_q` | gate statistic | `jacobian.jacobian_gaj` |
| (18) remainder heuristic | discussed in `docs/reproducibility.md` | — |
| (19) central differences of `Z_S` | fallback | `jacobian.jacobian_central_surrogate` |
| Procedure 1 | slow eigenpairs | `eigen.power_iteration_slow_modes` |
| Procedure 2 | residual + gated Jacobian | `jacobian.jacobian_gaj` |
| Table 2 | settings | `surrogate.GAJConfig` |
| Table 3 | configurations | `fit.JACOBIANS` |
| Table B.1 | phantom and bounds | `bloch_mcconnell.FivePoolBM.truth/bounds` |

## The model

`FivePoolBM` holds the page set and builds, for a stack of parameter vectors,

```
A_j = [[ -R2_j, -dw_j,     0  ],
       [ +dw_j, -R2_j,  -w1  ],
       [   0,   +w1,   -R1_j]]        plus longitudinal exchange on the z entries
c[z_j] = R1_j M0_j,        M0[z_w] = 1,  M0[z_i] = M0s_i
```

with `w1 = 2 pi gamma B1`, `dw_water = -2 pi gamma B0 Delta`,
`dw_i = 2 pi gamma B0 (delta_i - Delta)`, `ksw_i` from solute to water and
`kws_i = ksw_i M0s_i / M0w` back. The 17 fitted parameters are
`[T1w, T2w, T2s_1..5, ksw_1..5, M0s_1..5]`; `T1s = 1 s` is fixed (the paper states
that the solute longitudinal rate is fixed but does not tabulate it).

The derivatives `dA`, `dc`, `dM0` are closed form and **page independent** (M3 in
its strongest form), so they are built once per Jacobian and shared by all 147
pages; `tests/test_bloch_mcconnell.py` checks them against central differences
and across pages.

## The two numerical deviations from the paper's pseudocode

### 1. The `u_q` recursion

The paper writes

```
u_q = B u_{q-1},   u_0 = e0,   u_q := u_q - sum_s alpha_s lambda_s^{-q} v_s
```

which is `B^q Q_F e0`. The subtraction removes an O(1) slow component from a
vector whose useful part is `O(lambda_f^{-q})`; with `lambda_s / lambda_f ~ 0.05`
the rounding error is amplified by `(lambda_s/lambda_f)^{-q}`, and the term norms
stop decreasing at `q = 6` with `eta = 3e-8 > tau`. The implementation starts in
the discarded subspace instead,

```
u_0 = Q_F e0 = e0 - sum_s alpha_s v_s,   u_q = B u_{q-1} - sum_s (w_s^T u_q) v_s
```

The second term projects out only the *computed* leakage (analytically zero), so
no O(1) quantity is ever subtracted. The two forms are algebraically identical
(`B^q Q_F e0` both ways); the second one reaches `eta = 7e-10` at `q = 6`. The
same substitution is made for `c_q = o^T B^q Q_F`.

Effect on the Jacobian: none beyond truncation. `tests/test_jacobian.py` compares
the result with directional finite differences of the surrogate (`< 1e-4`) and with
the direct reduced resolvent on a consistent eigenbasis (`< 1e-6`).

### 2. `expm` for the reference forward model

`scipy.linalg.expm` loops in Python and needs ~3 ms per 18x18 generator whose norm
is `1e5` (the Bloch–McConnell matrices are stiff), so 147 pages cost ~0.4 s and a
central-difference Jacobian of the exact model costs 34 of those. `linalg.expm_batch`
is the same scaling-and-squaring algorithm vectorised over the page axis: 7.4 ms
for the whole grid, agreeing with SciPy to `3e-12`. It is used only by
`exact_response` (the reference and the Trad-C control); the surrogate path is
unchanged.

## What is *not* implemented

* **Complex retained pairs.** Proposition 1 assumes simple real modes. With
  `eig="exact"` the eigenpairs come from LAPACK and may be complex; the analytic
  Jacobian then raises rather than silently returning a wrong matrix. On the
  paper's phantom the two slowest modes are real on every page.
* **The (F,F) block.** Neglected, as in the paper: it is `O(e^{-d_F Ts})`.
* **Weights, joint B0/B1 estimation, and multi-voxel ROIs.** The reader takes one
  voxel's Z-spectrum; fitting B0/B1 as nuisance parameters is a modelling choice
  left to the user.
