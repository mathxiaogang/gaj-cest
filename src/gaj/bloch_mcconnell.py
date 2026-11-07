"""Multi-pool Bloch-McConnell model of CEST, in the form required by GAJ.

The state holds the x, y and z magnetisation of the water pool and of ``n``
solute pools, ``m = 3 + 3n``.  In the frame rotating at the saturation
frequency, with ``w1 = 2 pi gamma B1`` and ``dw_j = 2 pi gamma B0 (delta_j - Delta)``,

    dMx_j/dt = -R2_j Mx_j - dw_j My_j
    dMy_j/dt = +dw_j Mx_j - R2_j My_j - w1 Mz_j
    dMz_j/dt = +w1 My_j - R1_j (Mz_j - M0_j) + exchange

and longitudinal exchange between water and solute pool ``i`` with the
solute-to-water rate ``ksw_i`` and the reverse rate ``kws_i = ksw_i M0s_i / M0w``.

The fitted parameter vector is ``x = [T1w, T2w, T2s_1..n, ksw_1..n, M0s_1..n]``
(``p = 2 + 3n``); the solute longitudinal relaxation time is a fixed constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import Derivatives, PageSet, System

GAMMA_HZ_PER_T = 42.576e6          # gyromagnetic ratio of 1H, Hz/T


def ppm_to_rad_s(ppm: np.ndarray, b0_t: float) -> np.ndarray:
    """Angular frequency offset for a shift in ppm."""
    return 2.0 * np.pi * GAMMA_HZ_PER_T * b0_t * np.asarray(ppm, float) * 1e-6


@dataclass
class FivePoolBM:
    """Five-pool Bloch-McConnell phantom of Section 5.1 of the paper."""

    pages: PageSet
    pool_ppm: np.ndarray = field(
        default_factory=lambda: np.array([3.5, -3.5, 2.0, 0.0, -1.6]))
    b0_t: float = 3.0
    t1s: float = 1.0               # fixed solute longitudinal relaxation time
    m0w: float = 1.0

    # ---- problem size ---------------------------------------------------
    @property
    def n_pools(self) -> int:
        return int(self.pool_ppm.size)

    @property
    def m(self) -> int:
        return 3 + 3 * self.n_pools

    @property
    def p(self) -> int:
        return 2 + 3 * self.n_pools

    @property
    def water_z(self) -> int:
        return 2

    def solute_z(self, i: int) -> int:
        return 3 * (i + 1) + 2

    # ---- reference parameter vector (Table B.1) -------------------------
    def truth(self) -> np.ndarray:
        t2s = np.array([1.0e-3, 1.0e-3, 5.0e-4, 1.5e-5, 1.5e-3])
        ksw = np.array([30.0, 10.0, 500.0, 50.0, 12.0])
        m0s = np.array([8.0e-3, 0.030, 6.0e-3, 0.10, 4.0e-3])
        return np.concatenate(([2.0, 0.040], t2s, ksw, m0s))

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.concatenate((
            [1.0, 0.025],
            [5e-4, 5e-4, 1e-4, 1e-6, 5e-4],
            [5.0, 1e-9, 400.0, 1e-9, 1e-9],
            [1e-4, 2e-3, 5e-4, 0.03, 1e-3],
        ))
        hi = np.concatenate((
            [3.0, 0.090],
            [3e-3, 3e-3, 2e-3, 8e-5, 5e-3],
            [200.0, 30.0, 700.0, 180.0, 35.0],
            [0.025, 0.08, 0.04, 0.22, 0.01],
        ))
        return lo, hi

    def parameter_names(self) -> list[str]:
        n = self.n_pools
        return (["T1w", "T2w"]
                + [f"T2s{i + 1}" for i in range(n)]
                + [f"ksw{i + 1}" for i in range(n)]
                + [f"M0s{i + 1}" for i in range(n)])

    def unpack(self, x: np.ndarray) -> dict[str, np.ndarray]:
        n = self.n_pools
        x = np.asarray(x, float)
        return {
            "t1w": x[..., 0],
            "t2w": x[..., 1],
            "t2s": x[..., 2:2 + n],
            "ksw": x[..., 2 + n:2 + 2 * n],
            "m0s": x[..., 2 + 2 * n:2 + 3 * n],
        }

    def monitored(self, x: np.ndarray) -> np.ndarray:
        """The 22 monitored quantities: 17 parameters and 5 products ksw*M0s."""
        x = np.asarray(x, float)
        n = self.n_pools
        ksw = x[..., 2 + n:2 + 2 * n]
        m0s = x[..., 2 + 2 * n:2 + 3 * n]
        return np.concatenate((x, ksw * m0s), axis=-1)

    # ---- matrix assembly -------------------------------------------------
    def _assemble(self, X: np.ndarray) -> System:
        """Build ``A``, ``c`` and ``M0`` for a stack of parameter vectors.

        ``X`` has shape ``(n_x, p)``; the result is ``(n_x, n_pages, ...)``.
        """
        X = np.atleast_2d(np.asarray(X, float))
        nx = X.shape[0]
        kf = self.pages.n_pages
        m, n = self.m, self.n_pools
        u = self.unpack(X)

        r1s = 1.0 / self.t1s
        r1w = 1.0 / u["t1w"]                      # (nx,)
        r2w = 1.0 / u["t2w"]                      # (nx,)
        r2s = 1.0 / u["t2s"]                      # (nx, n)

        A = np.zeros((nx, kf, m, m))
        c = np.zeros((nx, kf, m))
        M0 = np.zeros((nx, kf, m))

        b1, off = self.pages.broadcast()                  # (kf,)
        w1 = 2.0 * np.pi * GAMMA_HZ_PER_T * b1 * 1e-6     # rad/s
        dw = np.empty((n + 1, kf))
        dw[0] = -ppm_to_rad_s(off, self.b0_t)             # water in the RF frame
        for j in range(n):
            dw[j + 1] = ppm_to_rad_s(self.pool_ppm[j] - off, self.b0_t)

        bases = [0] + [3 * (i + 1) for i in range(n)]
        for j, base in enumerate(bases):
            r2j = r2w if j == 0 else r2s[:, j - 1]        # (nx,)
            r1j = r1w if j == 0 else np.full(nx, r1s)
            A[:, :, base + 0, base + 0] = -r2j[:, None]
            A[:, :, base + 1, base + 1] = -r2j[:, None]
            A[:, :, base + 0, base + 1] = -dw[j][None, :]
            A[:, :, base + 1, base + 0] = +dw[j][None, :]
            A[:, :, base + 1, base + 2] = -w1[None, :]
            A[:, :, base + 2, base + 1] = +w1[None, :]
            A[:, :, base + 2, base + 2] = -r1j[:, None]

        # longitudinal exchange, water <-> solute i
        ksw = u["ksw"]                                    # (nx, n)
        m0s = u["m0s"]
        for i in range(n):
            kws = ksw[:, i] * m0s[:, i] / self.m0w        # (nx,)
            zs = self.solute_z(i)
            A[:, :, self.water_z, self.water_z] -= kws[:, None]
            A[:, :, self.water_z, zs] += ksw[:, i][:, None]
            A[:, :, zs, zs] -= ksw[:, i][:, None]
            A[:, :, zs, self.water_z] += kws[:, None]

        # source term and initial state
        c[:, :, self.water_z] = r1w[:, None] * self.m0w
        M0[:, :, self.water_z] = self.m0w
        for i in range(n):
            zs = self.solute_z(i)
            c[:, :, zs] = r1s * m0s[:, i][:, None]
            M0[:, :, zs] = m0s[:, i][:, None]

        o = np.zeros(m)
        o[self.water_z] = 1.0
        return System(A, c, M0, o)

    def build(self, x: np.ndarray) -> System:
        x = np.asarray(x, float)
        if x.ndim != 1:
            raise ValueError("build() expects a single parameter vector")
        sys_batch = self._assemble(x)
        return System(sys_batch.A[0], sys_batch.c[0], sys_batch.M0[0], sys_batch.o)

    def build_many(self, X: np.ndarray) -> System:
        return self._assemble(np.atleast_2d(np.asarray(X, float)))

    # ---- derivatives -----------------------------------------------------
    def derivatives(self, x: np.ndarray) -> Derivatives:
        """Closed-form ``dA/dx_i``, ``dc/dx_i``, ``dM0/dx_i`` (page independent)."""
        x = np.asarray(x, float)
        if x.ndim != 1:
            raise ValueError("derivatives() expects a single parameter vector")
        u = self.unpack(x)
        m, n, p = self.m, self.n_pools, self.p
        dA = np.zeros((p, m, m))
        dc = np.zeros((p, m))
        dM0 = np.zeros((p, m))
        zw = self.water_z

        # T1w  (R1w = 1/T1w appears in A and in c)
        dA[0, zw, zw] += 1.0 / u["t1w"] ** 2
        dc[0, zw] += -self.m0w / u["t1w"] ** 2
        # T2w
        dA[1, 0, 0] += 1.0 / u["t2w"] ** 2
        dA[1, 1, 1] += 1.0 / u["t2w"] ** 2

        for i in range(n):
            base = 3 * (i + 1)
            zs = self.solute_z(i)
            # T2s_i
            k = 2 + i
            dA[k, base, base] += 1.0 / u["t2s"][i] ** 2
            dA[k, base + 1, base + 1] += 1.0 / u["t2s"][i] ** 2
            # ksw_i
            k = 2 + n + i
            dkws = u["m0s"][i] / self.m0w
            dA[k, zw, zw] -= dkws
            dA[k, zw, zs] += 1.0
            dA[k, zs, zs] -= 1.0
            dA[k, zs, zw] += dkws
            # M0s_i  (enters c, the reverse rate and the initial state)
            k = 2 + 2 * n + i
            dc[k, zs] += 1.0 / self.t1s
            dkws = u["ksw"][i] / self.m0w
            dA[k, zw, zw] -= dkws
            dA[k, zs, zw] += dkws
            dM0[k, zs] += 1.0

        sparse = []
        for k in range(p):
            rows, cols = np.nonzero(dA[k])
            sparse.append((rows, cols, dA[k][rows, cols]))
        return Derivatives(dA=dA, dc=dc, dM0=dM0, dA_sparse=sparse)
