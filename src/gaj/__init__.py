"""GAJ: a gated analytic Jacobian for linear drive-and-observe models.

Reference implementation of

    G. Xiao, "GAJ: A Gated Analytic Jacobian for Fast Fitting of Linear
    Dynamical Models with Time-Scale Separation, with Application to
    Multi-Pool CEST".

The package is organised as the paper is: ``model`` (the linear drive-and-observe
class), ``bloch_mcconnell`` (the example), ``eigen`` (Procedure 1), ``surrogate``
(Section 4.2), ``jacobian`` (Sections 4.3-4.5) and ``fit`` (the solver wrapper).
"""

from .model import Derivatives, LinearDriveObserveModel, PageSet, System
from .bloch_mcconnell import FivePoolBM
from .data import Spectrum, load_zspectrum, make_starts, save_zspectrum, simulate
from .eigen import (SlowModes, exact_slow_modes, power_iteration_slow_modes,
                    spectral_diagnostics)
from .fit import FitResult, fit, fit_multi_start
from .jacobian import (jacobian_central_exact, jacobian_central_surrogate, jacobian_gaj,
                       jacobian_surrogate_direct, jacobian_vanloan)
from .metrics import meets_requirement, relative_errors, spectral_error, wilson_interval
from .surrogate import (GAJConfig, exact_response, forward, surrogate_response,
                        truncation_error)

__all__ = [
    "PageSet", "System", "Derivatives", "LinearDriveObserveModel", "FivePoolBM",
    "Spectrum", "load_zspectrum", "save_zspectrum", "simulate", "make_starts",
    "SlowModes", "power_iteration_slow_modes", "exact_slow_modes", "spectral_diagnostics",
    "GAJConfig", "forward", "surrogate_response", "exact_response", "truncation_error",
    "jacobian_gaj", "jacobian_surrogate_direct", "jacobian_central_surrogate",
    "jacobian_central_exact", "jacobian_vanloan",
    "fit", "fit_multi_start", "FitResult",
    "relative_errors", "spectral_error", "meets_requirement", "wilson_interval",
]

__version__ = "0.1.0"
