"""ReLic: atmospheric Retrievals using spectral Light Curves."""

from .core import Priors, Relic, RelicExoIris
from .tslpf import NewTSLPF
from .white import NewWhiteLPF
from .atmosphere import (
    BaseAtmosphere,
    GuillotFastChem,
    IsothermalEqChem,
    IsothermalFreeChem,
    M09EqChem,
    M09FastChem,
    M09FastChem_clear,
    M09FastChem_SO2,
    M09FreeChem,
)
from .utils import (
    generate_covariates,
    get_maxlike_estimates,
    print_elapsed_time,
    optimize_parallelization, 
    replace_outliers,
    SpectrumDownsampler,
)
from .physics import calc_teq
from .plots import RelicVisualization

__all__ = [
    "BaseAtmosphere",
    "GuillotFastChem",
    "IsothermalEqChem",
    "IsothermalFreeChem",
    "M09EqChem",
    "M09FastChem",
    "M09FastChem_clear",
    "M09FastChem_SO2",
    "M09FreeChem",
    "NewTSLPF",
    "NewWhiteLPF",
    "Priors",
    "Relic",
    "RelicExoIris",
    "RelicVisualization",
    "SpectrumDownsampler",
    "calc_teq",
    "generate_covariates",
    "get_maxlike_estimates",
    "optimize_parallelization",
    "print_elapsed_time",
    "replace_outliers",
]
