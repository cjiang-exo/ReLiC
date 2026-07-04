"""ReLic: atmospheric Retrievals using spectral Light Curves."""

from .core import Relic, RelicExoIris
from .tslpf import NewTSLPF
from .white import NewWhiteLPF
from .atmosphere import (
    BaseAtmosphere,
    IsothermalEqChem,
    IsothermalFreeChem,
    TP6EqChem,
    TP6FreeChem,
    TP6FastChem,
    TP6FastChem_SO2, 
    tp6madhu,
)
from .utils import (
    generate_covariates,
    get_maxlike_estimates,
    print_elapsed_time,
    optimize_parallelization,
    print_info,
    replace_outliers,
    SpectrumDownsampler,
)
from .physics import calc_teq
from .plots import RelicVisualization

__all__ = [
    "Relic",
    "RelicExoIris",
    "NewTSLPF",
    "NewWhiteLPF",
    "BaseAtmosphere",
    "IsothermalEqChem",
    "IsothermalFreeChem",
    "TP6EqChem",
    "TP6FreeChem",
    "TP6FastChem",
    "TP6FastChem_SO2",
    "GuillotEQChem",
    "tp6madhu",
    "generate_covariates",
    "get_maxlike_estimates",
    "print_elapsed_time",
    "optimize_parallelization",
    "print_info",
    "replace_outliers",
    "SpectrumDownsampler",
    "calc_teq",
    "RelicVisualization",
]
