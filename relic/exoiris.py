from multiprocessing import Pool
from exoiris import ExoIris, TSData, TSDataGroup
from numpy import array, floor, isfinite, unique, ndarray, squeeze
from typing import Callable, Literal, Tuple, Optional, Union 
from .tslpf import NewTSLPF
from .white import NewWhiteLPF 
from .atmosphere import BaseAtmosphere

class ReLicExoIris(ExoIris):
    def __init__(self, name: str, ldmodel, data: TSDataGroup | TSData,
            atmos_model: BaseAtmosphere, tmpars: dict | None = None,
            circular_orbit: bool = True, noise_model: Literal["white_profiled", 
            "white_marginalized", "fixed_gp", "free_gp"] = 'white_profiled',  
        ):
        
        data = TSDataGroup([data]) if isinstance(data, TSData) else data

        for d in data:
            if any(~isfinite(d.fluxes[d.mask])):
                raise ValueError(f"The {d.name} data set flux array contains unmasked nonfinite values.")

            if any(~isfinite(d.errors[d.mask])):
                raise ValueError(f"The {d.name} data set error array contains unmasked nonfinite values.")

        ngs = array(data.noise_groups)
        if not ((ngs.min() == 0) and (ngs.max() + 1 == unique(ngs).size)):
            raise ValueError("The noise groups must start from 0 and be consecutive.")

        ogs = array(data.offset_groups)
        if not ((ogs.min() == 0) and (ogs.max() + 1 == unique(ogs).size)):
            raise ValueError("The offset groups must start from 0 and be consecutive.")

        egs = array(data.epoch_groups)
        if not ((egs.min() == 0) and (egs.max() + 1 == unique(egs).size)):
            raise ValueError("The epoch groups must start from 0 and be consecutive.")
 
        self._tsa = NewTSLPF(self, name, ldmodel, data, atmos_model=atmos_model, 
            tmpars=tmpars, noise_model=noise_model, circular_orbit=circular_orbit)
        self._wa: None | NewWhiteLPF = None

        self.nthreads: int = 1

        self.period: float | None = None
        self.zero_epoch: float | None = None
        self.transit_duration: float | None= None
        self._tref = floor(self.data.tmin)

        self._white_times: None | list[ndarray] = None
        self._white_fluxes: None | list[ndarray] = None
        self._white_errors: None | list[ndarray] = None
        self._white_models: None | list[ndarray] = None
        self.white_gp_models: None | list[ndarray] = None

 