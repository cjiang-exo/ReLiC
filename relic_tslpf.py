from exoiris.tslpf import TSLPF, lnlike_normal, nlstsq
from exoiris import TSDataGroup 
from exoiris.lmlikelihood import marginalized_loglike_mbl2d
from numpy import isfinite, nan, ndarray, inf, arctan2, dstack, zeros_like, ones_like, empty
from numpy.linalg import LinAlgError 
from petitRADTRANS.physics import rebin_spectrum_bin
from pytransit.orbits import as_from_rhop, i_from_ba
from pytransit.param import ParameterSet, UniformPrior as UP, NormalPrior as NP, GParameter   
from typing import Literal, Tuple, Optional, Union

from relic_atmosphere import BaseAtmosphere
 
NM_WHITE_MARGINALIZED = 0
NM_GP_FIXED = 1
NM_GP_FREE = 2
NM_WHITE_PROFILED = 3

class NewTSLPF(TSLPF):

    def __init__(self, runner, name: str, ldmodel, data: TSDataGroup, 
        atmos_model: BaseAtmosphere, tmpars = None, circular_orbit: bool = True, 
        noise_model: Literal["white_profiled", "white_marginalized", 
        "fixed_gp", "free_gp"] = 'white_profiled'):

        self.circular_orbit = circular_orbit
        super().__init__(runner, name, ldmodel, data, tmpars=tmpars, noise_model=noise_model)

        # self.radius_ratios  = [zeros_like(w) for w in self.wavelengths]
        # self.fluxes         = [zeros_like(d.fluxes) for d in self.data]
        self.atmos_model    = atmos_model
        self.model_wl       = atmos_model.wavelengths
        self.bin_widths     = self._get_binwidths(data)
        # self.transpec       = zeros_like(self.model_wl)
        # self._baseline_models = [ones_like(d.fluxes) for d in self.data] 

    def _get_binwidths(self, data: TSDataGroup) -> list[ndarray]:       
        bin_widths = [d._wl_r_edges - d._wl_l_edges for d in data]
        return bin_widths
    
    def _init_parameters(self) -> None:

        self.ps = ParameterSet([])
        self._init_p_star()
        self._init_p_orbit()
        self._init_p_transit_centers()
        self._init_p_limb_darkening() 
        self._init_p_noise()
        if self._nm == NM_GP_FREE:
            self._init_p_gp()
        self._init_p_bias()
        self.ps.freeze() 

    def _init_p_orbit(self) -> None: 
        pp = [
            GParameter('p', 'period', 'd', NP(1.0, 1e-5), (0, inf)),
            GParameter('b', 'impact_parameter', 'R_s', UP(0.0, 1.0), (0, inf)),
        ]
        if not self.circular_orbit:
            pp += [ 
                GParameter('secw', 'sqrt(e) cos(w)', '', NP(0.0, 1e-5), (-1, 1)),
                GParameter('sesw', 'sqrt(e) sin(w)', '', NP(0.0, 1e-5), (-1, 1))
            ]
        self.ps.add_global_block('orbit', pp)
        self._start_orbit = self.ps.blocks[-1].start
        self._sl_orbit = self.ps.blocks[-1].slice

    def flux_model(self, pv: ndarray, include_baseline: bool = True):   
        transit_model = self.transit_model(pv)
        if self.spot_model is not None:
            self.spot_model.apply_spots(pv, transit_model)
            if self.spot_model.include_tlse:
                self.spot_model.apply_tlse(pv, transit_model)
        if include_baseline:
            _baseline_models = self.baseline_model(transit_model)
            for i in range(self.data.size):
                transit_model[i][:] *= _baseline_models[i]
        return transit_model

    def transit_model(self, pv: ndarray, copy=True) -> list[ndarray]:
        """Evaluates the transit model for parameter vector pv.

        Parameters
        ----------
        pv : numpy.ndarray
            Check ExoIris.ps for the order of parameters in the vector.

        Returns
        -------
        2D fluxes for each dataset: list[ndarray]
        """  
        transpec = self.atmos_model(pv)
        k = [ rebin_spectrum_bin(self.model_wl, transpec, data_wl,
            bin_widths=self.bin_widths[i])**0.5
            for i, data_wl in enumerate(self.wavelengths)
        ]

        if self.circular_orbit:
            p, b = pv[self._sl_orbit]
            ecc  = 0
            w    = 0
        else:
            p, b, secw, sesw = pv[self._sl_orbit]
            ecc = secw ** 2 + sesw ** 2
            w = arctan2(sesw, secw)

        aor = as_from_rhop(pv[0], p)
        inc = i_from_ba(b, aor) 
        t0s = pv[self._sl_tcs]

        ldp, istar = self.ldmodel(self.tms[0].mu, pv[self._sl_ld])
        ldpi = dstack([ldp, istar])

        fluxes = [ 
            tm.evaluate(k[i], ldpi[:, self.ldmodel.wlslices[i], :],
            t0s[self.data.epoch_groups[i]], p, aor, inc, ecc, w, copy)[0] 
            for i, tm in enumerate(self.tms)
        ]

        return fluxes

    def baseline_model(self, mtransit): 
        _baseline_models = [empty(m.shape) for m in mtransit]
        for i, d in enumerate(self.data): 
            res = d.fluxes / mtransit[i]
            try:
                coeffs = nlstsq(d.covs, res, d.mask, d._wlmask, d._wls_with_nan)
                _baseline_models[i][:] = (d.covs @ coeffs).T
            except LinAlgError:
                _baseline_models[i][:] = nan
        return _baseline_models

    def lnposterior(self, pv: ndarray) -> float:
        lnp = self.lnprior(pv)
        return -inf if not isfinite(lnp) else lnp + self.lnlikelihood(pv) 
    
    def lnprior(self, pv) -> float:
        """ 1D input only, no vectorization. """ 
        if any(pv <= self.ps.lbounds) | any(pv >= self.ps.ubounds):
            return -inf
        
        lnp = sum([p.lnprior(pv[i]) for i, p in enumerate(self.ps)])
        return lnp + self.additional_priors(pv)

    def additional_priors(self, pv) -> float:
        """ Used for additional constraints such as pv[0] > pv[1] """
        return 0

    def lnlikelihood(self, pv: ndarray) -> float:
        """ 1D input only, no vectorization. """ 
        wn_multipliers = pv[self._sl_wnm]
        lnl = 0
        if self._nm == NM_WHITE_MARGINALIZED:
            fmod = self.flux_model(pv, include_baseline=False) 
            try:
                for i, d in enumerate(self.data):
                    lnl += marginalized_loglike_mbl2d(d.fluxes, fmod[i], d.errors*wn_multipliers[d.noise_group], d.covs, d.mask)
            except LinAlgError:
                lnl = -inf
        elif self._nm == NM_WHITE_PROFILED:
            fmod = self.flux_model(pv, include_baseline=True)
            for i, d in enumerate(self.data):
                lnl += lnlike_normal(d.fluxes, fmod[i], d.errors, wn_multipliers[d.noise_group], d.mask)
        else: # GP
            fmod = self.flux_model(pv) 
            if self._nm == NM_GP_FREE:
                self.set_gp_hyperparameters(*pv[self._sl_gp])
            for i in range(self.data.size):
                lnl += self._gp[i].log_likelihood(self._gp_flux[i] - fmod[i][self.data[i].mask])
        return lnl 

    def lnlikelihood_ns(self, pv: ndarray) -> float:
        if any(pv <= self.ps.lbounds) | any(pv >= self.ps.ubounds):
            return -inf
        lnp = self.additional_priors(pv)
        return -inf if not isfinite(lnp) else self.lnlikelihood(pv) 
    
