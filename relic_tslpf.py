from exoiris.tslpf import TSLPF, lnlike_normal, nlstsq
from exoiris import TSDataGroup 
from exoiris.lmlikelihood import marginalized_loglike_mbl2d
from numpy import isfinite, nan, ndarray, inf, arctan2, dstack, zeros_like, ones_like, empty, newaxis, tile, arange
from celerite2 import GaussianProcess as GP
from celerite2.terms import Term, Matern32Term
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
NOISE_MODELS = dict(white=NM_WHITE_PROFILED, white_profiled=NM_WHITE_PROFILED, white_marginalized=NM_WHITE_MARGINALIZED, fixed_gp=NM_GP_FIXED, free_gp=NM_GP_FREE)

class NewTSLPF(TSLPF):

    def __init__(self, runner, name: str, ldmodel, data: TSDataGroup, 
        atmos_model: BaseAtmosphere, tmpars = None, circular_orbit: bool = True, 
        noise_model: Literal["white_profiled", "white_marginalized", 
        "fixed_gp", "free_gp"] = 'white_profiled'):

        self.circular_orbit = circular_orbit
        super().__init__(runner, name, ldmodel, data, tmpars=tmpars, noise_model=noise_model)
 
        self.atmos_model    = atmos_model
        self.model_wl       = atmos_model.wavelengths
        self.bin_widths     = self._get_binwidths(data) 

    def _get_binwidths(self, data: TSDataGroup) -> list[ndarray]:       
        bin_widths = [d._wl_r_edges - d._wl_l_edges for d in data]
        return bin_widths

    def set_noise_model(self, noise_model: str) -> None:
        """Sets the noise model for the analysis.

        Parameters
        ----------
        noise_model : str
            The noise model to be used. Must be one of the following: white_profiled, white_marginalized, fixed_gp, free_gp.

        Raises
        ------
        ValueError
            If noise_model is not one of the specified options.
        """
        if noise_model not in NOISE_MODELS.keys():
            raise ValueError('noise_model must be one of: white_profiled, white_marginalized, fixed_gp, free_gp')
        self.noise_model = noise_model
        self._nm = NOISE_MODELS[noise_model]
        if self._nm in (NM_GP_FIXED, NM_GP_FREE):
            self._init_gp() 

    def _init_gp(self) -> None:
        """Initializes the Gaussian Process (GP) .

        This method initializes the necessary variables and sets up the GP for the given data.
        """
        self._gp_time = []
        self._gp_flux = []
        # self._gp_ferr = []
        self._gp_ferr2 = []
        self._gp = []
        for d in self.data:
            self._gp_time.append((tile(d.time[newaxis, :], (d.nwl, 1)) + arange(d.nwl)[:, newaxis])[d.mask])
            self._gp_flux.append(d.fluxes[d.mask])
            self._gp_ferr2.append(d.errors[d.mask]**2)
            self._gp.append(GP(Matern32Term(sigma=self._gp_flux[-1].std(), rho=0.1)))
            self._gp[-1].compute(self._gp_time[-1], diag=self._gp_ferr2[-1], quiet=True)

    def set_gp_hyperparameters(self, sigma: float, rho: float, jitter: float = 0, idata: int | None = None) -> None:
        """Sets the Gaussian Process hyperparameters assuming a Matern32 kernel.

        Parameters
        ----------
        sigma
            The kernel amplitude parameter.
        rho
            The length scale parameter.
        jitter
            A jitter term to be added to the diagonal of the covariance matrix.
        idata
            The data set for which to set the hyperparameters. If None, the hyperparameters are set for all data sets.
        """
        if self._gp is None:
            raise RuntimeError('The GP needs to be initialized before setting hyperparameters.')

        for i in ([idata] if idata is not None else range(self.data.size)):
            self._gp[i].kernel = Matern32Term(sigma=sigma, rho=rho)
            self._gp[i].compute(self._gp_time[i], diag=self._gp_ferr2[i] + jitter**2, quiet=True)

    def set_gp_kernel(self, kernel: Term, jitter: float = 0) -> None:
        """Sets the kernel for the Gaussian Process (GP) model and recomputes the GP.

        Parameters
        ----------
        kernel : Term
            The kernel to be set for the GP.
        """
        for i, gp in enumerate(self._gp):
            gp.kernel = kernel
            gp.compute(self._gp_time[i], diag=self._gp_ferr2[i] + jitter**2, quiet=True)

    def _init_parameters(self) -> None:
        self.ps = ParameterSet([])
        self._init_p_star()
        self._init_p_orbit()
        self._init_p_transit_centers()
        self._init_p_limb_darkening() 
        if self._nm in (NM_WHITE_MARGINALIZED, NM_WHITE_PROFILED):
            self._init_p_noise()
        elif self._nm == NM_GP_FREE:
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

    def _init_p_gp(self):
        ps = self.ps
        if not hasattr(self, '_sl_gp'):
            pp = []
            for i in range(self.data.n_noise_groups):
                pp += [
                    GParameter(f'gp_lg_sigma_{i}', f'GP log10 sigma of noise group {i}', 
                        'log10', NP(0.0, 0.01), (-inf, inf)),
                    GParameter(f'gp_lg_rho_{i}', f'GP log10 rho of noise group {i}', 
                        'log10', NP(0.0, 0.01), (-inf, inf)),
                    GParameter(f'gp_lg_jitter_{i}', f'GP log10 jitter of noise group {i}', 
                        'log10', NP(0.0, 0.01), (-inf, inf))
                ]
            ps.add_global_block('gp_hyperparameters', pp)
            self._start_gp = ps.blocks[-1].start
            self._sl_gp = ps.blocks[-1].slice

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
        lnl = 0
        if self._nm == NM_WHITE_MARGINALIZED:
            wn_multipliers = pv[self._sl_wnm]
            fmod = self.flux_model(pv, include_baseline=False) 
            try:
                for i, d in enumerate(self.data):
                    lnl += marginalized_loglike_mbl2d(d.fluxes, fmod[i], d.errors*wn_multipliers[d.noise_group], d.covs, d.mask)
            except LinAlgError:
                lnl = -inf
        elif self._nm == NM_WHITE_PROFILED:
            wn_multipliers = pv[self._sl_wnm]
            fmod = self.flux_model(pv, include_baseline=True)
            for i, d in enumerate(self.data):
                lnl += lnlike_normal(d.fluxes, fmod[i], d.errors, wn_multipliers[d.noise_group], d.mask)
        else: # GP
            fmod = self.flux_model(pv, include_baseline=False) 
            if self._nm == NM_GP_FREE: 
                gp_pv = 10**pv[self._sl_gp]
                for n in self.data.noise_groups: 
                    self.set_gp_hyperparameters(*gp_pv[n*3:(n+1)*3], idata=n)
            for i in range(self.data.size):
                lnl += self._gp[i].log_likelihood(self._gp_flux[i] - fmod[i][self.data[i].mask])
        return lnl 

    def lnlikelihood_ns(self, pv: ndarray) -> float:
        if any(pv <= self.ps.lbounds) | any(pv >= self.ps.ubounds):
            return -inf
        lnp = self.additional_priors(pv)
        return -inf if not isfinite(lnp) else self.lnlikelihood(pv) 
    
