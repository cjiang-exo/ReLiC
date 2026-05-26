import shutil
import h5py
import numpy as np 
import os
import pickle
from exoiris.ldtkld import LDTkLD
from exoiris import TSData, TSDataGroup
from functools import reduce

from numpy import array, squeeze, ndarray, asarray, isfinite, zeros, empty_like 
from pytransit.param import UniformPrior as UP, NormalPrior as NP, GParameter   
from typing import Callable, Literal, Tuple, Optional, Union
from relic_atmosphere import BaseAtmosphere
from relic_exoiris import ReLicExoIris 
from relic_white import NewWhiteLPF
from multiprocessing.pool import Pool
from scipy.stats import norm, truncnorm
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty.utils import get_neff_from_logwt

NM_WHITE_MARGINALIZED = 0
NM_GP_FIXED = 1
NM_GP_FREE = 2
NM_WHITE_PROFILED = 3

""" pv should be scalar input, no more vectorization """

class ReLic:
    def __init__(self, atmos_model: BaseAtmosphere):
        if (not isinstance(atmos_model, BaseAtmosphere)) or (type(atmos_model) is BaseAtmosphere):
            raise TypeError(f"Expected the subclass of BaseAtmosphere, got {type(atmos_model).__name__}")
        if atmos_model.__class__.__call__ is BaseAtmosphere.__call__:
            raise NotImplementedError("The __call__ method must be implemented to compute a spectrum.")
        if atmos_model.wavelengths is None: 
            raise ValueError("The atmosphere model must have defined wavelengths.")

        self.cfg         = atmos_model.cfg
        self.atmos_model = atmos_model
        self.raw_data    = self._load_raw_data()
        self.tsdata      = self._init_TSData()
        self.ldmodel     = self._init_LDModel()
        self.exoiris     = self._init_ExoIris() 

        self._update_parameters() 
        
    def _load_raw_data(self) -> list[h5py.File]:
        filelist = self.cfg["PATH"]["input_file"]
        print("\nLoading data: ")
        [print(f"  {f}") for f in filelist]
        return [h5py.File(f, 'r') for f in filelist]
    
    def _init_TSData(self) -> TSDataGroup:
        dlist = []
        for i, rd in enumerate(self.raw_data):
            try:  # specify edges for STIS and WFC3
                wl_edges = rd['wavelength_bins'][:].T
            except KeyError:
                wl_edges = None

            _flux = rd['flux'][:]
            _flux_err = rd['flux_err'][:] 
            if _flux.shape[0] != len(rd['wavelength'][:]):
                _flux = _flux.T
                _flux_err = _flux_err.T
 
            dlist.append(TSData(
                time        = rd['bjd_tdb'][:], 
                wavelength  = rd['wavelength'][:], 
                fluxes      = _flux, 
                errors      = _flux_err, 
                wl_edges    = wl_edges, 
                name        = self.cfg['EXOIRIS']['instruments'][i] + f"_d{i}", 
                noise_group = self.cfg["EXOIRIS"]["noise_groups"][i], 
                n_baseline  = self.cfg['EXOIRIS']['n_baselines'][i],
            ))

            _wlrange = self.cfg["EXOIRIS"]["wl_range_micron"][i]
            _trange  = self.cfg["EXOIRIS"]["time_range_bjd"][i]
            if _wlrange != []:
                dlist[-1].crop_wavelength(*_wlrange)
            if _trange != []:
                dlist[-1].crop_time(*_trange)
            dlist[-1].mask_transit(
                t0  = self.cfg["PLANET"]["transit_epoch_bjd"][0], 
                p   = self.cfg["PLANET"]["transit_period_d"][0],
                t14 = self.cfg["PLANET"]["transit_duration_d"][0]
            ) 
            dlist[-1].normalize_to_poly()
            dlist[-1].mask_outliers(sigma=8.0)

            print("Loaded dataset #{0:d} with nwl={1:d}, nt={2:d}.".format(
                i, *dlist[i].fluxes.shape
            ))

            r = self.cfg["EXOIRIS"]["rebin_resolutions"][i]
            if r > 0:
                dlist[-1] = dlist[-1].bin_wavelength(r=r, estimate_errors=False)
                print(f"  Rebinned to resolution R={r}. "
                      f"New nwl={dlist[-1].fluxes.shape[0]}.")
            else:
                print("  No wavelength binning applied, using native resolution.")

        return reduce(lambda x,y: x+y, dlist)
    
    def _init_LDModel(self):
        print('\nInitializing LDTk model... It takes 1 -- 30 minutes. Be patient!')

        _t = self.cfg['STAR']['teff']
        _g = self.cfg['STAR']['logg']
        _m = self.cfg['STAR']['metal']
        
        return LDTkLD(
            data    = self.tsdata, 
            teff    = (_t[0], max(_t[1], 50)),
            logg    = (_g[0], max(_g[1], 0.05)), 
            metal   = (_m[0], max(_m[1], 0.05)),
            dataset = 'visir'
        )
    
    def _init_ExoIris(self):
        print("\nInitializing ExoIris model...")

        return ReLicExoIris( 
            name           = self.cfg["PLANET"]["name"], 
            ldmodel        = self.ldmodel, 
            data           = self.tsdata, 
            atmos_model    = self.atmos_model,  
            circular_orbit = self.cfg["PLANET"]["circular_orbit"],
            noise_model    = self.cfg["EXOIRIS"]["noise_model"],  
        )
    
    def _update_parameters(self, ):

        print("Updating parameters and priors...")
        self.exoiris.ps.thaw()

        """ Update the bounds of stellar parameters based on LDTk model """
        _tid = self.exoiris.ps.find_pid("teff")
        _gid = self.exoiris.ps.find_pid("logg")
        _mid = self.exoiris.ps.find_pid("metal")
        self.exoiris.ps[_tid].bounds = self.ldmodel.sc.client.teffl
        self.exoiris.ps[_gid].bounds = self.ldmodel.sc.client.loggl
        self.exoiris.ps[_mid].bounds = self.ldmodel.sc.client.zl

        """ Update atmospheric parameters """
        pmap = {"UP": UP, "NP": NP}
        atm_ps = [GParameter(k, v[5], v[6], pmap[v[0]](v[1], v[2]), v[3:5]) \
            for k, v in self.cfg["PRIORS"]["ATMOSPHERE"].items()] 
        self.exoiris.ps.add_global_block('atmosphere', atm_ps)
        self.exoiris._tsa._start_atm = self.exoiris.ps.blocks[-1].start
        self.exoiris._tsa._sl_atm = self.exoiris.ps.blocks[-1].slice
        self.atmos_model._sl_atm = self.exoiris._tsa._sl_atm

        """ Update all prior functions"""
        all_priors = self.cfg["PRIORS"]["TRANSIT"].copy()
        all_priors.update(self.cfg["PRIORS"]["STAR"])
        all_priors.update(self.cfg["PRIORS"]["NOISE"])
        all_priors.update(self.cfg["PRIORS"]["ATMOSPHERE"])
        for k, v in all_priors.items():  
            _ptype, _a, _b = v[:3]
            if (_ptype in ["UP", "NP"]) and isinstance(_a, (int, float)) \
                and isinstance(_b, (int, float)):
                self.exoiris.set_prior(k, _ptype, _a, _b)
            else:
                raise ValueError(f"Invalid prior for parameter "
                    f"'{k}': {_ptype}({_a}, {_b})")

        self.exoiris.ps.freeze()
        self.exoiris.print_parameters()

    def init_prior_transform(self,):
        self.prior_transform = Priors(self.exoiris.ps)

    def fit_white(self, covariates: list[np.ndarray], pool:Optional[Pool]=None):
        print("Fitting white light curves to validate covariates...")
        self.exoiris._wa = NewWhiteLPF(self.exoiris._tsa, covariates=covariates)

        niter = self.cfg["SAMPLER"]["niter_white"]
        npop = self.cfg["SAMPLER"]["nwalkers"]

        def white_lnpost(pv):
            return self.exoiris._wa.lnposterior(pv)
        
        self.exoiris._wa.optimize_global(niter=niter, npop=npop, pool=pool, 
            lnpost=white_lnpost, plot_convergence=False, use_tqdm=True, leave=True)

        self.exoiris._wa.optimize()

        self.exoiris.period = self.exoiris._wa.de.minimum_location[0]
        self.exoiris.zero_epoch = self.exoiris._wa.transit_center
        self.exoiris.transit_duration = self.exoiris._wa.transit_duration
        self.exoiris.data.mask_transit(self.exoiris.zero_epoch, 
            self.exoiris.period, self.exoiris.transit_duration)
        
    def update_covariates(self,):
        fm = squeeze(self.exoiris._wa.flux_model(self.exoiris._wa.de.minimum_location))
        for i, (_t, _cov) in enumerate(zip(self.exoiris._wa.times, self.exoiris._wa.covariates)):
            newt = self.exoiris.data[i].time
            # if "JWST" in self.exoiris.data[i].name:
            sl = self.exoiris._wa.lcslices[i]
            white_systematics = self.exoiris._wa.ofluxa[sl] - fm[sl]
            self.exoiris.data[i].covs[:, -1] = np.interp(newt, _t, white_systematics)
            # else: # HST
            #     newcov = [np.interp(newt, _t, _c) for _c in _cov.T]
            #     self.exoiris.data[i].covs[:] = np.array(newcov).T
        print("Covariates updated based on white light curve fit.")

    def sample_from_prior(self, size: int) -> ndarray:
        return squeeze(self.exoiris.ps.sample_from_prior(size))
 
    def lnposterior(self, pv: ndarray) -> float:
        """  Evaluate the log posterior for a given parameter vector pv. """
        return self.exoiris._tsa.lnposterior(pv)
    
    def lnlikelihood_ns(self, pv: ndarray) -> float:
        """ Evaluate the log likelihood for nested sampling. """
        # if rand() < 1e-3: # prevent memory leak in long runs 
        #     gc.collect()
        return self.exoiris._tsa.lnlikelihood_ns(pv)

    def run_de(self, niter: int = 200, npop: Optional[int] = None, 
        initial_population: Optional[ndarray] = None, pool: Optional[Pool] = None, 
        lnpost: Optional[Callable]=None, min_ptp: float = 2.0, use_tqdm: bool = True) -> None:
        """Fit the spectroscopic light curves jointly using Differential Evolution.

        Parameters
        ----------
        niter: int
            Number of iterations for optimization. Default is 200.
        npop: int
            Population size for optimization.
        initial_population: Optional[ndarray]
            Initial population for optimization. If provided, npop is ignored. Default is None.
        pool: Optional[Pool]
            Multiprocessing pool for parallel optimization. Default is None.
        lnpost: Optional[Callable]
            Log posterior function for optimization. Default is None.
        """
        if isinstance(initial_population, ndarray):
            if initial_population.shape[1] == len(self.exoiris.ps):
                x0 = initial_population
                npop = x0.shape[0]
            else:
                raise ValueError(f"Wrong shape for initial_population. Expected (N, {len(self.exoiris.ps)}), got {initial_population.shape}.")
        elif npop > 0:
            x0 = squeeze(self.exoiris.ps.sample_from_prior(npop))
        else:
            raise ValueError("Either initial_population must be provided or npop must be a positive integer.")

        print("\nFitting the spectroscopic light curves using Differential Evolution...")

        self.exoiris._tsa.optimize_global(niter=niter, npop=npop, population=x0, 
            pool=pool, lnpost=lnpost, min_ptp=min_ptp, use_tqdm=use_tqdm, 
            leave=True, plot_convergence=False, vectorize=False, )
        self.exoiris.de = self.exoiris._tsa.de

    def run_mcmc(self, niter: int = 200, population: Optional[ndarray] = None, 
        thin: int = 1, repeats: int = 1, pool: Optional[Pool] = None, 
        lnpost: Optional[Callable] = None, use_tqdm: bool = True) -> None:
        """ 
        Run MCMC sampling to obtain the posterior distribution of parameters.

        Parameters
        ----------
        niter: int
            Number of iterations for MCMC sampling. Default is 200.
        population: Optional[ndarray]
            Initial population for MCMC sampling. If None, it will be initialized from the DE results.
        thin: int
            Thinning factor for MCMC sampling. Default is 1 (no thinning).
        repeats: int
            Number of times to repeat the MCMC sampling. Default is 1.
        pool: Optional[Pool]
            Multiprocessing pool for parallel sampling. Default is None.
        lnpost: Optional[Callable]
            Log posterior function for MCMC sampling. Must be provided if pool is not None.
        use_tqdm: bool
            Whether to use tqdm progress bar. Default is True.
        
        """
        print("\nRunning MCMC sampling...")
        if population is None:
            print("`population` is None. Initializing MCMC population from DE results.")

        self.exoiris._tsa.sample_mcmc(niter=niter, population=population, 
            thin=thin, repeats=repeats, pool=pool, lnpost=lnpost, use_tqdm=use_tqdm,
            leave=True, vectorize=False, save=False)

    def save_mcmc(self, overwrite: bool = False, config_file: str = None):
        # Results from ExoIris
        self.exoiris.save(overwrite=overwrite)
        outname = os.path.join(self.cfg['PATH']['output_dir'], self.exoiris.name+'.fits')
        shutil.move(self.exoiris.name+'.fits', outname)

        if config_file is not None:
            outname = os.path.join(self.cfg['PATH']['output_dir'], os.path.basename(config_file))
            shutil.copy(config_file, outname)
            print(f"Configuration file copied to {outname}.")

    def run_dynesty(self, loglikelihood: Callable, prior_transform: Callable, pool: Optional[Pool] = None, nlivepoints: int = 100, bound='multi', sample='rwalk', queue_size: int = None): 
 
        sampler = DynamicNestedSampler( 
            loglikelihood,
            prior_transform,
            len(self.exoiris._tsa.ps), 
            pool=pool,
            nlive=nlivepoints, 
            bound=bound,
            sample=sample,
            queue_size=queue_size, 
        ) 
        sampler.run_nested(
            dlogz_init=self.cfg["SAMPLER"].get("dlogz_init", 1.0),
            n_effective=self.cfg["SAMPLER"].get("n_effective", None),
            maxiter_batch=self.cfg["SAMPLER"].get("maxiter_batch", None),
            maxbatch=self.cfg["SAMPLER"].get("maxbatch", None),
        )

        results = sampler.results
        odir = self.cfg["PATH"]["output_dir"]
        with open(os.path.join(odir, 'ns_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
            print(f"Dynesty results saved to {os.path.join(odir, 'ns_results.pkl')}")

        results.summary() 
        n_effective = get_neff_from_logwt(results.logwt)
        print(f"Number of effective samples: {n_effective}")

        try:
            fig, axes = dyplot.runplot(results) 
            fig.tight_layout()
            fig.savefig(os.path.join(odir, 'dynesty_runplot.png'), dpi=100)
        except Exception as e:
            print(f"Error generating dynesty runplot: {e}")

        return results


class Priors:
    def __init__(self, prior_list: list[GParameter]):

        # initialize lists for different prior types
        u_idx, u_a, u_b = [], [], []
        n_idx, n_mean, n_std = [], [], []
        t_idx, t_a, t_b, t_mean, t_std = [], [], [], [], []

        for i, item in enumerate(prior_list):
            if isinstance(item.prior, UP):
                u_idx.append(i)
                u_a.append(item.prior.a)
                u_b.append(item.prior.b)
            elif isinstance(item.prior, NP):
                lb, ub = item.bounds
                mean, std = item.prior.mean, item.prior.std
                if isfinite(lb) and isfinite(ub):
                    # truncated normal
                    t_idx.append(i)
                    t_a.append((lb - mean) / std)
                    t_b.append((ub - mean) / std)
                    t_mean.append(mean)
                    t_std.append(std)
                else:
                    # unbounded normal
                    n_idx.append(i)
                    n_mean.append(mean)
                    n_std.append(std)
            else:
                raise TypeError(f"Unsupported prior type: {type(item.prior)}")

        self.u_idx = array(u_idx)
        self.u_a = array(u_a)
        self.u_b = array(u_b)

        self.n_idx = array(n_idx)
        self.n_mean = array(n_mean)
        self.n_std = array(n_std)

        self.tn_idx = array(t_idx)
        self.tn_a = array(t_a)
        self.tn_b = array(t_b)
        self.tn_mean = array(t_mean)
        self.tn_std = array(t_std)
 
    def __call__(self, cube):
        """ Transforms a unit cube to the parameter space defined by the priors. """
        cube = asarray(cube) 
        pv = empty_like(cube)

        if len(self.u_idx) > 0: # uniform priors
            pv[self.u_idx] = self.u_a + cube[self.u_idx] * (self.u_b - self.u_a)

        if len(self.n_idx) > 0: # normal priors
            pv[self.n_idx] = norm.ppf(
                cube[self.n_idx], loc=self.n_mean, scale=self.n_std
            )

        if len(self.tn_idx) > 0: # truncated normal priors
            pv[self.tn_idx] = truncnorm.ppf(
                cube[self.tn_idx],
                self.tn_a,
                self.tn_b,
                loc=self.tn_mean,
                scale=self.tn_std,
            )

        return pv