import os
import pickle
import shutil
import tomllib
from functools import reduce
from multiprocessing.pool import Pool
from typing import Callable, Literal, Optional

import h5py
import numpy as np
from dynesty import DynamicNestedSampler
from dynesty import plotting as dyplot
from dynesty.utils import get_neff_from_logwt
from exoiris import ExoIris, TSData, TSDataGroup
from exoiris.ldtkld import LDTkLD
from numpy import array, asarray, empty_like, floor, hstack, isfinite, ndarray, squeeze, unique
from numpy.polynomial import Chebyshev
from numpy.random import default_rng
from pytransit.param import GParameter, NormalPrior as NP, UniformPrior as UP
from scipy.stats import norm, truncnorm

from .atmosphere import *
from .tslpf import NewTSLPF
from .white import NewWhiteLPF

NM_WHITE_MARGINALIZED = 0
NM_GP_FIXED = 1
NM_GP_FREE = 2
NM_WHITE_PROFILED = 3

""" pv should be scalar input, no more vectorization """

class Relic: 

    def __init__(self, configuration_file: str, idle: bool = False):

        self.idle = idle
        self.cfg = self._load_config(configuration_file)

        self.atmos_model = eval(self.cfg['ATMOSPHERE']['model_class'])(self.cfg)
        if self.atmos_model.__class__.__call__ is BaseAtmosphere.__call__:
            raise NotImplementedError("The __call__ method must be implemented to compute a spectrum.")
        if self.atmos_model.wavelengths is None: 
            raise ValueError("The atmosphere model must have initialized wavelength grid.") 

        if self.idle: 
            print("Idle mode.")
            return
        
        self.raw_data    = self._load_raw_data()
        self.tsdata      = self._init_TSData()
        self.ldmodel     = self._init_LDModel()
        self.exoiris     = self._init_ExoIris() 

        self._update_exoiris_parameters() 

        self.exoiris._wa = NewWhiteLPF(self.exoiris._tsa)

        if self.cfg['SAMPLER']['method'] in 'dynesty' :
            self.prior_transform = Priors(self.exoiris.ps)
        elif self.cfg['SAMPLER']['method'] == 'emcee':
            pass
        else:
            raise ValueError(f"Sampling method should be either 'dynesty' or 'emcee', got: {self.cfg['SAMPLER']['method']}")
        
        print("Initialization complete.", flush=True) 

    def _load_config(self, configuration_file: str):

        cfg = tomllib.load(open(configuration_file, 'rb'))
        if self.idle:
            return cfg
        
        # Verify that the configuration file contains all required sections and keys
        required_sections = [
            "PATH", "STAR", "PLANET", "ATMOSPHERE", "EXOIRIS", "PRIORS", "SAMPLER"
        ]
        for s in required_sections:
            if s not in cfg:
                raise ValueError(f"Missing required section '{s}' in configuration file.")

        # Verify that the input files exist
        lightcurve_files = cfg["PATH"].get("lightcurve_files", [])
        for f in lightcurve_files:
            if not os.path.isfile(f):
                raise FileNotFoundError(f"Input file '{f}' does not exist.")

        os.makedirs(cfg["PATH"]["output_dir"], exist_ok=True)
        shutil.copy(configuration_file, os.path.join(
            cfg["PATH"]["output_dir"], os.path.basename(configuration_file)
        ))
        print(f"Configuration file loaded: {configuration_file}", flush=True)
        return cfg
        
    def _load_raw_data(self) -> list[h5py.File]:
        filelist = self.cfg["PATH"]["lightcurve_files"]
        print("\nLoading data: ", flush=True)
        [print(f"  {f}") for f in filelist]
        return [h5py.File(f, 'r') for f in filelist]
    
    def _init_TSData(self) -> TSDataGroup:
        dlist = []
        for i, rd in enumerate(self.raw_data):
            try:  # specify edges for STIS and WFC3
                wl_edges = rd['wavelength_bins'][:].T
            except KeyError:
                wl_edges = None

            time = _get_data(rd, ['bjd_tdb', 'bjd', 'time'])
            wavelength = _get_data(rd, ['wavelength', 'wavelengths'])
            fluxes = _get_data(rd, ['flux', 'fluxes'])
            flux_errors = _get_data(rd, ['errors', 'flux_err', 'flux_error', 'flux_errors', 'ferrors']) 
            if fluxes.shape[0] != len(wavelength):
                fluxes = fluxes.T
                flux_errors = flux_errors.T
 
            dlist.append(TSData(
                time        = np.asarray(time) - 2459890.2, 
                wavelength  = wavelength, 
                fluxes      = fluxes, 
                errors      = flux_errors, 
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
            ), flush=True)

            r = self.cfg["EXOIRIS"]["rebin_resolutions"][i]
            if r > 0:
                dlist[-1] = dlist[-1].bin_wavelength(r=r, estimate_errors=False)
                print(f"  Rebinned to resolution R={r}. "
                      f"New nwl={dlist[-1].fluxes.shape[0]}.", flush=True)
            else:
                print("  No wavelength binning applied, using native resolution.", flush=True)

        return reduce(lambda x,y: x+y, dlist)
    
    def _init_LDModel(self):
        print('\nInitializing LDTk model... It takes 1 -- 30 minutes. Be patient!', flush=True)

        _t = self.cfg['STAR']['teff']
        _g = self.cfg['STAR']['logg']
        _m = self.cfg['STAR']['metal']
        
        return LDTkLD(
            data    = self.tsdata, 
            teff    = (_t[0], max(_t[1], 50)),
            logg    = (_g[0], max(_g[1], 0.02)),
            metal   = (_m[0], max(_m[1], 0.02)),
            dataset = 'visir'
        )
    
    def _init_ExoIris(self):
        print("\nInitializing ExoIris model...", flush=True)

        return RelicExoIris( 
            name           = self.cfg["PLANET"]["name"], 
            ldmodel        = self.ldmodel, 
            data           = self.tsdata, 
            atmos_model    = self.atmos_model,  
            spec_resolving_power_files = self.cfg["PATH"]["spec_resolving_power_files"],
            circular_orbit = self.cfg["PLANET"]["circular_orbit"],
            noise_model    = self.cfg["EXOIRIS"]["noise_model"],  
        )
    
    def _update_exoiris_parameters(self, ):

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

    def update_covariates(self, additional_covariates: list[np.ndarray], ):
        covariates = self.generate_covariates(additional_covariates)
        for i, cov in enumerate(covariates):
            self.exoiris.data[i].covs = cov.copy()
        self.exoiris._wa = NewWhiteLPF(self.exoiris._tsa, covariates=covariates)

    def fit_white(self, pool:Optional[Pool]=None, lnpost:Optional[Callable]=None, npop=100):
        print("Fitting white light curves to validate covariates...", flush=True)

        niter = self.cfg["SAMPLER"]["niter_white"] 
        vectorize = True if pool is None else False

        if pool is not None and lnpost is None:
            raise ValueError("If `pool` is provided, `lnpost` must also be provided for parallel optimization.")
        
        if lnpost is None:
            lnpost = self.exoiris._wa.lnposterior
        
        self.exoiris._wa.optimize_global(niter=niter, npop=npop, pool=pool, 
            lnpost=lnpost, plot_convergence=False, 
            vectorize=vectorize, use_tqdm=True, leave=True)

        self.exoiris._wa.optimize()

        self.exoiris.period = self.exoiris._wa.de.minimum_location[0]
        self.exoiris.zero_epoch = self.exoiris._wa.transit_center
        self.exoiris.transit_duration = self.exoiris._wa.transit_duration
        self.exoiris.data.mask_transit(self.exoiris.zero_epoch, 
            self.exoiris.period, self.exoiris.transit_duration
        ) 
        
    def generate_covariates(self, state_vectors: list=None) -> list[ndarray]:
        period_hst = 95.42 / 60.0 / 24.0 # in days
        _standardize = lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1 # to [-1, 1]

        covariates = []
        for i, d in enumerate(self.exoiris._tsa.data):
            n = self.exoiris.data[i].n_baseline
            if ("HST" in d.name) or ("STIS" in d.name) or ("WFC3" in d.name): 
                phases = (d.time - d.time[0]) % period_hst # phase-folded 
                phases[phases >= 0.75*period_hst] -= period_hst
                x = _standardize(phases) 
            else: # JWST
                x = _standardize(d.time) 
            _covs = array([Chebyshev.basis(deg)(x) for deg in range(n+1)]).T 
            if (state_vectors is not None) and (state_vectors[i] is not None):
                _covs = hstack([_covs, _standardize(state_vectors[i])]) 
            covariates.append(_covs)
        return covariates

    def sample_from_prior(self, size: int) -> ndarray:
        return squeeze(self.exoiris.ps.sample_from_prior(size))
 
    def lnposterior_mcmc(self, pv: ndarray) -> float:
        """  Evaluate the log posterior for a given parameter vector pv. """
        return self.exoiris._tsa.lnposterior(pv)
    
    def lnlikelihood_ns(self, pv: ndarray) -> float:
        """ Evaluate the log likelihood for nested sampling. """ 
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

        save_checkpoint = self.cfg["SAMPLER"].get("save_checkpoint", False)
        if save_checkpoint:
            checkpoint_file = os.path.join(self.cfg["PATH"]["output_dir"], 'checkpoint_dynesty.pkl')
        else:
            checkpoint_file = None

        resume = self.cfg["SAMPLER"].get("resume", False)
        if resume:
            if checkpoint_file is not None:
                sampler = DynamicNestedSampler.restore(checkpoint_file, pool=pool)
            else:
                raise ValueError("`resume` is True but no `checkpoint_file` specified in config.")
        else:
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
            dlogz_init=self.cfg["SAMPLER"].get("dlogz_init", 0.1),
            n_effective=self.cfg["SAMPLER"].get("n_effective", None),
            maxiter_init=self.cfg["SAMPLER"].get("maxiter_init", None),
            maxiter_batch=self.cfg["SAMPLER"].get("maxiter_batch", None),
            maxbatch=self.cfg["SAMPLER"].get("maxbatch", None),
            resume=self.cfg["SAMPLER"].get("resume", False),
            checkpoint_file=checkpoint_file,
            checkpoint_every=300,
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
    
    def run_test(self, nsamples:int=3, seed:int=None):
        print("Running a quick test of posterior evaluation...")

        ndim = len(self.exoiris._tsa.ps)
        
        if self.cfg['SAMPLER']['method'] == 'dynesty':
            rng = default_rng(seed)
            unit_cubes = rng.uniform(size=(nsamples, ndim))
            prior_params = [self.prior_transform(c) for c in unit_cubes]
            pp = [self.lnlikelihood_ns(p) for p in prior_params]
            [print(f"lnprob = {_v:.6e}") for _v in pp]

        elif self.cfg['SAMPLER']['method'] == 'emcee':
            prior_params = self.sample_from_prior(nsamples)
            pp = [self.lnposterior_mcmc(_p) for _p in prior_params]
            [print(f"lnprob = {_v:.6e}") for _v in pp]

        print("Test complete.") 
        return prior_params, pp

class RelicExoIris(ExoIris):
    def __init__(self, name: str, ldmodel, data: TSDataGroup | TSData,
            atmos_model: BaseAtmosphere, spec_resolving_power_files: list[str] = None,  
            tmpars: dict | None = None,
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
            spec_resolving_power_files=spec_resolving_power_files, tmpars=tmpars, noise_model=noise_model, circular_orbit=circular_orbit)
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

def _get_data(data: dict, key_aliases: list[str]) -> np.ndarray:
    for k in key_aliases:
        if k in data.keys():
            return data[k]
    raise KeyError(f"None of the keys {key_aliases} found in the data.")