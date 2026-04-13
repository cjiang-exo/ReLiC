import argparse
import h5py
import json
import numpy as np
import matplotlib.pyplot as pl  
from astropy.io import fits
from astropy.stats import sigma_clip
from copy import deepcopy   
from exoiris.tslpf import TSLPF
from exoiris.wlpf import WhiteLPF
from exoiris.ldtkld import LDTkLD
from exoiris import ExoIris, TSData
from multiprocessing import Pool 
from numpy import atleast_2d, arctan2, dstack, sqrt, where, sqrt, isfinite, array,  log10, unique, average 
from petitRADTRANS import physical_constants as nc  
from petitRADTRANS.radtrans import Radtrans 
from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.physics import temperature_profile_function_guillot_global as get_tprofile
from petitRADTRANS.physics import rebin_spectrum_bin
from pytransit.orbits import as_from_rhop, i_from_ba
from pytransit.param import ParameterSet, UniformPrior as UP, NormalPrior as NP, GParameter
from pytransit import BaseLPF

NM_WHITE_MARGINALIZED = 0
NM_GP_FIXED = 1
NM_GP_FREE = 2
NM_WHITE_PROFILED = 3

class CustomWhiteLPF(WhiteLPF):
    def __init__(self, tsa: TSLPF):
        self.tsa = tsa
        fluxes, times, errors = [], [], []
        for t, f, e in zip(tsa.data.times, tsa.data.fluxes, tsa.data.errors):
            weights = where(isfinite(f) & isfinite(e), 1/e**2, 0.0)
            mf = average(where(isfinite(f), f, 0), axis=0, weights=weights)
            me = sqrt(1 / weights.sum(0))
            m = isfinite(mf)
            times.append(t[m])
            fluxes.append(mf[m])
            errors.append(me[m])
        covs = [(t-t.mean())[:, np.newaxis] for t in times]
        self.std_errors = errors
        self.neps = max(self.tsa.data.epoch_groups) + 1

        pbs = unique(tsa.data.noise_groups).astype('<U21')
        super(WhiteLPF, self).__init__('white', pbs, times, fluxes,
                        covariates=covs, wnids=tsa.data.noise_groups, pbids=tsa.data.noise_groups)

        self.tm.epids = array(self.tsa.data.epoch_groups)

        for i in range(self.neps):
            self.set_prior(f'tc_{i:02d}', tsa.ps[tsa.ps.find_pid(f'tc_{i:02d}')].prior)
        self.set_prior('p', tsa.ps[tsa.ps.find_pid('p')].prior)
        self.set_prior('rho', tsa.ps[tsa.ps.find_pid('rho')].prior)
        self.set_prior('b', tsa.ps[tsa.ps.find_pid('b')].prior) 
        self.set_prior('k2', 'UP', 0.01**2, 0.3**2) 
        ngids = tsa.data.noise_groups[self.lcids]
        for i in range(tsa.data.n_noise_groups):
            self.set_prior(f'wn_loge_{i}', 'NP', log10(np.diff(self.ofluxa[ngids==i]).std() / sqrt(2)), 0.1)

def custom_transit_model(self, pv, copy=True):
    """Evaluates the transit model for parameter vector pv.

    Parameters
    ----------
    pv : numpy.ndarray
        Array of transit parameters. Each row represents a set of transit parameters for a single transit event.
        The columns of the array should be in the following order:
        - Column 0: stellar density (g/cm^3)
        - Column 1: transit center time (T0)
        - Column 2: orbital period (P)
        - Column 3: impact parameter
        - Column 4: sqrt e cos w
        - Column 5: sqrt e sin w
        - Column 6: planet-to-star radius ratio (Rp/R_star)
    """
    pv = atleast_2d(pv)
    k = self.get_radius_ratios(pv) 
    ldp = self._eval_ldc(pv)
    t0s = pv[:, self._sl_tcs]
    p = pv[:, 1] 
    aor = as_from_rhop(pv[:, 0], p)
    inc = i_from_ba(pv[:, 2], aor)
    # ecc = pv[:, 3] ** 2 + pv[:, 4] ** 2
    # w = arctan2(pv[:, 4], pv[:, 3])
    ecc = 0 * p
    w = 0 * p
    epids = self.data.epoch_groups
    fluxes = []
    if isinstance(self.ldmodel, LDTkLD):
        ldp, istar = self.ldmodel(self.tms[0].mu, ldp)
        ldpi = dstack([ldp, istar])
        for i, tm in enumerate(self.tms):
            fluxes.append(tm.evaluate(k[i], ldpi[:, self.ldmodel.wlslices[i], :],
                                        t0s[:, epids[i]], p, aor, inc, ecc, w, copy))
    else:
        for i, tm in enumerate(self.tms):
            fluxes.append(tm.evaluate(k[i], ldp[i], t0s[:, epids[i]], p, aor, inc, ecc, w, copy))

    for i, d in enumerate(self.data):
        if d.offset_group > 0:
            biases = pv[:, self._start_bias + d.offset_group - 1][:, None, None]
            fluxes[i] = biases + (1.0 - biases) * fluxes[i]
    return fluxes 

def custom_init_parameters(self):
    self.ps = ParameterSet([])
    self._init_p_star()
    self._init_p_orbit()
    self._init_p_transit_centers()
    self._init_p_limb_darkening()
    self._init_p_atmosphere()
    self._init_p_noise()
    if self._nm == NM_GP_FREE:
        self._init_p_gp()
    self._init_p_bias()
    self.ps.freeze()
    self.generate_bandwidths() 
    return

def custom_init_p_orbit(self): 
    """ for circular orbits """
    ps = self.ps
    pp = [
        GParameter('p', 'period', 'd', NP(1.0, 1e-5), (0, np.inf)),
        GParameter('b', 'impact_parameter', 'R_s', UP(0.0, 1.0), (0, np.inf)), ]
    ps.add_global_block('orbit', pp)
    self._start_orbit = ps.blocks[-1].start
    self._sl_orbit = ps.blocks[-1].slice

def custom_init_p_atmosphere(self): 
    pp = [
        GParameter('mp', 'planet_mass', 'M_jup', NP(1.0, 1e-2), (0, np.inf)),
        GParameter('ref_p', 'reference pressure', 'log10 bar', UP(-6, 2), (-np.inf, np.inf)),
        GParameter('cloud_p', 'cloud-top pressure', 'log10 bar', UP(-6, 2), (-np.inf, np.inf)),
        GParameter('tp', 'temperature', 'K', UP(300, 3000), (0, np.inf)),
        GParameter('m2h', 'metallicity', 'log10 solar', UP(-1, 3), (-np.inf, np.inf)),
        GParameter('c2o', 'C/O ratio', '', UP(0.1, 1.6), (0, np.inf)),
        GParameter('cloud_f', 'cloud fraction', '', UP(0.0, 1.0), (0, 1)),
        ]
    self.ps.add_global_block('atmosphere', pp)
    self._start_atm = self.ps.blocks[-1].start
    self._sl_atm = self.ps.blocks[-1].slice
    return

def get_radius_ratios(self, pv):
    radius_ratios = []
    pv_atm = pv[:, self._sl_atm]  
    ts_model  = np.array([self.get_ts_model(atm_params) for atm_params in pv_atm])  
    for i, _d in enumerate(self.data):
        ts_rebinned = array([rebin_spectrum_bin(self.prt_wl, _ts, self.wavelengths[i], bin_widths=self.bandwidths[i]) for _ts in ts_model])
        radius_ratios.append(ts_rebinned**0.5)
    return radius_ratios

def init_prt_model(self, prt_atmosphere: Radtrans, prt_chem: PreCalculatedEquilibriumChemistryTable, planet_radius=1.0, star_radius=1.0):
    self.prt_atmosphere = prt_atmosphere
    self.prt_wl = 1e4 * prt_atmosphere.get_wavelengths() # A to micron
    self.prt_pbar = prt_atmosphere.pressures*1e-6 # cgs to bar
    self.prt_chem = prt_chem
    self.planet_radius = planet_radius * nc.r_jup_mean # cm 
    self.star_radius = star_radius * nc.r_sun # cm
    return

def get_ts_model(self, atm_params):
    planet_mass = atm_params[0] * nc.m_jup # g
    ref_pressure = 10**atm_params[1] # bar
    cloudtop_pbar = 10**atm_params[2] # bar
    
    # calculate the temperature profile
    ref_gravity = nc.G * planet_mass / self.planet_radius**2
    temperatures = np.full_like(self.prt_pbar, atm_params[3]) 
    
    # calculate chemical abundances
    metallicities = np.full_like(self.prt_pbar, atm_params[4])
    co_ratios = np.full_like(self.prt_pbar, atm_params[5])
    mass_fractions, mmw, nabla_ad = self.prt_chem.interpolate_mass_fractions(
    co_ratios           = co_ratios,
    log10_metallicities = metallicities,
    temperatures        = temperatures,
    pressures           = self.prt_pbar,
    full                = True
    )
    mass_fractions['CO-NatAbund'] = mass_fractions.pop('CO')

    # calculate the transmission spectrum, with clouds
    _, tr_c, _ = self.prt_atmosphere.calculate_transit_radii(
        temperatures                = temperatures,
        mass_fractions              = mass_fractions,
        mean_molar_masses           = mmw,
        reference_gravity           = ref_gravity,
        planet_radius               = self.planet_radius,
        reference_pressure          = ref_pressure,
        opaque_cloud_top_pressure   = cloudtop_pbar,
    ) 

    transit_depths = (tr_c / self.star_radius)**2
    return transit_depths

def get_ts_model_patchycloud(self, atm_params):
    planet_mass = atm_params[0] * nc.m_jup # g
    ref_pressure = 10**atm_params[1] # bar
    cloudtop_pbar = 10**atm_params[2] # bar
    
    # calculate the temperature profile
    ref_gravity = nc.G * planet_mass / self.planet_radius**2
    temperatures = np.full_like(self.prt_pbar, atm_params[3]) 
    
    # calculate chemical abundances
    metallicities = np.full_like(self.prt_pbar, atm_params[4])
    co_ratios = np.full_like(self.prt_pbar, atm_params[5])
    mass_fractions, mmw, nabla_ad = self.prt_chem.interpolate_mass_fractions(
    log10_metallicities = metallicities,
    co_ratios           = co_ratios,
    temperatures        = temperatures,
    pressures           = self.prt_pbar,
    full                = True
    )
    mass_fractions['CO-NatAbund'] = mass_fractions.pop('CO')

    # calculate the transmission spectrum, with clouds
    _, tr_c, _ = self.prt_atmosphere.calculate_transit_radii(
        temperatures                = temperatures,
        mass_fractions              = mass_fractions,
        mean_molar_masses           = mmw,
        reference_gravity           = ref_gravity,
        planet_radius               = self.planet_radius,
        reference_pressure          = ref_pressure,
        opaque_cloud_top_pressure   = cloudtop_pbar,
    ) 

    # calculate the transmission spectrum, without clouds
    _, tr_nc, _ = self.prt_atmosphere.calculate_transit_radii(
        temperatures                = temperatures,
        mass_fractions              = mass_fractions,
        mean_molar_masses           = mmw,
        reference_gravity           = ref_gravity,
        planet_radius               = self.planet_radius,
        reference_pressure          = ref_pressure,
        opaque_cloud_top_pressure   = None,
    ) 

    # average the patchy cloud model
    cloud_frac = atm_params[6] 
    td_c = (tr_c / self.star_radius)**2
    td_nc = (tr_nc / self.star_radius)**2
    transit_depth = cloud_frac * td_c + (1 - cloud_frac) * td_nc
    return transit_depth

def generate_bandwidths(self):
    self.bandwidths = []
    for wl in self.wavelengths: 
        dwl         = np.zeros_like(wl)
        dwl[:-1]    = np.diff(wl)
        dwl[-1]     = dwl[-2] 
        self.bandwidths.append(dwl)
    return self.bandwidths

def replace_outliers(time, flux, ferr, sigma=8):
    mask = sigma_clip(flux, sigma=sigma, axis=1, masked=True, copy=False).mask
    print(f"{np.sum(mask)} outliers detected.")
    for i, maskrow in enumerate(mask):
        if np.any(maskrow):
            x = time[~maskrow]
            y = flux[i, ~maskrow]
            ye = ferr[i, ~maskrow]
            flux[i] = np.interp(time, x, y)
            ferr[i] = np.interp(time, x, ye) 
    return flux, ferr

def custom_fit_white(self, niter: int = 500) -> None: 
    """Fit a white light curve model and sets the out-of-transit mask.

    Parameters
    ----------
    niter : int, optional
        The number of iterations for the global optimization algorithm (default is 500).
    """
    self._wa = CustomWhiteLPF(self._tsa)
    self._wa.optimize_global(niter, plot_convergence=False, use_tqdm=False)
    self._wa.optimize()
    pv = self._wa._local_minimization.x
    self.period = pv[0]
    self.zero_epoch = self._wa.transit_center
    self.transit_duration = self._wa.transit_duration
    self.data.mask_transit(self.zero_epoch, self.period, self.transit_duration)

def print_info(comm, string):
    """
    Print something when using mpiexec
    """
    if comm.Get_rank() == 0:
        print(string, flush=True)
    comm.Barrier()

def print_elapsed_time(elapsed_time:float):   
    """
    Prints the elapsed time in a formatted string of hours, minutes, and seconds.

    from time import time as current_time
    time_start = current_time()
    sleep(5)
    time_end = current_time() 
    elapsed_time = time_end - time_start

    Args:
        elapsed_time (float): The elapsed time in seconds.

    Returns:
        None
    """
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    output_str = f"{int(hours):02}:{int(minutes):02}:{seconds:05.2f}"
    print("Time elapsed: "+output_str)
    return output_str

TSLPF.transit_model       = custom_transit_model
TSLPF._init_parameters    = custom_init_parameters
TSLPF._init_p_orbit       = custom_init_p_orbit
TSLPF._init_p_atmosphere  = custom_init_p_atmosphere
TSLPF.init_prt_model      = init_prt_model
TSLPF.get_ts_model        = get_ts_model_patchycloud
TSLPF.get_radius_ratios   = get_radius_ratios
TSLPF.generate_bandwidths = generate_bandwidths

ExoIris.fit_white         = custom_fit_white