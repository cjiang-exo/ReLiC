import numpy as np
from astropy.stats import sigma_clip 
from exoiris.tslpf import TSLPF 
from exoiris.ldtkld import LDTkLD
from exoiris import ExoIris

from numpy import (array, average, atleast_2d, arctan2, diff, dstack, inf, 
    isfinite, interp, log10, sqrt, where, unique, zeros_like, zeros, squeeze, 
    ones_like, isscalar)
from petitRADTRANS.physics import rebin_spectrum_bin
from pytransit.orbits import as_from_rhop, i_from_ba, epoch
from pytransit.param import ParameterSet, UniformPrior as UP, NormalPrior as NP, GParameter  

NM_WHITE_MARGINALIZED = 0
NM_GP_FIXED = 1
NM_GP_FREE = 2
NM_WHITE_PROFILED = 3

def calculate_transmission_spectrum(self, pv): 
    raise NotImplementedError()

def custom_flux_model(self, pv, include_baseline: bool = True): 
    pv = atleast_2d(pv)
    self._transmission_spectra = array([ 
        self.calculate_transmission_spectrum(_p) for _p in pv
    ]) 
    transit_models = self.transit_model(pv)
    if self.spot_model is not None:
        self.spot_model.apply_spots(pv, transit_models)
        if self.spot_model.include_tlse:
            self.spot_model.apply_tlse(pv, transit_models)
    if include_baseline:
        baseline_models = self.baseline_model(transit_models)
        for i in range(self.data.size):
            transit_models[i][:, :, :] *= baseline_models[i][:, :, :]
    return transit_models
    
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

    return fluxes 

def custom_init_parameters(self):
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
    return

def custom_init_p_orbit(self): 
    """ for circular orbits """
    ps = self.ps
    pp = [
        GParameter('p', 'period', 'd', NP(1.0, 1e-5), (0, inf)),
        GParameter('b', 'impact_parameter', 'R_s', UP(0.0, 1.0), (0, inf)), ]
    ps.add_global_block('orbit', pp)
    self._start_orbit = ps.blocks[-1].start
    self._sl_orbit = ps.blocks[-1].slice

def init_p_atmosphere(exoiris: ExoIris, atmosphere_parameter_set: list[GParameter]):  
    exoiris.ps.add_global_block('atmosphere', atmosphere_parameter_set)
    exoiris._tsa._start_atm = exoiris.ps.blocks[-1].start
    exoiris._tsa._sl_atm = exoiris.ps.blocks[-1].slice
    return

def get_radius_ratios(self, pv):
    radius_ratios = [] 
    for i, _d in enumerate(self.data):
        ts_rebinned = array([rebin_spectrum_bin(self.prt_wl, _ts, self.wavelengths[i], bin_widths=self.bin_widths[i]) for _ts in self._transmission_spectra])
        radius_ratios.append(ts_rebinned**0.5)
    return radius_ratios

def custom_lnposterior(self, pv):
    lnp = self.lnprior(pv)

    if isscalar(lnp):
        if isfinite(lnp):
            lnp += self.lnlikelihood(pv)
        return lnp if isfinite(lnp) else -inf 

    mask = isfinite(lnp) 
    lnp[mask] += self.lnlikelihood(pv[mask]) 
    return where(isfinite(lnp), lnp, -inf)

def replace_outliers(time, flux, ferr, sigma=8):
    mask = sigma_clip(flux, sigma=sigma, axis=1, masked=True, copy=False).mask
    print(f"{np.sum(mask)} outliers detected.")
    for i, maskrow in enumerate(mask):
        if np.any(maskrow):
            x = time[~maskrow]
            y = flux[i, ~maskrow]
            ye = ferr[i, ~maskrow]
            flux[i] = interp(time, x, y)
            ferr[i] = interp(time, x, ye) 
    return flux, ferr

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

def generate_binwidths(tsa: TSLPF):
    tsa.bin_widths = []
    for d in tsa.data:
        tsa.bin_widths.append(d._wl_r_edges - d._wl_l_edges)
    return None

TSLPF.flux_model          = custom_flux_model
TSLPF.transit_model       = custom_transit_model
TSLPF._init_parameters    = custom_init_parameters
TSLPF._init_p_orbit       = custom_init_p_orbit
# TSLPF._init_p_atmosphere  = custom_init_p_atmosphere
TSLPF.get_radius_ratios   = get_radius_ratios 
TSLPF.lnposterior         = custom_lnposterior
TSLPF.calculate_transmission_spectrum  = calculate_transmission_spectrum 

if __name__ == "__main__":
    print("Testing ...")