import numba
from astropy.stats import sigma_clip
from mpi4py import MPI
from numpy import (any, array, asarray, concatenate, exp, float64, hstack,
                   int64, interp, log, log2, ndarray, round, searchsorted,
                   sqrt, sum as np_sum, zeros, zeros_like)
from numpy.polynomial import Chebyshev

from petitRADTRANS.physics import rebin_spectrum_bin
from .core import ReLic

class SpectrumDownsampler:
    def __init__(self, wl_model: ndarray, wl_data: ndarray, wl_binwidths: ndarray, resolving_powers: ndarray, n_sigma=3):
        """ 
        Parameters
        ----------
        wl_model : ndarray (1D, float64)
            Wavelength array of the high-resolution model spectrum.
        wl_data : ndarray (1D, float64)
            Wavelength array of the observed spectra (concatenated and sorted).
        wl_binwidths : ndarray (1D, float64)
            Wavelength bin widths of the observed spectra.
        resolving_powers : ndarray (1D, float64)
            Instrument resolving power at each wavelength in wl_data.
        n_sigma : float
            Number of standard deviations to include in the Gaussian kernel for convolution.
        """
        self.wl_model = asarray(wl_model)
        self.wl_data = asarray(wl_data)
        self.wl_binwidths = asarray(wl_binwidths)
        self.rp_data = asarray(resolving_powers)
        self.rp_model = interp(wl_model, wl_data, resolving_powers)

        sigmas = wl_model / self.rp_model / (2 * sqrt(2 * log(2)))
        half_widths = n_sigma * sigmas
 
        n_wl = len(wl_model)
        self.slice_starts = zeros(n_wl, dtype=int64)
        self.slice_ends = zeros(n_wl, dtype=int64)
        kernel_list = []
        self.kernel_offsets = zeros(n_wl + 1, dtype=int64) 

        for i, (wl, hw, sigma) in enumerate(zip(wl_model, half_widths, sigmas)):
            lo, hi = wl - hw, wl + hw
            start = searchsorted(wl_model, lo, side='left')
            end = searchsorted(wl_model, hi, side='right')
            self.slice_starts[i] = start
            self.slice_ends[i] = end
            sub_wl = wl_model[start:end] - wl
            kernel = exp(-0.5 * (sub_wl / sigma) ** 2)
            kernel /= kernel.sum()
            kernel_list.append(kernel)
            self.kernel_offsets[i + 1] = self.kernel_offsets[i] + len(kernel)

        self.kernels_flat = concatenate(kernel_list).astype(float64)

        self.convolved_flux = zeros_like(wl_model)
        self.rebinned_flux = zeros_like(wl_data)

    def convolve(self, model_flux):
        _convolve_numba(
            asarray(model_flux, dtype=float64),
            self.slice_starts, self.slice_ends,
            self.kernels_flat, self.kernel_offsets,
            self.convolved_flux,
        )
        return self.convolved_flux

    def rebin(self, model_flux):
        self.rebinned_flux[:] = rebin_spectrum_bin(self.wl_model, model_flux, self.wl_data, self.wl_binwidths)
        return self.rebinned_flux

    def convolve_and_rebin(self, model_flux):
        _convolve_numba(
            asarray(model_flux, dtype=float64),
            self.slice_starts, self.slice_ends,
            self.kernels_flat, self.kernel_offsets,
            self.convolved_flux,
        )
        self.rebinned_flux[:] = rebin_spectrum_bin(self.wl_model, self.convolved_flux, self.wl_data, self.wl_binwidths)
        return self.rebinned_flux

@numba.njit(cache=True)
def _convolve_numba(model_flux, slice_starts, slice_ends, kernels_flat, kernel_offsets, convolved_flux):
    """Numba-accelerated Gaussian kernel convolution.
    
    Parameters
    ----------
    model_flux : np.ndarray (1D, float64)
        High-resolution model flux.
    slice_starts, slice_ends : np.ndarray (1D, int64)
        Start (inclusive) and end (exclusive) slice indices for each kernel window.
    kernels_flat : np.ndarray (1D, float64)
        All kernels concatenated into a single flat array.
    kernel_offsets : np.ndarray (1D, int64)
        Offset into kernels_flat for each kernel; length = n_kernels + 1.
    convolved_flux : np.ndarray (1D, float64)
        Pre-allocated output array.
    """
    n = len(slice_starts)
    for i in range(n):
        start = slice_starts[i]
        end = slice_ends[i]
        if start == end:
            convolved_flux[i] = 0.0
            continue
        k_start = kernel_offsets[i] 
        s = 0.0
        for j in range(end - start):
            s += model_flux[start + j] * kernels_flat[k_start + j]
        convolved_flux[i] = s
    return convolved_flux

def replace_outliers(time, flux, ferr, sigma=8):
    mask = sigma_clip(flux, sigma=sigma, axis=1, masked=True, copy=False).mask
    print(f"{np_sum(mask)} outliers detected.")
    for i, maskrow in enumerate(mask):
        if any(maskrow):
            x = time[~maskrow]
            y = flux[i, ~maskrow]
            ye = ferr[i, ~maskrow]
            flux[i] = interp(time, x, y)
            ferr[i] = interp(time, x, ye) 
    return flux, ferr

def generate_covariates(relic: ReLic, jitters: list) -> list[ndarray]:
    period_hst = 95.42 / 1440.0
    _standardize = lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1 # to [-1, 1]

    covariates = []
    for i, d in enumerate(relic.exoiris._tsa.data):
        if ("HST" in d.name) or ("STIS" in d.name) or ("WFC3" in d.name): 
            phases = (d.time - d.time[0]) % period_hst # folded phases
            phases[phases >= 0.75*period_hst] -= period_hst
            x = _standardize(phases)
            _covs = array([Chebyshev.basis(deg)(x) for deg in range(5)]).T
            if "STIS" in d.name:
                _covs = hstack((_covs, jitters[i])) 
        else:
            x = _standardize(d.time)
            n_baseline = relic.exoiris.data[i].n_baseline
            _covs = array([x**j for j in range(n_baseline+1)]).T 
        covariates.append(_covs)
    return covariates

def print_info(comm: MPI.Comm, string: str):
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

    Parameters:
        elapsed_time (float): The elapsed time in seconds.

    Returns:
        None
    """
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    output_str = f"{int(hours):02}:{int(minutes):02}:{seconds:05.2f}"
    print("Time elapsed: " + output_str)
    return output_str

def get_maxlike_estimates(relic: ReLic):
    try:
        lnp = relic.exoiris.sampler.get_log_prob() 
        postsamples = relic.exoiris._tsa.sampler.flatchain 
    except AttributeError as e: 
        raise RuntimeError("MCMC sampling not performed.") from e
    maxlike_params = postsamples[lnp.flatten().argmax()] 
    return maxlike_params

def optimize_parallelization(nlivepoints:int, npools:int, allow_memoryleak: float = 1.0):
    """    
    Optimize nested-sampling parallelization settings for multiprocessing.
 
    Args:
        nlivepoints (int):
            Initial number of nested-sampling live points.
        npools (int):
            Number of worker pools/processes available.
        allow_memoryleak (float, optional):
            Allowed memory growth budget in GiB used to scale `maxtasksperchild`.
            Defaults to 1.0.
    Returns:
        tuple:
            - suggested_nlivepoints (int): Adjusted number of live points for better parallel efficiency.
            - maxtasks (int): Suggested `maxtasksperchild` to mitigate memory leaks during sampling.
    """

    _d = log2(nlivepoints/npools) 
    suggested_nlivepoints = max(int(2**round(_d) * npools), npools)
    
    _chunksize, _extra = divmod(suggested_nlivepoints, (npools * 4))
    if _extra:
        _chunksize += 1 
    maxtasks = max(4, int(40000 * allow_memoryleak // nlivepoints) // 4 * 4)

    print(f"Suggested n_live_points: {suggested_nlivepoints} (rounded from {nlivepoints})")
    print(f"Suggested maxtasksperchild: {maxtasks}")
    return suggested_nlivepoints, maxtasks