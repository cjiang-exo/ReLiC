import numba
from astropy.stats import sigma_clip
from numpy import (any, array, asarray, concatenate, exp, float64, hstack,
                   int64, interp, log, log2, ndarray, round, searchsorted,
                   sqrt, sum as np_sum, zeros, zeros_like, loadtxt)
from numpy.polynomial import Chebyshev
from petitRADTRANS.physics import rebin_spectrum_bin 

class SpectrumDownsampler:
    def __init__(self, wl_model: ndarray, wl_data: ndarray, wl_binwidths: ndarray, spec_resolving_power_file: str | None = None, n_sigma=3):
        """ 
        Parameters
        ----------
        wl_model : ndarray (1D, float64)
            Wavelength array of the high-resolution model spectrum.
        wl_data : ndarray (1D, float64)
            Wavelength array of the observed spectra (concatenated and sorted).
        wl_binwidths : ndarray (1D, float64)
            Wavelength bin widths of the observed spectra.
        spec_resolving_power_file : str | None
            Path to the file containing the instrument resolving power at each wavelength in wl_data.
        n_sigma : float
            Number of standard deviations to include in the Gaussian kernel for convolution.
        """

        _minwl         = wl_data[0] - wl_binwidths[0]
        _maxwl         = wl_data[-1] + wl_binwidths[-1]
        start          = searchsorted(wl_model, _minwl, side='left')
        end            = searchsorted(wl_model, _maxwl, side='right')
        self.sl_wl     = slice(start, end) 

        self.wl_model_trim = asarray(wl_model[self.sl_wl])
        self.wl_data       = asarray(wl_data)
        self.wl_binwidths  = asarray(wl_binwidths)

        self.convolved_flux = zeros_like(self.wl_model_trim, dtype=float64)
        self.rebinned_flux  = zeros_like(self.wl_data, dtype=float64)

        if not spec_resolving_power_file:  # no convolution, just rebinning
            self._convolve = False
        else:  # precompute convolution kernels
            self._convolve = True

            _wl, _rp = loadtxt(spec_resolving_power_file).T
            self.rp_model = interp(self.wl_model_trim, _wl, _rp)

            sigmawls = self.wl_model_trim / self.rp_model / (2 * sqrt(2 * log(2)))
            half_widths = n_sigma * sigmawls

            n_wl                = len(self.wl_model_trim)
            self.slice_starts   = zeros(n_wl, dtype=int64)
            self.slice_ends     = zeros(n_wl, dtype=int64)
            self.kernel_offsets = zeros(n_wl + 1, dtype=int64)
            
            kernel_list         = []
            for i, (wl, hw, sigma) in enumerate(
                zip(self.wl_model_trim, half_widths, sigmawls)): 

                start  = searchsorted(self.wl_model_trim, wl - hw, side='left')
                end    = searchsorted(self.wl_model_trim, wl + hw, side='right')
                self.slice_starts[i] = start
                self.slice_ends[i] = end
                sub_wl = self.wl_model_trim[start:end] - wl
                kernel = exp(-0.5 * (sub_wl / sigma) ** 2)
                kernel /= kernel.sum()
                kernel_list.append(kernel)

                len_kernel = end - start
                self.kernel_offsets[i + 1] = self.kernel_offsets[i] + len_kernel

            self.kernels_flat = concatenate(kernel_list).astype(float64)

    def convolve(self, model_flux: ndarray): 
        if self._convolve:
            _convolve_numba(
                model_flux[self.sl_wl],
                self.slice_starts, self.slice_ends,
                self.kernels_flat, self.kernel_offsets,
                self.convolved_flux,
            ) 
            return self.convolved_flux
        return model_flux

    def rebin(self, model_flux: ndarray):
        self.rebinned_flux[:] = rebin_spectrum_bin(
            self.wl_model_trim, model_flux[self.sl_wl], 
            self.wl_data, self.wl_binwidths)
        return self.rebinned_flux

    def convolve_and_rebin(self, model_flux: ndarray):
        if self._convolve:
            _convolve_numba(
                model_flux[self.sl_wl],
                self.slice_starts, self.slice_ends,
                self.kernels_flat, self.kernel_offsets,
                self.convolved_flux,
            )
        else:
            self.convolved_flux[:] = model_flux[self.sl_wl]
        self.rebinned_flux[:] = rebin_spectrum_bin(self.wl_model_trim, self.convolved_flux, self.wl_data, self.wl_binwidths)
        return self.rebinned_flux

@numba.njit(cache=True)
def _convolve_numba(model_flux, slice_starts, slice_ends, kernels_flat, kernel_offsets, convolved_flux):
    """Numba-accelerated Gaussian kernel convolution.
    
    Parameters
    ----------
    model_flux : ndarray (1D, float64)
        High-resolution model flux.
    slice_starts, slice_ends : ndarray (1D, int64)
        Start (inclusive) and end (exclusive) slice indices for each kernel window.
    kernels_flat : ndarray (1D, float64)
        All kernels concatenated into a single flat array.
    kernel_offsets : ndarray (1D, int64)
        Offset into kernels_flat for each kernel; length = n_kernels + 1.
    convolved_flux : ndarray (1D, float64)
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

def generate_covariates(relic, jitters: list) -> list[ndarray]:
    period_hst = 95.42 / 1440.0
    _standardize = lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1 # to [-1, 1]

    covariates = []
    for i, d in enumerate(relic.exoiris._tsa.data):
        n_baseline = relic.exoiris.data[i].n_baseline
        if ("HST" in d.name) or ("STIS" in d.name) or ("WFC3" in d.name): 
            phases = (d.time - d.time[0]) % period_hst # folded phases
            phases[phases >= 0.75*period_hst] -= period_hst
            x = _standardize(phases)
            _covs = array([Chebyshev.basis(deg)(x) for deg in range(n_baseline+1)]).T
            if jitters[i] is not None:
                _covs = hstack([_covs, _standardize(jitters[i])]) 
        else: # JWST
            x = _standardize(d.time)
            _covs = array([Chebyshev.basis(deg)(x) for deg in range(n_baseline+1)]).T 
        covariates.append(_covs)
    return covariates

def print_info(comm, string: str):
    """
    Print something when using mpiexec

    Parameters
    ----------
    comm : mpi4py.MPI.Comm
        MPI communicator.
    string : str
        String to print.
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

def get_maxlike_estimates(relic):
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