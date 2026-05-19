from astropy.stats import sigma_clip 
from numpy import any, interp, sum as np_sum, array, hstack, ndarray, log2, round
from mpi4py import MPI
from numpy.polynomial import Chebyshev

from relic_core import ReLic

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