# ReLiC Quick Start

A step-by-step guide to running atmospheric retrievals of transiting exoplanets with ReLiC.
 

## 1. Prepare your data

Use a data reduction pipeline to reduce raw observations into spectral light curves.

Each observation's spectral light curves should be stored in an HDF5 file containing:

- `bjd_tdb` — 1D time array
- `wavelength` — 1D wavelength array (bin centers) in μm
- `flux` — 2D flux array (n_wavelength × n_time)
- `flux_err` — 2D error array (n_wavelength × n_time)
- `wavelength_bins` (optional) — 2D array (n_wavelength × 2) of left and right bin edges. If omitted, bin edges are estimated from `numpy.diff(wavelength)`.

## 2. Prepare your atmospheric model (optional)

`relic.atmosphere` provides several built-in atmospheric models based on `petitRADTRANS`. The atmospheric models must inherit from `BaseAtmosphere` and implement `__call__(pv) -> ndarray` returning 1D transit depth array.

| Model Class | Chemistry | Temperature-pressure profile |
|---|---|---|
| `IsothermalEqChem` | Equilibrium (petitRADTRANS tables) | Isothermal |
| `IsothermalFreeChem` | Free chemistry | Isothermal |
| `TP6EqChem` | Equilibrium (petitRADTRANS tables) | Six-parameter (2009ApJ...707...24M) |
| `TP6FreeChem` | Free chemistry | Six-parameter (2009ApJ...707...24M) |
| `TP6FastChem` | FastCHEM equilibrium chemistry | Six-parameter (2009ApJ...707...24M) |

To implement a custom model, subclass BaseAtmosphere:

```python
from relic.atmosphere import BaseAtmosphere 

class MyAtmos(BaseAtmosphere):
    def __init__(self, config):
        super().__init__(config) 

        # Model initialization.
        # You must set the model wavelength array here.

        self.wavelengths = 1e4 * self.radtrans.get_wavelengths() # in micron
        ...

    def __call__(self, pv: ndarray) -> ndarray:
        # pv is a 1D array of **all** free parameters.  
        # You may extract atmospheric parameters from pv by:
        atmosphere_params = pv[self._sl_atm] 
        # self._sl_atm is a slice object automatically initialized by Relic 

        # Compute a transmission spectrum using the atmospheric parameters
        ...

        return transmission_spectrum  # must correspond to self.wavelengths
```

Then set `model_class = "MyAtmos"` in your configuration file.

## 3. Prepare your configuration file

ReLiC is configured through a TOML file. A minimal configuration includes `[PATH]`, `[STAR]`, `[PLANET]`, `[ATMOSPHERE]`, `[EXOIRIS]`, `[PRIORS]`, and `[SAMPLER]` sections — see `config/` for examples.

Below is a fully annotated minimal configuration:

```toml
[PATH]
lightcurve_files = [ "data1.h5", "data2.h5", ] # input spectral light curves
spec_resolving_power_files = [ "NIRCam.dat", "", ]  # ASCII files with two columns (wavelength in micron, resolving power). Required for pixel-resolution retrievals; use empty strings to omit convolution.
output_dir = "output/"  # directory for output files

[STAR] 
name        = "HD209458" # name of the host star
teff        = [6091, 50] # stellar effective temperature and error
logg        = [4.45, 0.02] # stellar logg and error
metal       = [0.01, 0.05] # stellar metallicity and error
radius_rsun = [1.19, 0.02] # stellar radius and error

[PLANET] 
name                    = "HD209458b" # name of the planet
transit_epoch_bjd       = [2455216.405640, 0.000094] # transit epoch in BJD_TDB 
transit_period_d        = [3.52474859, 0.00000038] # transit period in days
transit_duration_d      = [0.127, 0.001] # transit duration in days
radius_rjup             = [1.39, 0.02] # planet radius in Jupiter radius
mass_mjup               = [0.73, 0.04] # planet mass in Jupiter mass
equilibrium_temperature = [1459, 12] # planet equilibrium temperature in Kelvin
circular_orbit          = true # if true, eccentricity is fixed to zero; if false, two additional free parameters are added: sqrt(e)*cos(ω) and sqrt(e)*sin(ω)

[ATMOSPHERE] # Keys depend on the chosen atmospheric model
model_class              = "MyAtmos" # (required) class name of the atmospheric model
wavelength_bounds_micron = [1, 2]
chemical_species         = [ "H2O", "CO", "CO2", "CH4", ]
...

[EXOIRIS]
instruments = [ "JWST/NIRCam_F322W2", "JWST/NIRCam_F444W"] # instrument name for each dataset 
wl_range_micron   = [ [], [] ] # crop useful wavelength range (optional)
time_range_bjd    = [ [], [] ] # crop useful time range (optional)
noise_groups      = [0, 1] # group datasets with the same instrument setup so they share the same jitter factors
n_baselines       = [2, 2] # maximum degree of the polynomial baseline (must be ≥ 1)
rebin_resolutions = [100, 200] # set to 0 to use native pixel resolution
noise_model        = "white_marginalized" # one of ["white_marginalized", "white_profiled", "free_gp", "fixed_gp"]; see the ExoIris documentation for details

[PRIORS.TRANSIT] # priors on transit parameters
rho         = ["NP", 1.04, 0.07, 0.7, 1.4, "stellar density", "cgs"]
p           = ["NP", 3.52474859, 0.00000038, 0, inf, "transit period", "days"]
b           = ["NP", 0.5056, 0.0133, 0, 1, "impact parameter", "R_s"]
tc_00       = ["UP", 2459890.15, 2459890.25, -inf, inf, "transit epoch", "BJD_TDB"]

[PRIORS.STAR] # priors on stellar parameters
teff        = ["NP", 6091, 50, 5000, 7000, "stellar effective temperature", "K"]
logg        = ["NP", 4.45, 0.02, 4, 5, "stellar gravity", "log10 cgs"]
metal       = ["NP", 0.01, 0.05, -1, 1, "stellar metallicity", ""]

[PRIORS.NOISE] # priors on noise parameters
sigma_m_00  = ["UP", 0.2, 5, -inf, inf, "noise factor 00", ""]
sigma_m_01  = ["UP", 0.2, 5, -inf, inf, "noise factor 01", ""]

[PRIORS.ATMOSPHERE] # priors on atmospheric parameters
mp          = ["NP", 0.73, 0.04, 0.53, 0.93, "planet mass", "M_j"]
temp        = ["UP", 100, 3000, -inf, inf, "temperature", "K"]
m2h         = ["UP", -3, 3, -inf, inf, "atmospheric metallicity", "log10"]
c2o         = ["UP", 0.1, 2, -inf, inf, "C/O ratio", " "]
...

[SAMPLER]
npools      = 60    # number of processes for multiprocessing
niter_white = 1000  # number of DE iterations for white-light curve fitting
method      = "dynesty"  # sampler for retrievals: ["dynesty", "emcee"]
bound       = "multi"    # dynesty bound method
sample      = "rwalk"    # dynesty sampling method
n_live_points = 240  # number of live points
dlogz_init  = 0.1   # dlogz convergence threshold
n_effective = 400   # minimum effective sample size (secondary stopping criterion)

save_checkpoint = true   # enable checkpointing for resuming
resume          = false  # resume from the latest checkpoint
```
 

## 4. Usage

Multiprocessing is strongly recommended, even for demos. Before starting, limit the thread-level parallelism:

```python
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'

from multiprocessing import Pool
```

### 4.1 Initialization

Initialize `Relic` with your configuration file:

```python
config_file = "your-configurations.toml"
relic = Relic(config_file)
```

The parsed configuration is accessible as `relic.cfg`.

### 4.2 Fitting white light curves

Because orbital parameters are free in the ReLiC framework, fitting white light curves is not strictly required. However, we recommend it as a quick sanity check to verify that your noise model is appropriate for the data.

By default, `ExoIris` constructs polynomial covariates in time up to the degree specified by `n_baseline`. For some datasets (e.g., ground-based or HST light curves), you may want to supply additional state vectors to model more complex systematics. To do this, define the state vectors for each dataset in a list:

```python
state_vectors_dataset1 = np.asarray([
    trace_xposition, 
    trace_yposition,
    fwhm,
    ...
])
...
state_vectors_alldata = [
    state_vectors_dataset1,
    state_vectors_dataset2,
    None, # if you don't want to add any state vectors for dataset3
    None, # if you don't want to add any state vectors for dataset4
    ...
]
```

Each `state_vectors_dataset` should be a 2D array of shape `(n_time, n_state_vectors)`. Your state vectors are stacked with the default polynomial covariates to form the final covariate matrix for each dataset, so **do not** include polynomial time terms in `state_vectors_dataset`.

Update the covariates with:

```python
relic.update_covariates(state_vectors_alldata)
```

To fit the white light curves, run
```python
relic.fit_white()  # global optimization via differential evolution
```

Or with multiprocessing:
```python
def lnpost_white(pv):
    return relic.exoiris._wa.lnposterior(pv)

with Pool(relic.cfg["SAMPLER"]["npools"]) as pool:
    relic.fit_white(pool=pool, lnpost=lnpost_white)
```
The log-posterior function must be defined at the module level for pickling

To display the best-fit white light curves:

```python
visual = RelicVisualization(relic, dpi=100, save=True)
visual.plot_white()
```

Note: this white-light fit is a thin wrapper around `exoiris.wlpf.WhiteLPF` with minor modifications — no atmospheric or stellar model is involved.

### 4.3 Test likelihood evaluation

Before launching a full retrieval, evaluate the likelihood on a few random prior samples to catch errors in configuration early:

```python
relic.run_test(3)
```

### 4.4 Run the sampler

Below is an example of running `dynesty` nested sampling with multiprocessing. As before, the likelihood and prior-transform functions must be defined at the module level:

```python
def loglikelihood(pv):
    return relic.lnlikelihood_ns(pv)
def prior_transform(uv):
    return relic.prior_transform(uv)

with Pool(npools, maxtasksperchild=100) as pool:
    results = relic.run_dynesty(
        loglikelihood   = loglikelihood,
        prior_transform = prior_transform,
        pool            = pool,
        queue_size      = npools,
        nlivepoints     = relic.cfg["SAMPLER"]["n_live_points"],
        bound           = "multi",
        sample          = "rwalk", 
    )
```

The output `results` is a standard `dynesty.utils.Results` instance.

As a reference: with two JWST NIRCam datasets (2.5–5.0 μm), a retrieval with `n_live_points=240`, `npools=60`, and 10 chemical species takes roughly 10 hours.

> **Note:** We have observed a small, random memory leak (a few KB per call) in the likelihood function. For retrievals that involve millions of likelihood evaluations, set `maxtasksperchild` so worker processes are periodically recycled. `maxtasksperchild=100` is a safe default for most cases.

### 4.5 Visualize results

```python
# Before the retrieval: inspect 2D spectral light curves and limb-darkening profiles
visual.plot_2dfluxes(figname='fluxes.png')
visual.plot_ldprofiles(figname='ldprofiles.png')

# After the retrieval: posterior distributions, best-fit residuals, and the
# transmission spectrum at the maximum-likelihood point
maxlike_params = results['samples'][results['logl'].argmax()]

visual.plot_corners(samples=results.samples_equal(), truths=maxlike_params, figname='corners.pdf')
visual.plot_residuals(maxlike_params, figname='residuals.png')
visual.plot_transmission_spectra(maxlike_params, samples=results.samples_equal())
```
