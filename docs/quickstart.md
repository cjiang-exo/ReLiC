# ReLiC Quick Start

A step-by-step guide to running atmospheric retrievals of transiting exoplanets with [ReLiC](https://github.com/cjiang-exo/ReLiC).
 

## 1. Prepare your data

Use a data reduction pipeline to reduce raw observations into spectral light curves.

Each observation's spectral light curves should be stored in an HDF5 file containing:

- `bjd_tdb` — 1D time array
- `wavelength` — 1D wavelength array (bin centers) in μm
- `flux` — 2D flux array (n_wavelength × n_time)
- `flux_err` — 2D error array (n_wavelength × n_time)
- `wavelength_bins` (optional) — 2D array (n_wavelength × 2) of left and right bin edges. If omitted, bin edges are estimated from `numpy.diff(wavelength)`.

## 2. Prepare your atmospheric model (optional)

`relic.atmosphere` provides several built-in atmospheric models based on [`petitRADTRANS`](https://petitradtrans.readthedocs.io/en/latest/). The atmospheric models must inherit from `BaseAtmosphere` and implement `__call__(pv) -> ndarray` returning 1D transit depth array.

| Model Class | Temperature-pressure profile | Chemistry | 
|---|---|---| 
| `IsothermalFreeChem` | Isothermal | Free chemistry |
| `IsothermalFastChem` | Isothermal | [FastCHEM](https://newstrangeworlds.github.io/FastChem/) chemistry |
| `IsothermalEqChem` | Isothermal | Equilibrium (petitRADTRANS tables) |
| `GuillotFreeChem` | [Guillot (2010)](https://ui.adsabs.harvard.edu/abs/2010A%26A...520A..27G/abstract) | Free chemistry |
| `GuillotFastChem` | [Guillot (2010)](https://ui.adsabs.harvard.edu/abs/2010A%26A...520A..27G/abstract) | [FastCHEM](https://newstrangeworlds.github.io/FastChem/) chemistry |
| `GuillotEqChem` | [Guillot (2010)](https://ui.adsabs.harvard.edu/abs/2010A%26A...520A..27G/abstract) | Equilibrium (petitRADTRANS tables) |
| `M09FreeChem` | [Madhusudhan & Seager (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJ...707...24M/abstract) | Free chemistry |
| `M09FastChem` | [Madhusudhan & Seager (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJ...707...24M/abstract) | [FastCHEM](https://newstrangeworlds.github.io/FastChem/) chemistry |
| `M09EqChem` | [Madhusudhan & Seager (2009)](https://ui.adsabs.harvard.edu/abs/2009ApJ...707...24M/abstract) | Equilibrium (petitRADTRANS tables) |

To implement a custom model, subclass BaseAtmosphere:

```python
from relic.atmosphere import BaseAtmosphere 

class NewAtmosphere(BaseAtmosphere):
    def __init__(self, config):
        super().__init__(config) 

        # Model initialization.
        # You must set the model wavelength array here.

        self.wavelengths = 1e4 * self.radtrans.get_wavelengths() # in micron
        ...

    def __call__(self, pv: ndarray) -> ndarray:
        # pv is a 1D array of ALL free parameters including orbital, stellar, atmospheric, and noise parameters.  
        # You may extract atmospheric parameters from pv by:
        atmosphere_params = pv[self._sl_atm] 
        # self._sl_atm is a slice object automatically initialized by Relic 

        # Compute a transmission spectrum using the atmospheric parameters
        ...

        return transmission_spectrum  # must correspond to self.wavelengths
```

Then set `model_class = "NewAtmosphere"` in your configuration file.

## 3. Prepare your configuration file

ReLiC is configured through a TOML file. A minimal configuration includes `[PATH]`, `[STAR]`, `[PLANET]`, `[ATMOSPHERE]`, `[EXOIRIS]`, `[PRIORS]`, and `[SAMPLER]` sections — see `config/` for examples.

Below is a fully annotated minimal configuration:

```toml
[PATH]
lightcurve_files = [ "data1.h5", "data2.h5", ] # input spectral light curves
spec_resolving_power_files = [ "NIRCam.dat", "NIRCam.dat", ]  # ASCII files with two columns (wavelength in micron, resolving power), necessary for pixel-resolution retrievals; use empty strings "" to omit line spread function convolution for low-resolution retrievals.
output_dir = "output/"  # directory for output files

[STAR] 
name        = "HD 209458" # name of the host star
teff        = [6091, 50] # stellar effective temperature and error
logg        = [4.45, 0.02] # stellar logg and error
metal       = [0.01, 0.05] # stellar metallicity and error
radius_rsun = [1.19, 0.02] # stellar radius and error

[PLANET] 
name                    = "HD 209458 b" # name of the planet
transit_epoch_bjd       = [2455216.405640, 0.000094] # transit epoch in BJD_TDB 
transit_period_d        = [3.52474859, 0.00000038] # transit period in days
transit_duration_d      = [0.127, 0.001] # transit duration in days
radius_rjup             = [1.39, 0.02] # planet radius in Jupiter radius
mass_mjup               = [0.73, 0.04] # planet mass in Jupiter mass
equilibrium_temperature = [1459, 12] # planet equilibrium temperature in Kelvin
circular_orbit          = true # if true, eccentricity is fixed to zero; if false, two additional free parameters are added: sqrt(e)*cos(ω) and sqrt(e)*sin(ω)

[ATMOSPHERE] # Keys depend on the chosen atmospheric model
pressure_layers          = 101
pressure_bounds_log10bar = [-8, 2]
wavelength_bounds_micron = [2.0, 5.5]
chemical_species = [ 
  "H2O", "CO", "CO2", "CH4", 
]
rayleigh_species  = ["H2", "He"]
continuum_species = ["H2--H2", "H2--He", "H-"]
opacity_mode      = "lbl"
line_by_line_opacity_sampling = 1
model_class       = "GuillotFastChem"

[EXOIRIS]
instruments = [ "JWST/NIRCam_F322W2", "JWST/NIRCam_F444W"] # instrument name for each dataset 
wl_range_micron   = [ [], [] ] # crop useful wavelength range (optional)
time_range_bjd    = [ [], [] ] # crop useful time range (optional)
noise_groups      = [0, 1] # group datasets with the same instrument setup so they share the same jitter factors
n_baselines       = [2, 2] # maximum degree of the polynomial baseline (must be ≥ 1)
rebin_resolutions = [100, 200] # separate resolution for each instrument; set to 0 to use native pixel resolution
noise_model        = "white_marginalized" # select from ["white_marginalized", "white_profiled", "free_gp", "fixed_gp"]; see the ExoIris documentation for details

[FIXED_PARAMETERS]
planet_radius_rjup  = 1.39
star_radius_rsun    = 1.19
atm_tint            = 100

[PRIORS.TRANSIT] # priors on transit parameters 
# Format: ["NP"/"UP", a, b, lower_bound, upper_bound, description, unit]
# For NP (normal prior): (mean=a, std=b); for UP (uniform prior): (lower_bound=a, upper_bound=b)
rho         = ["NP", 1.04, 0.07, 0.7, 1.4, "stellar density", "cgs"]
p           = ["NP", 3.52474859, 0.00000038, 0, inf, "transit period", "days"]
b           = ["UP", 0, 1, 0, 1, "impact parameter", "R_s"]
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
ref_p       = ["UP", -8, 2, -inf, inf, "reference pressure", "log10bar"] 
cloud_p     = ["UP", -8, 2, -inf, inf, "cloud-top pressure", "log10bar"] 
cloud_f     = ["UP", 0.0, 1.0, 0, 1.0, "cloud fraction", ""]
haze_f      = ["UP", -3, 6, -inf, inf, "haze factor", "log10"]
lgkir       = ["UP", -5, 0, -inf, inf, "infrared opacity", "cm^2/g"]
lggamma     = ["UP", -2, 2, -inf, inf, "log10 of the ratio of visible to infrared opacity.", ""] 
beta        = ["UP", 0.03, 0.3, -inf, inf, "f * (1 - A_bond)", ""]
m2h         = ["UP", -3, 3, -inf, inf, "atmospheric metallicity", ""]
c2o         = ["UP", 0.1, 2.0, -inf, inf, "C/O ratio", ""]

[SAMPLER]
method      = "nautilus" 
npools      = 10    # number of processes for multiprocessing
niter_white = 1000  # number of DE iterations for white-light curve fitting
n_live_points = 100  # number of live points
n_effective = 400   # minimum effective sample size (stopping criterion)
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

Fitting the white light curve is not a required step. However, we recommend doing so as a quick sanity check to verify that your noise model is appropriate for the data.

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

Below is an example of running `nautilus` nested sampling with multiprocessing. As before, the likelihood and prior-transform functions must be defined at the module level:

```python
def loglikelihood(pv):
    return relic.lnlikelihood_ns(pv)
def prior_transform(uv):
    return relic.prior_transform(uv)

with Pool(relic.cfg["SAMPLER"]["npools"], maxtasksperchild=100) as pool:
    sampler, results = relic.run_nautilus(
        prior           = prior_transform,
        loglikelihood   = loglikelihood,
        pool            = pool,
    )
```

The output `results` is a dictionary containing posterior samples, log weights, and other summary statistics.

As a reference: with two JWST NIRCam datasets (2.5–5.0 μm), a retrieval with `n_live_points=240`, `npools=60`, and 10 chemical species takes roughly 10 hours.

> **Note:** We have observed a small, random memory leak in the likelihood function. For retrievals that involve millions of likelihood evaluations, set `maxtasksperchild` so worker processes are periodically recycled. `maxtasksperchild=100` is a safe default for most cases.

### 4.5 Visualize results

```python
# Before the retrieval: inspect 2D spectral light curves and limb-darkening profiles
visual.plot_2dfluxes(figname='fluxes.png')
visual.plot_ldprofiles(figname='ldprofiles.png')

# After the retrieval
samples = results['posterior_samples']
maxlike_params = results['maxlike_params']
weights = np.exp(results['log_weights'])

""" Plot posterior corner plots """
visual.plot_corners(samples=samples, weights=weights, truths=maxlike_params, figname='corners.pdf') 

""" Plot best-fit residuals """ 
visual.plot_residuals(maxlike_params, figname='residuals.png')

""" Plot transmission spectra """
visual.plot_transmission_spectra(maxlike_params, figname='transmission_spectra.png')
```
