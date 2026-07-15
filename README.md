# ReLiC: atmospheric **Re**trievals using spectral **Li**ght **C**urves

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A Python package for low-to-medium-resolution atmospheric retrieval of transiting exoplanets using spectral light curves. 

ReLiC integrates four components into a unified Bayesian inference framework:
1. **Atmospheric model** — computes a transmission or emission spectrum given planetary atmospheric paramters. 
2. **Stellar model** — computes limb-darkening profiles (for transits using [LDTk](https://github.com/hpparvi/ldtk)) or stellar fluxes (for eclipses) given stellar parameters.
3. **Transit/eclipse model** — computes spectral light curves from transmission/emission spectra given orbital parameters using [PyTransit](https://github.com/hpparvi/PyTransit) and [ExoIris](https://github.com/hpparvi/ExoIris).
4. **Noise model** — handles systematic noise (covariate baselines, Gaussian Processes, and/or jitters) per dataset.

The likelihood function is built upon jointly fitting all spectral lightcurve of all datasets.

Advantages of ReLiC compared with traditional retrievals using transmission/emission spectra:

1. **Scalable** — Transit/eclipse depths are no longer free parameters in lightcurve fitting. As the number of observations increases, the dimension of the parameter space grows very slowly.
2. **Physics-driven** — The chromatic transit depths of the spectral light curves are directly derived from the atmospheric model. The limb-darkening effect of transit shapes is directly modeled by stellar templates instead of parameterized approximation. 
3. **Pixel-resolution oriented** — Since the parameter space is robust against the data volume, pixel-resolution retrieval has no downside other than a slight increase in computational cost. 
3. **Compatible with high-resolution retrievals** — Spectral light curves are essentially time-series spectra. Therefore, for pixel-resolution retrieval, the likelihood definition is equivalent to that in high-resolution retrievals. Combining low- and high-resolution retrievals simply involves adding their likelihood functions without assigning extra weights.

**Currently, ReLiC is still under development.** Transmission spectroscopy is ready to go. Other missing functionality (emission spectroscopy and uniform output formats) will be released in the next major update.

---

## Installation

It's recommended to create a new conda environment with Python=3.12.
```bash
conda create --name relic python=3.12
conda activate relic
```
Install the package and all dependencies in development mode:

```bash
git clone https://github.com/cjiang-exo/ReLiC.git
cd ReLiC
conda install -c conda-forge openmpi
pip install -e .
```

---

## Usage and examples

Basic usage:
```python
from relic import Relic

relic = Relic('input_config.toml')

def loglikelihood(pv):
    return relic.lnlikelihood_ns(pv)
def prior_transform(uv):
    return relic.prior_transform(uv)

results = relic.run_dynesty(
    loglikelihood   = loglikelihood,
    prior_transform = prior_transform, 
    nlivepoints     = 100, 
) 

```

An example retrieval script can be found: [`example_scripts/pipeline_ns.py`](example_scripts/pipeline_ns.py)

See [ReLiC documentation](https://relic.readthedocs.io/en/latest/quickstart.html) for details.

---

## Citation

Our ReLiC paper is going to be submitted.

The ReLiC code is primarily built upon [ExoIris](https://github.com/hpparvi/ExoIris), [PyTransit](https://github.com/hpparvi/PyTransit), and [LDTk](https://github.com/hpparvi/ldtk) developed by Hannu Parviainen. Thus, please also consider citing the following papers:

```latex

@ARTICLE{2026AJ....171..237P,
       author = {{Parviainen}, Hannu},
        title = "{ExoIris: Fast Exoplanet Transmission Spectroscopy in Python}",
      journal = {\aj}, 
         year = 2026, 
       volume = {171}, 
        pages = {237},
          doi = {10.3847/1538-3881/ae47ee},   
}

@ARTICLE{2015MNRAS.450.3233P,
       author = {{Parviainen}, Hannu},
        title = "{PYTRANSIT: fast and easy exoplanet transit modelling in PYTHON}",
      journal = {\mnras}, 
         year = 2015, 
       volume = {450}, 
        pages = {3233-3238},
          doi = {10.1093/mnras/stv894}, 
}

@ARTICLE{2015MNRAS.453.3821P,
       author = {{Parviainen}, H. and {Aigrain}, S.},
        title = "{LDTK: Limb Darkening Toolkit}",
      journal = {\mnras}, 
         year = 2015, 
       volume = {453}, 
        pages = {3821-3826},
          doi = {10.1093/mnras/stv1857}, 
}


```

---

