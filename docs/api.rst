API Reference
=============

This reference covers the public classes, functions, and methods defined in
each ReLiC module. 

Core
----

Retrieval definition, prior transforms, and ExoIris integration.

* :class:`relic.core.Relic` 
    loads a TOML configuration, initializes the data, atmosphere and likelihood, and creates initial diagnostic plots. 
* :class:`relic.core.Priors` 
    transforms a unit-cube vector to physical parameter values according to the prior definitions.
* :func:`relic.core.Relic.update_covariates` 
    prepares baseline covariates, 
* :func:`relic.core.Relic.fit_white`
    jointly fits the white light curves from each data set.
* :func:`relic.core.Relic.run_test` 
    randomly draws prior samples and evaluates the likelihood function. 
* :func:`relic.core.Relic.run_nautilus` 
    runs nested sampling using the Nautilus algorithm.
* :func:`relic.core.Relic.run_emcee` 
    runs MCMC sampling using the emcee algorithm. 

.. automodule:: relic.core

Atmosphere
----------

Forward models of 1D transmission spectrum.

The built-in models combine isothermal, Guillot (2010), or Madhusudhan and
Seager (2009) temperature profiles with free, FastChem, or petitRADTRANS
equilibrium chemistry.  

Custom models should subclass :class:`relic.atmosphere.BaseAtmosphere`, set ``wavelengths`` during initialization, and implement ``__call__``.

.. automodule:: relic.atmosphere

TSLPF
-----

Spectroscopic light-curve likelihood and noise models.

:class:`relic.tslpf.NewTSLPF` 
    evaluates transit and baseline models per dataset. 

.. automodule:: relic.tslpf

White
-----

White-light curve likelihood and fitting.

:class:`relic.white.NewWhiteLPF` 
    is used for white-light curve fitting. 

.. automodule:: relic.white

Utils
-----

Spectral downsampling, covariates, and retrieval utilities.

* :class:`relic.utils.SpectrumDownsampler` 
    convolves a transmission spectrum using a wavelength-dependent resolving-power file and rebins it onto data wavelength bins. 
* :class:`relic.utils.PhotometricDownsampler` 
    integrates a model over a two-column wavelength and throughput response. The model grid must cover this window.
* :func:`relic.utils.generate_covariates` 
    creates Chebyshev baseline vectors.
* :func:`relic.utils.get_maxlike_estimates` 
    selects the highest log-posterior MCMC sample. 

.. automodule:: relic.utils

Physics
-------

Equilibrium temperature and atmospheric temperature profiles.

* :func:`relic.physics.calc_teq` 
	returns equilibrium temperature in kelvin.
* :func:`relic.physics.get_temperatures_g10` and :func:`relic.physics.get_temperatures_m09` 
  	return temperature profiles for pressure in bar.

.. automodule:: relic.physics

Plots
-----

Visualization of light curves and retrieval results.

* :class:`relic.plots.RelicVisualization` 
    plots white-light fits, spectral fluxes, residuals, posterior distributions, limb-darkening profiles, and transmission spectra. 

.. automodule:: relic.plots

LDTk Patch
----------

Performance patch applied to LDTk during core import.

.. automodule:: relic.ldtk_patch
