API Reference
=============

This reference covers the public classes, functions, and methods defined in
each ReLiC module. Frequently used objects are also available directly from
``relic``. The package's ``__all__`` lists these top-level exports. 

Core
----

Retrieval definition, prior transforms, and ExoIris integration.

* :class:`relic.core.Relic` loads a TOML configuration, initializes the data,
	atmosphere and likelihood, and creates initial diagnostic plots. Its
	``cfg``, ``atmos_model``, ``tsdata``, ``ldmodel``, ``exoiris`` and ``visual``
	attributes expose the initialized components.
* Use ``update_covariates`` to prepare baseline covariates, ``fit_white`` for
	an optional white-light fit, and ``run_de`` followed by ``run_mcmc`` for
	MCMC. ``save_mcmc`` saves the results to the configured output directory.
* ``run_nautilus`` takes a prior transform and log-likelihood callable, writes
	a checkpoint and results to the output directory, and returns the sampler and
	results dictionary. ``run_test`` draws prior samples and evaluates them.
* :class:`relic.core.Priors` transforms a unit-cube vector according to a list
	of PyTransit parameters. :class:`relic.core.RelicExoIris` connects the
	atmosphere to ExoIris, while :class:`relic.core.ResponseLDTkLD` constructs
	limb-darkening filters from spectral bins or photometric response curves.

.. automodule:: relic.core

Atmosphere
----------

Atmospheric forward models.

The built-in models combine isothermal, Guillot (2010), or Madhusudhan and
Seager (2009) temperature profiles with free, FastChem, or petitRADTRANS
equilibrium chemistry.  

Models take a configuration dictionary and are called with the complete retrieval parameter vector. Custom models should subclass :class:`relic.atmosphere.BaseAtmosphere`, set ``wavelengths`` during initialization, and implement ``__call__``.

.. automodule:: relic.atmosphere

TSLPF
-----

Spectroscopic light-curve likelihood and noise models.

:class:`relic.tslpf.NewTSLPF` evaluates transit and baseline models per
dataset. ``flux_model`` optionally includes the fitted baseline, while
``lnprior``, ``lnlikelihood`` and ``lnposterior`` operate on a single
parameter vector. ``lnlikelihood_ns`` provides the nested-sampling interface.

.. automodule:: relic.tslpf

White
-----

White-light curve likelihood and fitting.

:class:`relic.white.NewWhiteLPF` forms inverse-variance weighted white-light
curves from a spectroscopic likelihood object. 

.. automodule:: relic.white

Utils
-----

Spectral downsampling, covariates, and retrieval utilities.

* :class:`relic.utils.SpectrumDownsampler` optionally convolves a model using
	a wavelength-dependent resolving-power file and rebins it onto data bins.
	Without a resolving-power file, ``convolve_and_rebin`` only rebins.
* :class:`relic.utils.PhotometricDownsampler` integrates a model over a
	two-column wavelength and throughput response. The model grid must cover
	the response wavelength range.
* :func:`relic.utils.generate_covariates` creates Chebyshev baseline vectors.
* :func:`relic.utils.get_maxlike_estimates` selects the highest log-posterior
	MCMC sample. 

.. automodule:: relic.utils

Physics
-------

Equilibrium temperature and atmospheric temperature profiles.

:func:`relic.physics.calc_teq` returns equilibrium temperature in kelvin.
:func:`relic.physics.get_temperatures_g10` and
:func:`relic.physics.get_temperatures_m09` return temperature profiles for
pressure in bar.

.. automodule:: relic.physics

Plots
-----

Visualization of light curves and retrieval results.

:class:`relic.plots.RelicVisualization` plots white-light fits, spectral
fluxes, residuals, posterior distributions, limb-darkening profiles, and transmission spectra. 

.. automodule:: relic.plots

LDTk Patch
----------

Performance patch applied to LDTk during core import.

.. automodule:: relic.ldtk_patch
