import os
import numba
os.environ["OMP_NUM_THREADS"] =        "1"
os.environ["OPENBLAS_NUM_THREADS"] =   "1"
os.environ["MKL_NUM_THREADS"] =        "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] =    "1"
os.environ["NUMBA_NUM_THREADS"] =      "1" 
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'  

import numpy as np
import matplotlib.pyplot as pl 
import corner
from astropy.io import fits
from copy import deepcopy 
from exoiris.tslpf import TSLPF
from exoiris.ldtkld import LDTkLD
from exoiris import ExoIris, TSData
from multiprocessing import Pool 
from numpy import atleast_2d, arctan2, dstack, array, sqrt
from petitRADTRANS import physical_constants as nc  
from petitRADTRANS.radtrans import Radtrans 
from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from petitRADTRANS.physics import temperature_profile_function_guillot_global as get_tprofile
from petitRADTRANS.physics import rebin_spectrum_bin
from pytransit.orbits import as_from_rhop, i_from_ba
from pytransit.param import ParameterSet, UniformPrior as UP, NormalPrior as NP, GParameter

NM_WHITE_MARGINALIZED = 0
NM_GP_FIXED = 1
NM_GP_FREE = 2
NM_WHITE_PROFILED = 3

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
    ecc = pv[:, 3] ** 2 + pv[:, 4] ** 2
    w = arctan2(pv[:, 4], pv[:, 3])
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

def _init_parameters_new(self):
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

def _init_p_atmosphere(self):
    ps = self.ps
    pp = [GParameter('mp', 'planet_mass', 'M_jup', NP(1.0, 1e-2), (0, np.inf)),
          GParameter('ref_p', 'reference pressure', 'log10 bar', UP(-6, 2), (-np.inf, np.inf)),
          GParameter('cloud_p', 'cloud-top pressure', 'log10 bar', UP(-6, 2), (-np.inf, np.inf)),
          GParameter('tp', 'temperature', 'K', UP(300, 3000), (0, np.inf)),
          GParameter('c2o', 'C/O ratio', '', UP(0.1, 1.6), (0, np.inf)),
          GParameter('m2h', 'metallicity', 'log10 solar', UP(-1, 3), (-np.inf, np.inf))
          ]
    ps.add_global_block('atmosphere', pp)
    self._start_atm = ps.blocks[-1].start
    self._sl_atm = ps.blocks[-1].slice

def get_radius_ratios(self, pv):
    radius_ratios = []
    pv_atm = pv[:, self._sl_atm]  
    rr_model  = np.array([self.get_ts_model(atm_params) for atm_params in pv_atm])  
    for i, _d in enumerate(self.data):
        radius_ratios_rebinned = array([rebin_spectrum_bin(self.prt_wl, rr_row, self.wavelengths[i], bin_widths=self.bandwidths[i]) for rr_row in rr_model])
        radius_ratios.append(radius_ratios_rebinned)
    return atleast_2d(radius_ratios)

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
    co_ratios = np.full_like(self.prt_pbar, atm_params[4])
    metallicities = np.full_like(self.prt_pbar, atm_params[5])
    mass_fractions, mmw, nabla_ad = self.prt_chem.interpolate_mass_fractions(
    co_ratios=co_ratios,
    log10_metallicities=metallicities,
    temperatures=temperatures,
    pressures=self.prt_pbar,
    full=True
    )
    mass_fractions['CO-NatAbund'] = mass_fractions.pop('CO')

    # calculate the transmission spectrum
    _, transit_radii, _ = self.prt_atmosphere.calculate_transit_radii(
    temperatures=temperatures,
    mass_fractions=mass_fractions,
    mean_molar_masses=mmw,
    reference_gravity=ref_gravity,
    planet_radius=self.planet_radius,
    reference_pressure=ref_pressure,
    opaque_cloud_top_pressure=cloudtop_pbar,
    ) 
    radius_ratios = transit_radii/self.star_radius
    return radius_ratios
    
def generate_bandwidths(self):
    self.bandwidths = []
    for wl in self.wavelengths: 
        dwl = np.zeros_like(wl)
        dwl[:-1] = np.diff(wl)
        dwl[-1] = dwl[-2] 
        self.bandwidths.append(dwl)
    return self.bandwidths

TSLPF.transit_model = custom_transit_model
TSLPF.init_prt_model = init_prt_model
TSLPF.get_ts_model = get_ts_model
TSLPF._init_parameters = _init_parameters_new
TSLPF._init_p_atmosphere = _init_p_atmosphere
TSLPF.get_radius_ratios = get_radius_ratios
TSLPF.generate_bandwidths = generate_bandwidths

#%% initialize pRT and chemical model

species_names = ['H2O', 'CO-NatAbund', 'CO2', 'CH4'] 
co_ratio = 0.55
metallicity = 0

atmosphere = Radtrans(
            pressures = np.logspace(-6, 2, 120),
            line_species = species_names, 
            rayleigh_species = ['H2', 'He'],
            gas_continuum_contributors = ['H2-H2', 'H2-He'],
            wavelength_boundaries = [0.95, 5.1],
            line_opacity_mode = 'c-k', )
wavelengths = 1e4 * atmosphere.get_wavelengths() # from cm to micron
pressures_bar = 1e-6*atmosphere.pressures # from cgs to bar 

chem = PreCalculatedEquilibriumChemistryTable()
co_ratios = np.full_like(pressures_bar, co_ratio)
metallicities = np.full_like(pressures_bar, metallicity)

#%% initialize exoiris 

time = np.linspace(-0.1, 0.1, 1900) + 2450000.5
fake_wavelengths = np.linspace(1, 5, 135)
fake_fluxes = np.ones([len(fake_wavelengths), len(time)])
fake_errors = 1000e-6 * np.ones_like(fake_fluxes)
fake_fluxes += np.random.normal(0, fake_errors, size=fake_fluxes.shape)
fake_data = TSData(time=time, wavelength=fake_wavelengths, fluxes=fake_fluxes, errors=fake_errors, name='fake_data', noise_group=0, n_baseline=1)
fake_data.mask_transit(t0=2450000.5, p=3.0, t14=0.1)

ldmodel = 'power-2'
print('Initializing LDTk model... It takes several minutes, be patient!')
ldmodel = LDTkLD(data=fake_data, teff=(5772, 100), logg=(4.44, 0.05), metal=(0.0, 0.05), dataset='visir')

print("Initializing ExoIris model...")
exoiris = ExoIris('test', ldmodel=ldmodel, noise_model='white_marginalized', data=fake_data, nk=50, nldc=10, nthreads=1)

exoiris.set_prior('rho', 'NP', 1.4, 0.07)
exoiris.set_prior('p', 'NP', 3.0, 1e-6)
exoiris.set_prior('b', 'NP', 0.10, 0.005)
exoiris.set_prior('secw', 'NP', 0, 1e-5)
exoiris.set_prior('sesw', 'NP', 0, 1e-5)
exoiris.set_prior('tc_00', 'NP', 2450000.5, 1e-4)
exoiris.set_prior('mp', 'NP', 0.8, 0.04)
exoiris.set_prior('ref_p', 'UP', -6, 2)
exoiris.set_prior('cloud_p', 'UP', -6, 2)
exoiris.set_prior('tp', 'UP', 300, 3000)
exoiris.set_prior('c2o', 'UP', 0.1, 1.6)
exoiris.set_prior('m2h', 'UP', -1, 3)
exoiris.set_prior('sigma_m_00', 'UP', 0.5, 5)
exoiris.print_parameters()

#%% update synthetic data

exoiris._tsa.init_prt_model(atmosphere, chem, planet_radius=1.5, star_radius=1.0)
exoiris._tsa.generate_bandwidths()  
pv_inject = np.array([[1.4, 3.0, 0.1, 0, 0, 2450000.5, 
                     5772.0, 4.44, 0.0,
                     0.8, -1, -1, 1500, 0.55, 0, 
                     2.0]])
fluxes_model = exoiris._tsa.flux_model(pv_inject, include_baseline=False)[0][0]
fluxes_withnoise = fluxes_model + np.random.normal(0, 2*fake_errors)

synthetic_data = TSData(time=time, wavelength=fake_wavelengths, fluxes=fluxes_withnoise, errors=fake_errors, name='synthetic', noise_group=0, n_baseline=1)
synthetic_data.mask_transit(t0=2450000.5, p=3.0, t14=0.1)
exoiris.set_data(synthetic_data)

#%% save synthetic data

fig, ax = pl.subplots(2,1, figsize=(6,6))
ax[0].imshow(fluxes_model, aspect='auto', origin='lower', extent=[fake_data.time.min(), fake_data.time.max(), fake_data.wavelength.min(), fake_data.wavelength.max()])
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Wavelength (micron)')
fig.colorbar(ax[0].images[0], ax=ax[0], label='Flux')
ax[1].imshow(fluxes_withnoise, aspect='auto', origin='lower', extent=[fake_data.time.min(), fake_data.time.max(), fake_data.wavelength.min(), fake_data.wavelength.max()])
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Wavelength (micron)')
fig.colorbar(ax[1].images[0], ax=ax[1], label='Flux')
fig.tight_layout()
fig.savefig('synthetic_data.png', dpi=150)

#%% test likelihood evaluation 

initial_population = exoiris.ps.sample_from_prior(3)
ll = exoiris._tsa.lnlikelihood(initial_population)
pp = exoiris.lnposterior(initial_population)
print("Evaluating test parameter vectors:")
for val in zip(ll,pp):
    print("ll=%.2f \t\t pp=%.2f" % (val[0], val[1]))

#%% run DE evaluation

npools = 16
nchains = 4 * npools
niter_de = 100
niter_mcmc = 1000

def lnpostf(pv):
    ''' DON'T USE LAMBDA FUNCTION FOR THIS, 
    OTHERWISE IT CAUSES PICKLE ISSUES WITH MULTIPROCESSING '''
    return exoiris.lnposterior(pv) 

init_population = exoiris.ps.sample_from_prior(nchains) 

outputfile = exoiris.name + '.fits' 
with Pool(npools) as pool: 
    exoiris.fit(niter=niter_de, population=init_population, pool=pool, lnpost=lnpostf, plot_convergence=False) 
    de_population = exoiris.de.population
exoiris.save(overwrite=True)
exoiris._tsa._de_population = de_population.copy()

#%% run MCMC sampling

with Pool(npools) as pool:
    exoiris.sample(niter=niter_mcmc, thin=1, pool=pool, lnpost=lnpostf)
exoiris.save(overwrite=True)
print(f"Results saved as {outputfile}.")

#%% Posterior probabilities
lnp = exoiris.sampler.get_log_prob()
np.savetxt(exoiris.name + '_lnprob.txt', lnp)
print(f"Log-probabilities saved as {exoiris.name}_lnprob.txt.")

fig, ax = pl.subplots(1,1,figsize=(6,4))
ax.plot(lnp, c='k', lw=0.5, alpha=0.5) 
ax.set_xlabel(f'Steps ({lnp.shape[1]} walkers)')
ax.set_ylabel('post-probability')
fig.tight_layout()
fig.savefig('lnprob_convergence.png', dpi=150)
print("Figure saved as lnprob_convergence.png.")

#%% plot posterior distributions

post_samples = exoiris._tsa.sampler.flatchain
labels = [p.name for p in exoiris.ps]

fig = corner.corner(
    post_samples, labels=labels, 
    truths=pv_inject.flatten(),
    show_titles=True, title_fmt='.4g',
    plot_datapoints=False, plot_density=True,
    range=0.999*np.ones(post_samples.shape[1]),
    levels=[0.3935, 0.8647, 0.9889], 
    quantiles=[0.16, 0.5, 0.84])
fig.savefig('corner_plot.pdf') 
print("Figure saved as corner_plot.pdf")

print("Done!")