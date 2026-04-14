import os
import numba
os.environ["OMP_NUM_THREADS"] =        "1"
os.environ["OPENBLAS_NUM_THREADS"] =   "1"
os.environ["MKL_NUM_THREADS"] =        "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] =    "1"
os.environ["NUMBA_NUM_THREADS"] =      "1" 
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'  

from lcr_core import *
from lcr_plots import plot_2dfluxes, plot_corners, plot_residuals

import shutil

DEFAULT_CFG = 'config/HD209458b_mcmc.json'

if 'get_ipython' in globals():
    class Args:
        config = DEFAULT_CFG
    py_args = Args()
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', 
                        default=DEFAULT_CFG,
                        help="the input configuration file")
    py_args = parser.parse_args()
 
cfg = json.load(open(py_args.config, 'r'))  
os.makedirs(cfg["PATH"]["output_dir"], exist_ok=True)
print(f"Configuration file loaded: {py_args.config}")

#%% initialize pRT and chemical model

pressures_bar       = np.logspace(*cfg["ATMOSPHERE"]["pressure_bounds_log10bar"], 101)
wavelength_bounds   = cfg["ATMOSPHERE"]["wavelength_bounds_micron"]
species_names       = cfg["ATMOSPHERE"]["chemical_species"] 
rayleigh_species    = cfg["ATMOSPHERE"]["rayleigh_species"]
continuum_species   = cfg["ATMOSPHERE"]["continuum_species"]

atmosphere = Radtrans(
            pressures = pressures_bar,
            wavelength_boundaries = wavelength_bounds,
            line_species = species_names, 
            rayleigh_species = rayleigh_species,
            gas_continuum_contributors = continuum_species,
            line_opacity_mode = 'c-k', )
wavelengths = 1e4 * atmosphere.get_wavelengths() # from cm to micron 

chem = PreCalculatedEquilibriumChemistryTable() 

#%% initialize exoiris #########################################################

print("Loading data: ")
[print(f"  {f}") for f in cfg["PATH"]["input_file"]]

raw_data  = [h5py.File(f, 'r') for f in cfg["PATH"]["input_file"]]
tsdata_list = []
for i, rd in enumerate(raw_data):
    _time = rd['time'][:] + 2400000.5
    _wave = rd['wavelength'][:] 
    _flux = rd['flux'][:].T
    _ferr = rd['flux_err'][:].T 
    _flux, _ferr = replace_outliers(_time, _flux, _ferr, sigma=8)
    _name = cfg['PLANET']['name'] + f"_d{i}"
    tsdata_list.append(TSData(time=_time, wavelength=_wave, fluxes=_flux, errors=_ferr, name=_name, noise_group=i, n_baseline=1))

    _wlrange = cfg["EXOIRIS"]["WL_RANGE_MICRON"][str(i)]
    _trange  = cfg["EXOIRIS"]["TIME_RANGE_BJD"][str(i)]
    if _wlrange is not None:
        tsdata_list[-1].crop_wavelength(*_wlrange)
    if _trange is not None:
        tsdata_list[-1].crop_time(*_trange)
 
    print("Loaded dataset #{0:d} with nwl={1:d}, nt={2:d}.".format(i, *tsdata_list[i].fluxes.shape))

    tsdata_list[-1].mask_transit(t0  = cfg["PLANET"]["orb_t0_bjd"][0], 
                                 p   = cfg["PLANET"]["orb_p_d"][0], 
                                 t14 = cfg["PLANET"]["transit_duration_d"][0]) 
    tsdata_list[-1].normalize_to_poly()
    r = cfg["EXOIRIS"]["bin_resolution"]
    tsdata_list[-1] = tsdata_list[-1].bin_wavelength(r=r, estimate_errors=False)
tsdata = tsdata_list[0] + tsdata_list[1]
 
print('Initializing LDTk model... It takes several minutes. Be patient!')
_tv, _te = cfg['STAR']['teff']
_gv, _ge = cfg['STAR']['logg']
_mv, _me = cfg['STAR']['metal']
ldmodel = LDTkLD(data=tsdata, 
                 teff=(_tv, max(_te, 50)), 
                 logg=(_gv, max(_ge, 0.05)), 
                 metal=(_mv, max(_me, 0.05)), 
                 dataset='visir')

print("Initializing ExoIris model...")
exoiris = ExoIris(cfg["PLANET"]["name"], ldmodel=ldmodel, data=tsdata, 
                  noise_model='white_profiled', nthreads=1)

for k, v in cfg["PRIORS"].items():
    exoiris.set_prior(k, *v) 
exoiris._tsa.init_prt_model(atmosphere, chem, 
                            planet_radius=cfg["PLANET"]["radius_rjup"][0], 
                            star_radius=cfg["STAR"]["radius_rsun"][0]
                            )
exoiris.print_parameters()

#%% Generate an injection spectrum #############################################

test_params = array([
    1.04, 3.52474859, 0.5056, 2459890.20, 6090, 4.45, 0.0, 
    0.73, -1, -1, 1500, 0.5, 0.5, 1.5, 1.5
])
fmod = exoiris._tsa.flux_model(test_params, include_baseline=False)
exoiris.data[0].fluxes = fmod[0][0] + np.random.normal(np.zeros_like(exoiris.data[0].errors), 1.5*exoiris.data[0].errors)
exoiris.data[1].fluxes = fmod[1][0] + np.random.normal(np.zeros_like(exoiris.data[1].errors), 1.5*exoiris.data[1].errors)

#%% fit white light curve for systematics correction ###########################

exoiris.fit_white()
fig = exoiris.plot_white()
outname = os.path.join(cfg['PATH']['output_dir'], 'white_fit.png')
fig.savefig(outname, dpi=100)
print(f"A preview of white light curve fit saved as {outname}.")
# # update covariances with white systematics
# for i in range(len(exoiris.data)): 
#     sl = exoiris._wa.lcslices[i]
#     fm = exoiris._wa.flux_model(exoiris._wa._local_minimization.x)
#     white_systematics = exoiris._wa.ofluxa[sl] - fm[sl]
#     exoiris.data[i].covs[:, -1] = white_systematics
#     exoiris.data[i].covs[:, 1:] /= exoiris.data[i].covs[:, 1:].std(axis=0)

#%% test likelihood evaluation #################################################

initial_population = exoiris.ps.sample_from_prior(3)
ll = exoiris._tsa.lnlikelihood(initial_population)
pp = exoiris.lnposterior(initial_population)
print("Evaluating test parameters:")
for val in zip(ll,pp):
    print("ll={:.2f} \t\t pp={:.2f}".format(*val))
# raise SystemExit("Test complete. Exiting.")

#%% run DE evaluation ##########################################################

def lnpostf(pv):
    ''' DON'T USE LAMBDA FUNCTION FOR THIS, 
    OTHERWISE IT CAUSES PICKLE ISSUES WITH MULTIPROCESSING '''
    return exoiris.lnposterior(pv) 

init_population = exoiris.ps.sample_from_prior(cfg["SAMPLER"]["nwalkers"]) 
with Pool(cfg["SAMPLER"]["npools"]) as pool: 
    exoiris.fit(niter=cfg["SAMPLER"]["niter_de"], population=init_population, 
                pool=pool, lnpost=lnpostf, plot_convergence=False)   

#%% run MCMC sampling ##########################################################

with Pool(cfg["SAMPLER"]["npools"]) as pool:
    exoiris.sample(niter=cfg["SAMPLER"]["niter_mcmc"], thin=1, 
                   pool=pool, lnpost=lnpostf)

outname = os.path.join(cfg['PATH']['output_dir'], exoiris.name+'.fits')
exoiris.save(overwrite=True)
shutil.move(exoiris.name+'.fits', outname)
print(f"Results saved as {outname}.")


#%% Plot 2D fluxes #############################################################

plot_2dfluxes(exoiris.data, outputdir=cfg['PATH']['output_dir'])  

#%% Plot limb darkening profiles ###############################################

_tgm = cfg['STAR']['teff'][0], cfg['STAR']['logg'][0], cfg['STAR']['metal'][0]
_title = r'$T_{{\rm eff}}$={:.0f} K, $\log g$={:.2f}, [Fe/H]={:.2f}'.format(*_tgm)
fig = ldmodel.plot_profiles(*_tgm) 
fig.axes[0].set_title(_title)
outname = os.path.join(cfg['PATH']['output_dir'], 'ldprofiles.png') 
fig.savefig(outname, dpi=100)
print(f"A preview of LD profiles is saved to {outname}.")

#%% Plot posterior probabilities ####################################################

lnp = exoiris.sampler.get_log_prob() 
outputname = os.path.join(cfg['PATH']['output_dir'], 'lnprob.txt')
np.savetxt(outputname, lnp)
print(f"Sampling evolution saved as {outputname}.")

fig, ax = pl.subplots(1,1,figsize=(6,4))
ax.plot(lnp, c='k', lw=0.5, alpha=0.5) 
ax.set_xlabel(f'Iterations ({cfg["SAMPLER"]["nwalkers"]} walkers)')
ax.set_ylabel('Posterior probability')
fig.tight_layout()
outname = os.path.join(cfg['PATH']['output_dir'], 'lnprob.png')
fig.savefig(outname, dpi=100)
print(f"A preview of sampling evolution saved as {outname}.")

#%% plot posterior distributions

postsamples = exoiris._tsa.sampler.flatchain 
maxlike_params = postsamples[lnp.flatten().argmax()]

plot_corners(postsamples, 
             labels=[p.name for p in exoiris.ps], 
             truths=test_params,
             outputdir=cfg['PATH']['output_dir'])


#%% plot best-fit residuals
 
fmod = exoiris._tsa.flux_model(maxlike_params, include_baseline=True)
plot_residuals(exoiris.data, fmod, outputdir=cfg['PATH']['output_dir']) 

outname = os.path.join(cfg['PATH']['output_dir'], 'output.log') 
shutil.copy('output.log', outname)
print("Done!")