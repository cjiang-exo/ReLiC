#%% 
import os
import numba
os.environ["OMP_NUM_THREADS"] =        "1"
os.environ["OPENBLAS_NUM_THREADS"] =   "1"
os.environ["MKL_NUM_THREADS"] =        "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] =    "1"
os.environ["NUMBA_NUM_THREADS"] =      "1" 
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'  

import argparse
import tomllib
import h5py
import shutil
from lcr_core import *
from lcr_plots import plot_2dfluxes, plot_corners, plot_residuals
from multiprocessing import Pool 
from functools import reduce
from numpy.polynomial import Chebyshev
from numpy.linalg import lstsq

DEFAULT_CFG = 'config/HD209458b_joint.toml'

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

cfg = tomllib.load(open(py_args.config, 'rb'))
os.makedirs(cfg["PATH"]["output_dir"], exist_ok=True)
print(f"Configuration file loaded: {py_args.config}")

#%% load data from input files

print("Loading data: ")
[print(f"  {f}") for f in cfg["PATH"]["input_file"]]

raw_data  = [h5py.File(f, 'r') for f in cfg["PATH"]["input_file"]]

# raise SystemExit("Configuration loaded. Exiting for testing purposes.")

#%% initialize TSData #########################################################

tsdata_list = []
for i, rd in enumerate(raw_data): 
    try:
        wl_edges = rd['wavelength_bins'][:].T 
    except KeyError:
        wl_edges = None
    tsdata_list.append(TSData(
        time        = rd['bjd_tdb'][:], 
        wavelength  = rd['wavelength'][:], 
        fluxes      = rd['flux'][:], 
        errors      = rd['flux_err'][:], 
        wl_edges    = wl_edges, # specify edges for STIS and WFC3
        name        = cfg['EXOIRIS']['INSTRUMENT'][str(i)] + f"_d{i}", 
        noise_group = cfg["EXOIRIS"]["NOISE_GROUP"][str(i)], 
        n_baseline  = cfg['EXOIRIS']['N_BASELINE'][str(i)],
    ))

    _wlrange = cfg["EXOIRIS"]["WL_RANGE_MICRON"][str(i)]
    _trange  = cfg["EXOIRIS"]["TIME_RANGE_BJD"][str(i)]
    if _wlrange != []:
        tsdata_list[-1].crop_wavelength(*_wlrange)
    if _trange != []:
        tsdata_list[-1].crop_time(*_trange)
 
    print("Loaded dataset #{0:d} with nwl={1:d}, nt={2:d}.".format(
        i, *tsdata_list[i].fluxes.shape
    ))

    tsdata_list[-1].mask_transit(
        t0  = cfg["PLANET"]["orb_t0_bjd"][0], 
        p   = cfg["PLANET"]["orb_p_d"][0], 
        t14 = cfg["PLANET"]["transit_duration_d"][0]
    ) 
    tsdata_list[-1].normalize_to_poly()
    tsdata_list[-1].mask_outliers(sigma=8.0)
    r = cfg["EXOIRIS"]["bin_resolution"]
    if ("JWST" in cfg["EXOIRIS"]["INSTRUMENT"][str(i)]) & (r > 0):
        tsdata_list[-1] = tsdata_list[-1].bin_wavelength(
            r=r, estimate_errors=False
        )
        print(f"Wavelength binning applied with resolution R={r}. New nwl={tsdata_list[-1].fluxes.shape[0]}.")
    else:
        print("No wavelength binning applied. Running retrievals on native resolution.")

tsdata = reduce(lambda x,y: x+y, tsdata_list)
del tsdata_list

#%% initialize pRT and chemical model
print("Initializing atmospheric model...")

atmosphere = Radtrans(
    pressures = np.logspace(*cfg["ATMOSPHERE"]["pressure_bounds_log10bar"], 121),
    wavelength_boundaries       = cfg["ATMOSPHERE"]["wavelength_bounds_micron"],
    line_species                = cfg["ATMOSPHERE"]["chemical_species"], 
    rayleigh_species            = cfg["ATMOSPHERE"]["rayleigh_species"],
    gas_continuum_contributors  = cfg["ATMOSPHERE"]["continuum_species"],
    line_opacity_mode           = cfg["ATMOSPHERE"]["opacity_mode"], 
)

chem = PreCalculatedEquilibriumChemistryTable() 

#%% initialize LDTk model
print('Initializing LDTk model... It takes several minutes. Be patient!')

_tv, _te = cfg['STAR']['teff']
_gv, _ge = cfg['STAR']['logg']
_mv, _me = cfg['STAR']['metal']
ldmodel = LDTkLD(
    data=tsdata, 
    teff=(_tv, max(_te, 50)), 
    logg=(_gv, max(_ge, 0.05)), 
    metal=(_mv, max(_me, 0.05)), 
    dataset='visir'
)

#%% initialize ExoIris model
print("Initializing ExoIris model...")

exoiris = ExoIris(cfg["PLANET"]["name"], ldmodel=ldmodel, data=tsdata, 
    noise_model='white_profiled', nthreads=1)

for k, v in cfg["PRIORS"].items():
    exoiris.set_prior(k, *v) 

exoiris._tsa.init_prt_model(
    atmosphere, chem, 
    planet_radius=cfg["PLANET"]["radius_rjup"][0], 
    star_radius=cfg["STAR"]["radius_rsun"][0],
    equilibrium_temperature=cfg["PLANET"]["equilibrium_temperature"][0]
)

exoiris.print_parameters()

#%% Update covariates for baseline detrending

# period_hst = 95.42 / 1440.0 # [days]
# x = lambda phi: 2 * (phi - phi.min()) / (phi.max() - phi.min()) - 1



# for i, d in enumerate(exoiris.data): 
#     if "STIS" in cfg["EXOIRIS"]["INSTRUMENT"][str(i)]:
#         t = d.time
#         f = d.fluxes[0]
#         fe = d.errors[0]
#         _mask = d.transit_mask & isfinite(f) & isfinite(fe)
#         _covs = raw_data[i]["detrend_vectors"][:, 0]
#     elif "WFC3" in cfg["EXOIRIS"]["INSTRUMENT"][str(i)]:
#         _covs = raw_data[i]["sky_background_level_array"][:]**-1
#     else:
#         continue # keep default for JWST
#     exoiris.data[i].covs[:, -1] = _covs
#     exoiris.data[i].covs[:, 1:] -= exoiris.data[i].covs[:, 1:].mean(axis=0)
#     exoiris.data[i].covs[:, 1:] /= exoiris.data[i].covs[:, 1:].std(axis=0)

#%% STIS

# period_hst = 95.42/1440.0 # [days]
# x = lambda phi: 2 * (phi - phi.min()) / (phi.max() - phi.min()) - 1

# t = exoiris.data[2].time
# f = exoiris.data[2].fluxes[8]
# fe = exoiris.data[2].errors[8]
# _mask = exoiris.data[2].transit_mask & isfinite(f) & isfinite(fe)

# phases = (t - t[0]) % period_hst # folded phases  
# phases[phases > 0.7*period_hst] -= period_hst
# tsub = (t - t.mean()) / t.std() 

# covariates = array([Chebyshev.basis(i)(x(phases)) for i in range(5)]).T
# covariates = np.hstack((covariates, raw_data[2]['detrend_vectors'][:])) 
# covariates = covariates[_mask]

# x_sim = x(phases)[_mask]
# y_sim = f[_mask]
# ye_sim = fe[_mask]
 
# coeffs = lstsq(covariates, y_sim)[0]
# y_fit = (covariates @ coeffs).T

# reduced_chisq = sum(((y_sim - y_fit) / ye_sim)**2) / (len(y_sim) - covariates.shape[1])
# print(f"LSQ fit reduced chi-squared: {reduced_chisq:.2f}")

# pl.errorbar(phases[_mask], y_sim, yerr=ye_sim, fmt='.k')
# pl.plot(phases[_mask], y_fit, '.r')
# pl.show()


# #%% WFC3 

# t = exoiris.data[4].time
# f = exoiris.data[4].fluxes[9]
# fe = exoiris.data[4].errors[9]
# _mask = exoiris.data[4].transit_mask

# phases = (t - t[0]) % period_hst # folded phases  
# phases[phases > 0.7*period_hst] -= period_hst
# tsub = (t - t.mean()) / t.std() 

# covariates = array([Chebyshev.basis(i)(x(phases)) for i in range(5)])   
# covariates = covariates[:, _mask].T

# x_sim = x(phases)[_mask]
# y_sim = f[_mask]
# ye_sim = fe[_mask]
 
# coeffs = lstsq(covariates, y_sim)[0]
# y_fit = (covariates @ coeffs).T

# reduced_chisq = sum(((y_sim - y_fit) / ye_sim)**2) / (len(y_sim) - covariates.shape[1])
# print(f"LSQ fit reduced chi-squared: {reduced_chisq:.2f}")

# pl.errorbar(phases[_mask], y_sim, yerr=ye_sim, fmt='.k')
# pl.plot(phases[_mask], y_fit, '.r')
# pl.show()

# raise SystemExit("Configuration loaded. Exiting for testing purposes.")

#%% fit white light curve for systematics correction ###########################

jitters = []
for i, rd in enumerate(raw_data):
    if "STIS" in exoiris.data[i].name: # use the first two PCs
        jitters.append(rd["detrend_vectors"][:, :2]) 
    else:
        jitters.append(None) 

analyze_white_lightcurve(exoiris, jitters=jitters)

# exoiris.custom_fit_white(jitters=jitters)


#%%

initial_population = exoiris._wa.ps.sample_from_prior(3)
fmodel = exoiris._wa.flux_model(initial_population) 

fig, ax = pl.subplots(1, len(exoiris.data), figsize=(3*len(exoiris.data), 4), sharey=True)
for i, sl in enumerate(exoiris._wa.lcslices):
    ax[i].errorbar(exoiris._wa.times[i], exoiris._wa.fluxes[i], 
        yerr=exoiris._wa.errors[i], fmt='.k')
    ax[i].plot(exoiris._wa.times[i], fmodel[0, sl], '.r')
    ax[i].set_title(cfg["EXOIRIS"]["INSTRUMENT"][str(i)]) 
    ax[i].set_xlabel('Time')

ax[0].set_ylabel('Normalized flux')
fig.tight_layout()
raise SystemExit("Configuration loaded. Exiting for testing purposes.")

#%%
_ncol = exoiris.data.size
fig = exoiris.plot_white(figsize=(3*_ncol, 4), ncols=_ncol)
outname = os.path.join(cfg['PATH']['output_dir'], 'white_fit.png')
fig.tight_layout()
fig.savefig(outname, dpi=100)
print(f"A preview of white light curve fit saved as {outname}.")

# update covariances with white systematics
for i in range(len(exoiris.data)): 
    if "STIS" in cfg["EXOIRIS"]["INSTRUMENT"][str(i)]:
        _systematics = raw_data[i]["detrend_vectors"][:, 0]
    elif "WFC3" in cfg["EXOIRIS"]["INSTRUMENT"][str(i)]:
        _systematics = raw_data[i]["sky_background_level_array"][:]**-1
    elif "JWST" in cfg["EXOIRIS"]["INSTRUMENT"][str(i)]:
        sl = exoiris._wa.lcslices[i]
        fm = exoiris._wa.flux_model(exoiris._wa._local_minimization.x)
        _systematics = exoiris._wa.ofluxa[sl] - fm[sl]
    exoiris.data[i].covs[:, -1] = _systematics
    exoiris.data[i].covs[:, 1:] -= exoiris.data[i].covs[:, 1:].mean(axis=0)
    exoiris.data[i].covs[:, 1:] /= exoiris.data[i].covs[:, 1:].std(axis=0)

#%% test likelihood evaluation #################################################

initial_population = exoiris.ps.sample_from_prior(3)
ll = exoiris._tsa.lnlikelihood(initial_population)
pp = exoiris.lnposterior(initial_population)
print("Evaluating test parameters:")
for val in zip(ll, pp):
    print("ll={:.2f} \t\t pp={:.2f}".format(*val))
# raise SystemExit("Test complete. Exiting.")

#%% run DE evaluation ##########################################################

def lnpostf(pv):
    ''' DON'T USE LAMBDA FUNCTION FOR THIS, 
    OTHERWISE IT CAUSES PICKLE ISSUES WITH MULTIPROCESSING '''
    return exoiris.lnposterior(pv) 

init_population = exoiris.ps.sample_from_prior(cfg["SAMPLER"]["nwalkers"]) 
with Pool(cfg["SAMPLER"]["npools"]) as pool: 
    exoiris.fit(
        population       = init_population, 
        lnpost           = lnpostf, 
        niter            = cfg["SAMPLER"]["niter_de"], 
        pool             = pool, 
        plot_convergence = False
    )   


#%% run MCMC sampling ##########################################################

with Pool(cfg["SAMPLER"]["npools"]) as pool:
    exoiris.sample(niter=cfg["SAMPLER"]["niter_mcmc"], 
        thin=1, pool=pool, lnpost=lnpostf)

outname = os.path.join(cfg['PATH']['output_dir'], exoiris.name+'.fits')
exoiris.save(overwrite=True)
shutil.move(exoiris.name+'.fits', outname)
print(f"Results saved as {outname}.")

outname = os.path.join(cfg['PATH']['output_dir'], os.path.basename(py_args.config))
shutil.copy(py_args.config, outname)
print(f"Configuration file saved as {outname}.")

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

#%% Plot 2D fluxes #############################################################

plot_2dfluxes(exoiris.data, outputdir=cfg['PATH']['output_dir'])  

#%% Plot limb darkening profiles ###############################################

_tgm = cfg['STAR']['teff'][0], cfg['STAR']['logg'][0], cfg['STAR']['metal'][0]
_title = r'$T_{{\rm eff}}$={:.0f} K, $\log g$={:.2f}, [Fe/H]={:.2f}'.format(*_tgm)
outname = os.path.join(cfg['PATH']['output_dir'], 'ldprofiles.png') 

fig = ldmodel.plot_profiles(*_tgm) 
fig.axes[0].set_title(_title)
fig.tight_layout()
fig.savefig(outname, dpi=100)
print(f"A preview of LD profiles is saved to {outname}.")


#%% plot posterior distributions

postsamples = exoiris._tsa.sampler.flatchain 
maxlike_params = postsamples[lnp.flatten().argmax()]

plot_corners(postsamples, labels=[p.name for p in exoiris.ps],
    truths=maxlike_params, outputdir=cfg['PATH']['output_dir'])

#%% plot best-fit residuals
 
fmod = exoiris._tsa.flux_model(maxlike_params, include_baseline=True)
plot_residuals(exoiris.data, fmod, outputdir=cfg['PATH']['output_dir']) 

outname = os.path.join(cfg['PATH']['output_dir'], 'output.log') 
shutil.copy('output.log', outname)
print("Done!")
# %%
