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
from lcr_atm import *
from lcr_white import analyze_white_lightcurve
from lcr_plots import plot_2dfluxes, plot_corners, plot_residuals
from exoiris import ExoIris, TSData
from petitRADTRANS.radtrans import Radtrans 
from petitRADTRANS.chemistry.pre_calculated_chemistry import PreCalculatedEquilibriumChemistryTable
from multiprocessing import Pool 
from functools import reduce 

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


#%% initialize TSData #########################################################

tsdata_list = []
for i, rd in enumerate(raw_data): 
    try: # specify edges for STIS and WFC3
        wl_edges = rd['wavelength_bins'][:].T 
    except KeyError:
        wl_edges = None
    tsdata_list.append(TSData(
        time        = rd['bjd_tdb'][:], 
        wavelength  = rd['wavelength'][:], 
        fluxes      = rd['flux'][:], 
        errors      = rd['flux_err'][:], 
        wl_edges    = wl_edges, 
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

#%% initialize LDTk model ######################################################
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


#%% initialize ExoIris model ###################################################

print("Initializing ExoIris model...")

exoiris = ExoIris(cfg["PLANET"]["name"], ldmodel=ldmodel, data=tsdata, 
    noise_model='white_profiled', nthreads=1)

for i, (k, v) in enumerate(cfg["PRIORS"].items()): # update priors
    exoiris.set_prior(k, *v) 

exoiris.ps[exoiris.ps.names.index("teff")].bounds  = ldmodel.sc.client.teffl
exoiris.ps[exoiris.ps.names.index("logg")].bounds  = ldmodel.sc.client.loggl 
exoiris.ps[exoiris.ps.names.index("metal")].bounds = ldmodel.sc.client.zl

# raise SystemExit("Test complete. Exiting.")

#%% initialize TS model ########################################################

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

def calculate_transmission_spectrum(atm_params: np.ndarray):
    return calc_ts_prt(
        atm_params=atm_params, 
        atmosphere=atmosphere,
        chem=chem,
        planet_radius_cm=cfg["PLANET"]["radius_rjup"][0] * r_jup_mean,
        star_radius_cm=cfg["STAR"]["radius_rsun"][0] * r_sun,
        equilibrium_temperature=cfg["PLANET"]["equilibrium_temperature"][0]
    )

exoiris._tsa.prt_wl = atmosphere.get_wavelengths() * 1e4 # A --> micron
exoiris._tsa.calculate_transmission_spectrum = calculate_transmission_spectrum
generate_binwidths(exoiris._tsa)

exoiris.print_parameters()
print("Initialization complete.")

#%% fit white light curves to validate the covariates ##########################

jitters = []
for i, rd in enumerate(raw_data):
    if "STIS" in exoiris.data[i].name: # use the first two PCs
        jitters.append(rd["detrend_vectors"][:, :2]) 
    else:
        jitters.append(None) 

with Pool(cfg["SAMPLER"]["npools"]) as pool:
    analyze_white_lightcurve(exoiris, jitters=jitters, niter=500, npop=60, pool=pool)
 
fig = exoiris.plot_white(figsize=(3 * exoiris.data.size, 8))
outname = os.path.join(cfg['PATH']['output_dir'], 'white_fit.png')
fig.tight_layout()
fig.savefig(outname, dpi=100)
print(f"A preview of white light curve fit saved as {outname}.")

#%% update covariances for spectral light curves ###############################

for i, (_t, _cov) in enumerate(zip(exoiris._wa.times, exoiris._wa.covariates)):
    newt = exoiris.data[i].time
    newcov = [np.interp(newt, _t, _c) for _c in _cov.T] 
    exoiris.data[i].covs[:] = np.array(newcov).T  
 

#%% test likelihood evaluation #################################################

print("Running a quick test of posterior evaluation...")

initial_population = exoiris.ps.sample_from_prior(3) 
pp = exoiris.lnposterior(initial_population) 
[print(f"pp = {_v:.6e}") for _v in pp]

print("Test complete.")

# raise SystemExit("Test complete. Exiting.")

#%% run DE optimization ##########################################################

print("Running Differential Evolution Optimization...")

def lnpostf(pv):
    ''' DON'T USE LAMBDA FUNCTION FOR THIS, 
    OTHERWISE IT CAUSES PICKLE ISSUES WITH MULTIPROCESSING '''
    return exoiris.lnposterior(pv) 

init_population = exoiris.ps.sample_from_prior(cfg["SAMPLER"]["nwalkers"]) 
with Pool(cfg["SAMPLER"]["npools"]) as pool: 
    exoiris.fit(
        lnpost           = lnpostf, 
        population       = init_population,  
        niter            = cfg["SAMPLER"]["niter_de"], 
        pool             = pool, 
        plot_convergence = False
    )   


#%% run MCMC sampling ##########################################################

print("Running MCMC sampling...")

with Pool(cfg["SAMPLER"]["npools"]) as pool:
    exoiris.sample(
        lnpost  = lnpostf,
        niter   = cfg["SAMPLER"]["niter_mcmc"], 
        thin    = 1, 
        pool    = pool, 
    )
exoiris.save(overwrite=True)

outname = os.path.join(cfg['PATH']['output_dir'], exoiris.name+'.fits')
shutil.move(exoiris.name+'.fits', outname)
print(f"Results saved as {outname}.")

outname = os.path.join(cfg['PATH']['output_dir'], os.path.basename(py_args.config))
shutil.copy(py_args.config, outname)
print(f"Configuration file saved as {outname}.")

#%% Plot likelihood evolutions #################################################

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


#%% plot posterior distributions ###############################################

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
