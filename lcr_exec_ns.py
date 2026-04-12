import pickle
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
from lcr_ns import run_multinest, Priors
from lcr_plots import plot_2dfluxes, plot_corners, plot_residuals
from time import time as current_time
from pymultinest.analyse import Analyzer as Multinest_Analyzer
from sys import exit as sys_exit
from mpi4py import MPI 
comm = MPI.COMM_WORLD  

DEFAULT_CFG = 'config/HD209458b.json'
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
print_info(comm, f"Configuration file loaded: {py_args.config}")

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

print_info(comm, "Loading data: ")
[print_info(comm, f"  {f}") for f in cfg["PATH"]["input_file"]]

raw_data  = [h5py.File(f, 'r') for f in cfg["PATH"]["input_file"]]
tsdata_list = []
for i, rd in enumerate(raw_data):
    _time = rd['time'][:] + 2400000.5
    _wave = rd['wavelength'][:] 
    _flux = rd['flux'][:].T
    _ferr = rd['flux_err'][:].T 
    _flux, _ferr = replace_outliers(_time, _flux, _ferr, sigma=8)
    _name = cfg['PLANET']['name'] + f"_d{i}"
    tsdata_list.append(TSData(time=_time, wavelength=_wave, fluxes=_flux, errors=_ferr, name=_name, noise_group=i, n_baseline=2))

    _wlrange = cfg["EXOIRIS"]["WL_RANGE_MICRON"][str(i)]
    _trange  = cfg["EXOIRIS"]["TIME_RANGE_BJD"][str(i)]
    if _wlrange is not None:
        tsdata_list[-1].crop_wavelength(*_wlrange)
    if _trange is not None:
        tsdata_list[-1].crop_time(*_trange)

    print_info(comm, "Loaded dataset #{0:d} with nwl={1:d}, nt={2:d}.".format(i, *tsdata_list[i].fluxes.shape))

    tsdata_list[-1].mask_transit(t0  = cfg["PLANET"]["orb_t0_bjd"][0], 
                                 p   = cfg["PLANET"]["orb_p_d"][0], 
                                 t14 = cfg["PLANET"]["transit_duration_d"][0]) 
    tsdata_list[-1].normalize_to_poly()
    r = cfg["EXOIRIS"]["bin_resolution"]
    tsdata_list[-1] = tsdata_list[-1].bin_wavelength(r=r, estimate_errors=False)
tsdata = tsdata_list[0] + tsdata_list[1]

print_info(comm, 'Initializing LDTk model... It takes several minutes. Be patient!')
_tv, _te = cfg['STAR']['teff']
_gv, _ge = cfg['STAR']['logg']
_mv, _me = cfg['STAR']['metal']
ldmodel = LDTkLD(data=tsdata, 
                 teff=(_tv, max(_te, 50)), 
                 logg=(_gv, max(_ge, 0.02)), 
                 metal=(_mv, max(_me, 0.05)), 
                 dataset='visir')

print_info(comm, "Initializing ExoIris model...")
exoiris = ExoIris(cfg["PLANET"]["name"], ldmodel=ldmodel, data=tsdata, 
                  noise_model='white_marginalized', nthreads=1)

for k, v in cfg["PRIORS"].items():
    exoiris.set_prior(k, *v) 
exoiris.print_parameters()
exoiris._tsa.init_prt_model(atmosphere, chem, 
                            planet_radius=cfg["PLANET"]["radius_rjup"][0], 
                            star_radius=cfg["STAR"]["radius_rsun"][0]
                            )

#%% Initialize prior functions for nested sampling

prior_list = list(cfg['PRIORS'].values())
ps = Priors(prior_list)

#%% fit white light curve for systematics correction ###########################

# exoiris.fit_white()
# # update covariances with white systematics
# for i in range(len(exoiris.data)): 
#     sl = exoiris._wa.lcslices[i]
#     fm = exoiris._wa.flux_model(exoiris._wa._local_minimization.x)
#     white_systematics = exoiris._wa.ofluxa[sl] - fm[sl]
#     exoiris.data[i].covs[:, -1] = white_systematics
#     exoiris.data[i].covs[:, 1:] /= exoiris.data[i].covs[:, 1:].std(axis=0)

#%% test likelihood evaluation #################################################

if comm.Get_rank() == 0: 
    # initial_population = exoiris.ps.sample_from_prior(3)
    testcube = np.random.rand(3, len(prior_list))
    initial_population = np.array([ps.get_priors(cube) for cube in testcube])
    ll = exoiris._tsa.lnlikelihood(initial_population)
    pp = exoiris.lnposterior(initial_population)
    print("Evaluating test parameters:")
    for val in zip(ll,pp):
        print("ll={:.2f} \t\t pp={:.2f}".format(*val))
comm.Barrier()
# raise SystemExit("Test complete. Exiting.")

#%% run nested sampling ########################################################

time_start = current_time()  
run_multinest(
    LogLikelihood=exoiris._tsa.lnlikelihood,
    Prior=ps.get_priors,
    n_dims=len(prior_list),
    n_live_points=cfg['SAMPLER']['nlivepoints'],
    sampling_efficiency=cfg['SAMPLER']['ns_efficiency'],
    evidence_tolerance=cfg['SAMPLER']['ns_tolerance'],
    outputfiles_basename=cfg['PATH']['output_dir'],
    importance_nested_sampling=False,
    resume=cfg['SAMPLER']['ns_resume'],
    verbose = True,
)
time_end = current_time()  

# exit mpi after nested sampling
comm_size = comm.Get_size()
comm.Barrier()
if comm.Get_rank() != 0: 
    sys_exit(0) 

elapsed_time_str = print_elapsed_time(time_end - time_start)

#%% Saving results #############################################################

analyzer = Multinest_Analyzer(len(prior_list), 
                              outputfiles_basename = cfg['PATH']['output_dir'])
stats = analyzer.get_stats()
samples = analyzer.get_equal_weighted_posterior()[:,:-1]
results = dict(
    logZ    = stats['nested sampling global log-evidence'],
    logZerr = stats['nested sampling global log-evidence error'],
    samples = samples,
) 
results.update({
    "config": cfg,
    "parameter_names": [p.name for p in exoiris.ps],
    "process_time": elapsed_time_str,
    "processors": str(comm_size),
})

outname = os.path.join(cfg['PATH']['output_dir'], 'ns_results.pkl')
print(f'Number of posterior samples: {len(samples)}')  
print(f"Saving results to {outname}.")

with open(outname, 'wb') as f:
    pickle.dump(results, f) 
os.system('cp ' + py_args.config + ' ' + os.path.join(cfg['PATH']['output_dir']))
os.system('rm ' + os.path.join(cfg['PATH']['output_dir'], 'IS.*'))

#%% Plot limb darkening profiles ###############################################

_t = cfg['STAR']['teff'][0]
_g = cfg['STAR']['logg'][0]
_m = cfg['STAR']['metal'][0] 
fig = ldmodel.plot_profiles(teff=_t, logg=_g, metal=_m)
fig.axes[0].set_title(r'$T_{{\rm eff}}$={:.0f} K, $\log g$={:.2f}, [Fe/H]={:.2f}'.format(_t, _g, _m))
outname = os.path.join(cfg['PATH']['output_dir'], 'ldprofiles.png')
fig.savefig(outname, dpi=100)
print(f"A preview of LD profile is saved to {outname}.")


#%% Plot 2D fluxes #############################################################

plot_2dfluxes(exoiris.data, outputdir=cfg['PATH']['output_dir'])  

#%% plot posterior distributions ###############################################
 
plot_corners(samples, 
             labels=[p.name for p in exoiris.ps], 
             outputdir=cfg['PATH']['output_dir'])

#%% plot best-fit residuals ####################################################
 
median_params = np.median(samples, axis=0)
fmod = exoiris._tsa.flux_model(median_params, include_baseline=False)
plot_residuals(exoiris.data, fmod, outputdir=cfg['PATH']['output_dir']) 

print("Done!")