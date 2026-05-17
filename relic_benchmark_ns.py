#%% 
import os
import pickle
os.environ["OMP_NUM_THREADS"] =        "1"
os.environ["OPENBLAS_NUM_THREADS"] =   "1"
os.environ["MKL_NUM_THREADS"] =        "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] =    "1"
os.environ["NUMBA_NUM_THREADS"] =      "1" 
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'   

import argparse
import numpy as np
import tomllib    

from relic_core import ReLic, Priors
from relic_atmosphere import TP6EqChem as AtmosModel
from relic_plots import *
from relic_utils import generate_covariates, get_maxlike_estimates, print_elapsed_time

from multiprocessing import Pool   
from numpy import inf
from time import time as current_time
from pymultinest.analyse import Analyzer as Multinest_Analyzer
from sys import exit as sys_exit
from mpi4py import MPI 
comm = MPI.COMM_WORLD  

DEFAULT_CFG = 'config/HD209458b_benchmark-r100.toml'

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


#%% Initialization #############################################################

atmos_model = AtmosModel(cfg) # user-defined

relic = ReLic(atmos_model)

def additional_priors(pv):
    p1, p2, p3 = pv[16:19]
    if (p1 > p3) or (p2 > p3):
        return -inf
    return 0.0

relic.exoiris._tsa.additional_priors = additional_priors

nsprior = Priors(relic.exoiris._tsa.ps)

print("Initialization complete.") 

#%% inject a truth spectrum for testing

truth_pv = np.array([
    1.0, 3.5, 0.5, 0.00, 6000.0, 4.5, 0.0, 1.5, 1.5,
    0.8, -2, -1, 0.5, 1400, 0, -0.3, -3, -3, 0, 0.0, 0.5
])

rng = np.random.default_rng(42)
truth_fmod = relic.exoiris._tsa.flux_model(truth_pv, include_baseline=False)
for i in range(relic.exoiris.data.size):
    _errors = relic.exoiris.data[i].errors
    relic.exoiris.data[i].fluxes = truth_fmod[i] + rng.normal(np.zeros_like(_errors), 1.5*_errors)

#%% fit white light curves to validate the covariates ##########################
 
jitters = []
for i, rd in enumerate(relic.raw_data):
    if "STIS" in relic.exoiris.data[i].name: # use the first two PCs
        jitters.append(rd["detrend_vectors"][:, :2]) 
    else:
        jitters.append(None) 

covariates = generate_covariates(relic, jitters)

# with Pool(cfg["SAMPLER"]["npools"]) as pool:
relic.fit_white(covariates=covariates, pool=None)

# relic.update_covariates()

if comm.Get_rank() == 0:
    plot_white(relic)
comm.Barrier()


#%% test likelihood evaluation #################################################

if comm.Get_rank() == 0:

    print("Running a quick test of posterior evaluation...")

    initial_population = relic.sample_from_prior(1)
    pp = relic.lnposterior(initial_population)
    print(f"lnprob = {pp:.6e}")

    initial_population = relic.sample_from_prior(3)
    pp = [relic.lnposterior(_p) for _p in initial_population]
    [print(f"lnprob = {_v:.6e}") for _v in pp]

    print("Test complete.")
comm.Barrier()

#%% run multinest sampling #####################################################

time_start = current_time()  
relic.run_multinest(
    LogLikelihood       = relic.lnlikelihood_ns,
    Prior               = nsprior.prior_transform,
    n_live_points       = cfg["SAMPLER"]["n_live_points"],
    evidence_tolerance  = cfg["SAMPLER"]["evidence_tolerance"],
    sampling_efficiency = cfg["SAMPLER"]["sampling_efficiency"],
)
time_end = current_time()  

# exit mpi after nested sampling
comm_size = comm.Get_size()
comm.Barrier()
if comm.Get_rank() != 0: 
    sys_exit(0) 

elapsed_time_str = print_elapsed_time(time_end - time_start)

#%% Post analysis and saving results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

analyzer = Multinest_Analyzer(len(relic.exoiris.ps), 
    outputfiles_basename = cfg['PATH']['output_dir'])
stats = analyzer.get_stats()

samples = analyzer.get_equal_weighted_posterior()[:,:-1]
results = dict(
    logZ    = stats['nested sampling global log-evidence'],
    logZerr = stats['nested sampling global log-evidence error'],
    maxlike = stats['modes'][0]["maximum"],
    samples = samples,
) 
results.update({
    "config": cfg,
    "parameter_names": [p.name for p in relic.exoiris.ps],
    "process_time": elapsed_time_str,
    "processors": str(comm_size),
})

outname = os.path.join(cfg['PATH']['output_dir'], 'ns_results.pkl')
print(f'Number of posterior samples: {len(samples)}')  
print(f"Saving results to {outname}.")

with open(outname, 'wb') as f:
    pickle.dump(results, f) 
os.system('cp ' + py_args.config + ' ' + os.path.join(cfg['PATH']['output_dir']))

#%% test tp6
 
# temp = atmos_model.tp6madhu(atmos_model.pressures_bar, 1400, 1, 0.5, -3, -3, 0)

# fig, ax  = pl.subplots(figsize=(6,4))
# ax.plot(temp, relic.atmos_model.pressures_bar)
# ax.set_xlabel("Temperature (K)")
# ax.set_yscale("log")
# ax.invert_yaxis()

#%% run DE optimization ########################################################

# def lnpostf(pv):
#     ''' DON'T USE LAMBDA FUNCTION FOR THIS, 
#     OTHERWISE IT CAUSES PICKLE ISSUES WITH MULTIPROCESSING '''
#     return relic.lnposterior(pv) 

# with Pool(cfg["SAMPLER"]["npools"]) as pool:
#     relic.run_de(
#         niter  = cfg["SAMPLER"]["niter_de"], 
#         npop   = cfg["SAMPLER"]["nwalkers"],
#         lnpost = lnpostf,  
#         pool   = pool,  
#     )   

# #%% run MCMC sampling ##########################################################

# with Pool(cfg["SAMPLER"]["npools"]) as pool:
#     relic.run_mcmc(
#         niter   = cfg["SAMPLER"]["niter_mcmc"], 
#         lnpost  = lnpostf, 
#         pool    = pool, 
#     )


#%% Post analysis and plotting #################################################

""" Plot likelihood evolutions """
# plot_lnprob_evolution(relic, figname='lnprob.png', dpi=100, save=True)

""" Plot 2D fluxes and errors """
plot_2dfluxes(relic, figname='fluxes.png', dpi=100, save=True)

""" Plot limb darkening profiles """
plot_ldprofiles(relic, figname='ldprofiles.png', dpi=100, save=True)

""" Plot posterior distributions """ 
plot_corners(relic, samples=samples, truths=truth_pv, figname='corners.pdf', save=True)

""" Plot best-fit residuals """
maxlike_params = stats['modes'][0]["maximum"]
plot_residuals(relic, maxlike_params, figname='residuals.png', dpi=100, save=True)

print("Done!")
# %%
