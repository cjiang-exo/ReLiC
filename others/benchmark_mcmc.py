#%% 
import os
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
from relic.core import Relic 
from relic.atmosphere import TP6EqChem as AtmosModel
from relic.plots import RelicVisualization

from relic.utils import generate_covariates, get_maxlike_estimates
from multiprocessing import Pool   
from numpy import inf

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

relic = Relic(atmos_model)

def additional_priors(pv):
    p1, p2, p3 = pv[16:19]
    if (p1 > p3) or (p2 > p3):
        return -inf
    return 0.0

relic.exoiris._tsa.additional_priors = additional_priors

print("Initialization complete.") 

#%% inject a truth spectrum for testing

truth_pv = np.array([
    1.0, 3.5, 0.5, 0.00, 6000.0, 4.5, 0.0, 1.5, 1.5,
    0.8, -2, -1, 0.5, 1400, 1, 0.5, -3, -3, 0, 0.0, 0.5
])

truth_fmod = relic.exoiris._tsa.flux_model(truth_pv, include_baseline=False)
for i in range(relic.exoiris.data.size):
    _errors = relic.exoiris.data[i].errors
    relic.exoiris.data[i].fluxes = truth_fmod[i] + np.random.normal(np.zeros_like(_errors), 1.5*_errors)

#%% fit white light curves to validate the covariates ##########################
 
jitters = []
for i, rd in enumerate(relic.raw_data):
    if "STIS" in relic.exoiris.data[i].name: # use the first two PCs
        jitters.append(rd["detrend_vectors"][:, :2]) 
    else:
        jitters.append(None) 

covariates = generate_covariates(relic, jitters)

with Pool(cfg["SAMPLER"]["npools"]) as pool:
    relic.fit_white(covariates=covariates, pool=pool)

relic.update_covariates()

RelicVisualization(relic).plot_white()


#%% test likelihood evaluation #################################################

print("Running a quick test of posterior evaluation...")

initial_population = relic.sample_from_prior(1)
pp = relic.lnposterior_mcmc(initial_population)
print(f"lnprob = {pp:.6e}")

initial_population = relic.sample_from_prior(3)
pp = [relic.lnposterior_mcmc(_p) for _p in initial_population]
[print(f"lnprob = {_v:.6e}") for _v in pp]

print("Test complete.")

#%% test tp6
 
# temp = atmos_model.tp6madhu(atmos_model.pressures_bar, 1400, 1, 0.5, -3, -3, 0)

# fig, ax  = pl.subplots(figsize=(6,4))
# ax.plot(temp, relic.atmos_model.pressures_bar)
# ax.set_xlabel("Temperature (K)")
# ax.set_yscale("log")
# ax.invert_yaxis()

#%% run DE optimization ########################################################

def lnpostf(pv):
    ''' DON'T USE LAMBDA FUNCTION FOR THIS, 
    OTHERWISE IT CAUSES PICKLE ISSUES WITH MULTIPROCESSING '''
    return relic.lnposterior_mcmc(pv) 

with Pool(cfg["SAMPLER"]["npools"]) as pool:
    relic.run_de(
        niter  = cfg["SAMPLER"]["niter_de"], 
        npop   = cfg["SAMPLER"]["nwalkers"],
        lnpost = lnpostf,  
        pool   = pool,  
    )   

#%% run MCMC sampling ##########################################################

with Pool(cfg["SAMPLER"]["npools"]) as pool:
    relic.run_mcmc(
        niter   = cfg["SAMPLER"]["niter_mcmc"], 
        lnpost  = lnpostf, 
        pool    = pool, 
    )


#%% Post analysis and plotting #################################################
relic.save_mcmc(overwrite=True, config_file=py_args.config)

""" Plot likelihood evolutions """
RelicVisualization(relic).plot_lnprob_evolution(figname='lnprob.png')

""" Plot 2D fluxes and errors """
RelicVisualization(relic).plot_2dfluxes(figname='fluxes.png')

""" Plot limb darkening profiles """
RelicVisualization(relic).plot_ldprofiles(figname='ldprofiles.png')

""" Plot posterior distributions """
maxlike_params = get_maxlike_estimates(relic)
RelicVisualization(relic).plot_corners(truths=truth_pv, figname='corners.pdf')

""" Plot best-fit residuals """
RelicVisualization(relic).plot_residuals(maxlike_params, figname='residuals.png')

print("Done!")
# %%
