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
from relic.core import ReLic 
from relic.atmosphere import TP6EqChem as AtmosModel
from relic.plots import *

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

relic = ReLic(atmos_model)

print("Initialization complete.") 

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

plot_white(relic)


#%% test likelihood evaluation #################################################

print("Running a quick test of posterior evaluation...")

initial_population = relic.sample_from_prior(1)
pp = relic.lnposterior(initial_population)
print(f"lnprob = {pp:.6e}")

initial_population = relic.sample_from_prior(3)
pp = [relic.lnposterior(_p) for _p in initial_population]
[print(f"lnprob = {_v:.6e}") for _v in pp]

print("Test complete.")


#%% run DE optimization ########################################################

def lnpostf(pv):
    ''' DON'T USE LAMBDA FUNCTION FOR THIS, 
    OTHERWISE IT CAUSES PICKLE ISSUES WITH MULTIPROCESSING '''
    return relic.lnposterior(pv) 

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
plot_lnprob_evolution(relic, figname='lnprob.png', dpi=100, save=True)

""" Plot 2D fluxes and errors """
plot_2dfluxes(relic, figname='fluxes.png', dpi=100, save=True)

""" Plot limb darkening profiles """
plot_ldprofiles(relic, figname='ldprofiles.png', dpi=100, save=True)

""" Plot posterior distributions """
maxlike_params = get_maxlike_estimates(relic)
plot_corners(relic, truths=maxlike_params, figname='corners.pdf', save=True)

""" Plot best-fit residuals """
plot_residuals(relic, maxlike_params, figname='residuals.png', dpi=100, save=True)

print("Done!")
# %%
