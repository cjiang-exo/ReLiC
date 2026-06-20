#%% 
import os
os.environ["OMP_NUM_THREADS"] =        "1"
os.environ["OPENBLAS_NUM_THREADS"] =   "1"
os.environ["MKL_NUM_THREADS"] =        "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] =    "1"
os.environ["NUMBA_NUM_THREADS"] =      "1" 
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'   
from multiprocessing import Pool   

import argparse
import numpy as np 

from relic.core import ReLic  
from relic.plots import PlotFigure 
from relic.utils import get_maxlike_estimates, optimize_parallelization 

DEFAULT_CFG = '/work/relic/source/config/HD209458b-jwst-pix-tp6fastchem.toml'

if 'get_ipython' in globals(): 
    config = DEFAULT_CFG # use DEFAULT_CFG if running in Jupyter Notebook
else: # for command line execution 
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', 
                        default=DEFAULT_CFG,
                        help="the input configuration file")
    config = parser.parse_args().config

#%% Initialization #############################################################

relic = ReLic(config)
pfig  = PlotFigure(relic)

#%% fit white light curves to validate the covariates ##########################
 
jitters = []
for i, rd in enumerate(relic.raw_data):
    if "STIS" in relic.exoiris.data[i].name:
        _jit = rd["pca_jitters"][:, :2]
    elif "WFC3" in relic.exoiris.data[i].name:
        _jit = rd["pca_jitters"][:, :2]
    else:
        _jit = None
    jitters.append(_jit)
white_covariates = relic.generate_covariates(jitters)
 
npools = relic.cfg["SAMPLER"]["npools"]
with Pool(npools) as pool:
    relic.fit_white(covariates=white_covariates, update_covariates=False, pool=pool)

pfig.plot_white()

#%% test likelihood evaluation #################################################

relic.run_test(3)

#%% run DE optimization ########################################################

def lnpostf(pv):
    ''' DON'T USE LAMBDA FUNCTION FOR THIS, 
    OTHERWISE IT CAUSES PICKLE ISSUES WITH MULTIPROCESSING '''
    return relic.lnposterior_mcmc(pv) 

with Pool(npools) as pool:
    relic.run_de(
        niter  = relic.cfg["SAMPLER"]["niter_de"], 
        npop   = relic.cfg["SAMPLER"]["nwalkers"],
        lnpost = lnpostf,  
        pool   = pool,  
    )   

#%% run MCMC sampling ##########################################################

with Pool(npools) as pool:
    relic.run_mcmc(
        niter   = relic.cfg["SAMPLER"]["niter_mcmc"], 
        lnpost  = lnpostf, 
        pool    = pool, 
    )


#%% Post analysis and plotting #################################################

relic.save_mcmc(overwrite=True, config_file=config)

""" Plot likelihood evolutions """
pfig.plot_mcmc_lnprob(figname='lnprob.png')

""" Plot 2D fluxes and errors """
pfig.plot_2dfluxes(figname='fluxes.png')

""" Plot limb darkening profiles """
pfig.plot_ldprofiles(figname='ldprofiles.png')

""" Plot posterior distributions """
maxlike_params = get_maxlike_estimates(relic)
pfig.plot_corners(truths=maxlike_params, figname='corners.pdf')

""" Plot best-fit residuals """
pfig.plot_residuals(maxlike_params, figname='residuals.png')

print("Done!")
# %%
