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
from relic.utils import optimize_parallelization

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
pfig  = PlotFigure(relic, dpi=100, save=True)
 
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

#%% run nested sampling ########################################################

def loglikelihood(pv):
    return relic.lnlikelihood_ns(pv)
def prior_transform(uv):
    return relic.prior_transform(uv)

nlivepoints, maxtasks = optimize_parallelization(
    relic.cfg["SAMPLER"]["n_live_points"], relic.cfg["SAMPLER"]["npools"])

with Pool(npools, maxtasksperchild=maxtasks) as pool:
    results = relic.run_dynesty(
        loglikelihood   = loglikelihood,
        prior_transform = prior_transform,
        pool            = pool,
        queue_size      = npools,
        nlivepoints     = nlivepoints,
        bound           = "multi",
        sample          = "rwalk", 
    )

#%% Post analysis and plotting #################################################

""" Plot 2D fluxes and errors """
pfig.plot_2dfluxes(figname='fluxes.png')

""" Plot limb darkening profiles """
pfig.plot_ldprofiles(figname='ldprofiles.png')

""" Plot posterior distributions """  
maxlike_params = results['samples'][results['logl'].argmax()]
pfig.plot_corners(samples=results.samples_equal(), truths=maxlike_params, figname='corners.pdf') 

""" Plot best-fit residuals """ 
pfig.plot_residuals(maxlike_params, figname='residuals.png')

""" Plot transmission spectra """
pfig.plot_transmission_spectra(maxlike_params, samples=results.samples_equal())

print("Done!")

# %%
