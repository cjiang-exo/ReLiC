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
from relic.core import Relic 
from relic.plots import RelicVisualization 

DEFAULT_CFG = 'tmp.toml'

if 'get_ipython' in globals(): 
    config = DEFAULT_CFG # use DEFAULT_CFG if running in Jupyter Notebook
else: # for command line execution 
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', 
                        default=DEFAULT_CFG,
                        help="the input configuration file")
    config = parser.parse_args().config
 
#%% Initialization #############################################################
 
relic = Relic(config)
visual = RelicVisualization(relic, dpi=100, save=True)
 
#%% fit white light curves to validate the covariates ##########################
 
state_vectors_alldata = []
for i, rd in enumerate(relic.raw_data):
    if "STIS" in relic.exoiris.data[i].name:
        _state_vectors = rd["pca_jitters"][:, :2]
    elif "WFC3" in relic.exoiris.data[i].name:
        _state_vectors = rd["pca_jitters"][:, :2]
    else:
        _state_vectors = None
    state_vectors_alldata.append(_state_vectors) 

relic.update_covariates(state_vectors_alldata)

def lnpost_white(pv):
    return relic.exoiris._wa.lnposterior(pv)

npools = relic.cfg["SAMPLER"]["npools"]
with Pool(npools) as pool:
    relic.fit_white(pool=pool, lnpost=lnpost_white)
 
visual.plot_white()

#%% test likelihood evaluation #################################################

relic.run_test(3)

#%% run nested sampling ########################################################

def loglikelihood(pv):
    return relic.lnlikelihood_ns(pv)
def prior_transform(uv):
    return relic.prior_transform(uv)

with Pool(npools, maxtasksperchild=500) as pool:
    results = relic.run_nautilus(
        loglikelihood   = loglikelihood, 
        pool            = pool, 
        n_live_points   = relic.cfg["SAMPLER"]["n_live_points"],
        n_effective     = relic.cfg["SAMPLER"]["n_effective"],
    )

#%% Post analysis and plotting #################################################

""" Plot 2D fluxes and errors """
visual.plot_2dfluxes(figname='fluxes.png')

""" Plot limb darkening profiles """
visual.plot_ldprofiles(figname='ldprofiles.png')

""" Plot posterior distributions """   
samples = results['posterior_samples']
maxlike_params = samples[results['log_likelihoods'].argmax()]
weights = np.exp(results['log_weights'])
visual.plot_corners(samples=samples, weights=weights, truths=maxlike_params, figname='corners.pdf') 

""" Plot best-fit residuals """ 
visual.plot_residuals(maxlike_params, figname='residuals.png')

""" Plot transmission spectra """
visual.plot_transmission_spectra(maxlike_params, figname='transmission_spectra.png')

print("Done!")

# %%
