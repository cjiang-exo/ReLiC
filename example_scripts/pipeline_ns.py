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
from multiprocessing import Pool
from pathlib import Path
from relic import Relic
from relic.atmosphere import *

DEFAULT_CFG = 'example_config/HD209458b_ns.toml'

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
 
#%% fit white light curves to validate the covariates ##########################
 
state_vectors_alldata = []
for i, rd in enumerate(relic.raw_data):  
    if "JWST" in relic.exoiris.data[i].name:
        _state_vectors = None
    else:
        _state_vectors = rd["pca_jitters"][:, :2]   
    state_vectors_alldata.append(_state_vectors) 
relic.update_covariates(state_vectors_alldata)

def lnpost_white(pv):
    return relic.exoiris._wa.lnposterior(pv)

npools = relic.cfg["SAMPLER"]["npools"]
with Pool(npools) as pool:
    relic.fit_white(pool=pool, lnpost=lnpost_white)
 
#%% test likelihood evaluation #################################################

relic.run_test(3)

#%% run nested sampling ########################################################

def loglikelihood(pv): 
    return relic.lnlikelihood_ns(pv)
def prior_transform(uv):
    return relic.prior_transform(uv)
 
with Pool(relic.cfg["SAMPLER"]["npools"], maxtasksperchild=200) as pool:
    sampler, results = relic.run_nautilus(
        prior           = prior_transform,
        loglikelihood   = loglikelihood, 
        pool            = pool,   
    )

print("Done!")

# %%
