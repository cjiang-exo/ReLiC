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
import shutil

from relic_core import ReLic
from relic_atmosphere import TP6EqChem, TP6FreeChem
from relic_plots import *
from relic_utils import generate_covariates, optimize_parallelization

from multiprocessing import Pool

DEFAULT_CFG = 'config/HD209458b-joint-r100-tp6fc.toml'

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
shutil.copy(py_args.config, os.path.join(cfg["PATH"]["output_dir"], os.path.basename(py_args.config)))

#%% Initialization #############################################################

atmclass = eval(cfg['ATMOSPHERE']['model_class'])
atmos_model = atmclass(cfg) # user-defined

relic = ReLic(atmos_model)

relic.init_prior_transform() # if using nested sampling

print("Initialization complete.") 

#%% fit white light curves to validate the covariates ##########################
 
jitters = []
for i, rd in enumerate(relic.raw_data):
    _jit = rd["detrend_vectors"][:, :2] if "STIS" in relic.exoiris.data[i].name else None 
    jitters.append(_jit) 

white_covariates = generate_covariates(relic, jitters)

with Pool(cfg["SAMPLER"]["npools"]) as pool:
    relic.fit_white(covariates=white_covariates, pool=pool)

relic.update_covariates()

plot_white(relic)

#%% test likelihood evaluation #################################################

print("Running a quick test of posterior evaluation...")

initial_cube = np.random.uniform(size=(3, len(relic.exoiris._tsa.ps)))

initial_population = [relic.prior_transform(c) for c in initial_cube]
pp = [relic.lnlikelihood_ns(_p) for _p in initial_population]
[print(f"lnprob = {_v:.6e}") for _v in pp]

print("Test complete.") 

#%% run nested sampling ########################################################

def loglikelihood(pv):
    return relic.lnlikelihood_ns(pv)
def prior_transform(uv):
    return relic.prior_transform(uv)

nlivepoints, maxtasks = optimize_parallelization(
    cfg["SAMPLER"]["n_live_points"], cfg["SAMPLER"]["npools"])

with Pool(cfg["SAMPLER"]["npools"], maxtasksperchild=maxtasks) as pool:
    results = relic.run_dynesty(
        loglikelihood   = loglikelihood,
        prior_transform = prior_transform,
        pool            = pool,
        queue_size      = cfg["SAMPLER"]["npools"],
        nlivepoints     = nlivepoints,
        bound           = "multi",
        sample          = "rwalk", 
    )

#%% Post analysis and plotting #################################################

""" Plot 2D fluxes and errors """
plot_2dfluxes(relic, figname='fluxes.png', dpi=100, save=True)

""" Plot limb darkening profiles """
plot_ldprofiles(relic, figname='ldprofiles.png', dpi=100, save=True)

""" Plot posterior distributions """  
maxlike_params = results['samples'][results['logl'].argmax()]
plot_corners(relic, samples=results.samples_equal(), truths=maxlike_params, figname='corners.pdf', save=True) 

""" Plot best-fit residuals """ 
plot_residuals(relic, maxlike_params, figname='residuals.png', dpi=100, save=True)

""" Plot transmission spectra """
plot_transmission_spectra(relic, maxlike_params, samples=results.samples_equal(), figname='ts_preview.png', dpi=100, save=True)

print("Done!")

# %%
