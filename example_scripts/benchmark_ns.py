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

from relic.core import ReLic
from relic.atmosphere import TP6EqChem as AtmosModel
from relic.plots import PlotFigure
from relic.utils import generate_covariates, get_maxlike_estimates, print_elapsed_time, optimize_parallelization
 
from multiprocessing import Pool
 
DEFAULT_CFG = 'config/HD209458b-benchmark-r100.toml'

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
 
atmos_model = AtmosModel(cfg) # user-defined

relic = ReLic(atmos_model)
 
relic.init_prior_transform() # if using nested sampling

print("Initialization complete.") 

#%% inject a truth spectrum for testing ########################################

truth_pv = np.array([
    1.0, 3.5, 0.5, 0.00, 6000.0, 4.5, 0.0, 
    # -3.5, -1, -3.5, -3.5, -1, -3.5,
    1.5, 1.5,
    0.8, -2, -1, 0.5, 1400, 0, -0.3, -3, -3, 0.0, 0.5
])

rng = np.random.default_rng(42)
truth_fmod = relic.exoiris._tsa.flux_model(truth_pv, include_baseline=False)

# for i in range(relic.exoiris.data.size):
#     relic.exoiris._tsa.set_gp_hyperparameters(10**-3.5, 10**-1, 10**-3.5, idata=i)
# gp_sim = [relic.exoiris._tsa._gp[i].sample(include_mean=False).reshape(relic.exoiris.data[i].fluxes.shape) for i in range(relic.exoiris.data.size)]

for i in range(relic.exoiris.data.size):
    _errors = relic.exoiris.data[i].errors
    relic.exoiris.data[i].fluxes = truth_fmod[i] + rng.normal(np.zeros_like(_errors), 1.5*_errors)


#%% test likelihood evaluation #################################################

''' 
测试单线程运行是否存在内存溢出问题 

import tracemalloc

tracemalloc.start()
snap1 = tracemalloc.take_snapshot()
for _ in range(10):
    print("iteration", _)
    initial_cube = np.random.uniform(size=(30, len(relic.exoiris._tsa.ps)))
    initial_population = [nsprior.prior_transform(c) for c in initial_cube]
    pp = [relic.exoiris._tsa.lnlikelihood(_p) for _p in initial_population]
snap2 = tracemalloc.take_snapshot()
tracemalloc.stop()

# 获取内存增长最大的代码行
stats = snap2.compare_to(snap1, 'lineno')
for stat in stats[:10]:
    print(stat)

    165 ms, when using five molecules, GP, JWST NIRCam R100
    151 ms, when using five molecules, BF, JWST NIRCam R100
'''

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
    cfg["SAMPLER"]["n_live_points"], cfg["SAMPLER"]["npools"]
)

with Pool(cfg["SAMPLER"]["npools"], maxtasksperchild=maxtasks) as pool:
    results = relic.run_dynesty(
        loglikelihood   = loglikelihood,
        prior_transform = prior_transform,
        pool            = pool,
        nlivepoints     = nlivepoints,
        bound           = "single",
        sample          = "rwalk",
        maxbatch        = 1,
        queue_size      = nlivepoints
    )


#%% Post analysis and plotting #################################################

""" Plot 2D fluxes and errors """
PlotFigure(relic).plot_2dfluxes(figname='fluxes.png')

""" Plot limb darkening profiles """
PlotFigure(relic).plot_ldprofiles(figname='ldprofiles.png')

""" Plot posterior distributions """ 
samples = results.samples_equal()
PlotFigure(relic).plot_corners(samples=samples, truths=truth_pv, figname='corners.pdf')

""" Plot best-fit residuals """ 
median_params = np.median(samples, axis=0)
PlotFigure(relic).plot_residuals(median_params, figname='residuals.png')

print("Done!")
# %%
