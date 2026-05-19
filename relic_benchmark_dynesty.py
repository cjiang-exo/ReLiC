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
import pickle
import shutil

from relic_core import ReLic
from relic_atmosphere import TP6EqChem as AtmosModel
from relic_plots import *
from relic_utils import generate_covariates, get_maxlike_estimates, print_elapsed_time, optimize_parallelization
 
from multiprocessing import Pool

# from dynesty.pool import Pool as DynestyPool
# from dynesty import plotting as dyplot
# from dynesty import DynamicNestedSampler

DEFAULT_CFG = 'config/HD209458b_benchmark-pix.toml'

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

#%% inject a truth spectrum for testing

truth_pv = np.array([
    1.0, 3.5, 0.5, 0.00, 6000.0, 4.5, 0.0, 1.5, 1.5,
    0.8, -2, -1, 0.5, 1400, 0, -0.3, -3, -3, 0.0, 0.5
])


rng = np.random.default_rng(42)
truth_fmod = relic.exoiris._tsa.flux_model(truth_pv, include_baseline=False)
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

'''

print("Running a quick test of posterior evaluation...")

initial_cube = np.random.uniform(size=(3, len(relic.exoiris._tsa.ps)))

initial_population = [relic.prior_transform(c) for c in initial_cube]
pp = [relic.lnlikelihood_ns(_p) for _p in initial_population]
[print(f"lnprob = {_v:.6e}") for _v in pp]

print("Test complete.") 

#%% run multinest sampling #####################################################

nlivepoints, _maxtasks = optimize_parallelization(cfg["SAMPLER"]["n_live_points"], cfg["SAMPLER"]["npools"])
# _chunksize, _extra = divmod(nlivepoints, (cfg["SAMPLER"]["npools"] * 4))
# if _extra:
#     _chunksize += 1
# allow_memoryleak = 1 # GiB
# _maxtasks = max(4, int(40000 * allow_memoryleak // nlivepoints) // 4 * 4)

with Pool(cfg["SAMPLER"]["npools"], maxtasksperchild=_maxtasks) as pool:
    results = relic.run_dynesty(
        pool=pool,
        nlivepoints=nlivepoints,
        bound="single",
        sample="rwalk",
        n_effective=None,
        maxbatch=1,
        queue_size=cfg["SAMPLER"]["npools"]
    )

# def loglikelihood(pv):
#     return relic.lnlikelihood_ns(pv)
# def prior_transform(uv):
#     return nsprior.prior_transform(uv)

# nlivepoints = suggest_livepoints(cfg["SAMPLER"]["n_live_points"], cfg["SAMPLER"]["npools"])
# _chunksize = nlivepoints // (cfg["SAMPLER"]["npools"] * 4)
# _maxtasks = 100 // _chunksize

# with Pool(cfg["SAMPLER"]["npools"], maxtasksperchild=10) as pool:
#     sampler = DynamicNestedSampler( 
#         loglikelihood,
#         prior_transform,
#         len(relic.exoiris._tsa.ps), 
#         pool=pool,
#         nlive=cfg["SAMPLER"]["n_live_points"],
#         bound="single",
#         sample="rwalk",
#         queue_size=cfg["SAMPLER"]["npools"],
#         use_pool={"prior_transform": False}
#     ) 
#     sampler.run_nested(
#         dlogz_init=cfg["SAMPLER"]["evidence_tolerance"], 
#         # n_effective=n_effective,
#         maxbatch=1,
#     )

# results = sampler.results
# with open(os.path.join(cfg["PATH"]["output_dir"], 'ns_results.pkl'), 'wb') as f:
#     pickle.dump(results, f)
#     print(f"Dynesty results saved to {os.path.join(cfg['PATH']['output_dir'], 'ns_results.pkl')}")

# results.summary()
# samples = results.samples_equal()
# print(f"Posterior samples collected: {samples.shape[0]}")

# try:
#     fig, axes = dyplot.runplot(results) 
#     fig.tight_layout()
#     fig.savefig(os.path.join(cfg["PATH"]["output_dir"], 'dynesty_runplot.png'), dpi=100)
# except Exception as e:
#     print(f"Error generating dynesty runplot: {e}")

#%% Post analysis and plotting #################################################

""" Plot 2D fluxes and errors """
plot_2dfluxes(relic, figname='fluxes.png', dpi=100, save=True)

""" Plot limb darkening profiles """
plot_ldprofiles(relic, figname='ldprofiles.png', dpi=100, save=True)

""" Plot posterior distributions """ 
samples = results.samples_equal()
plot_corners(relic, samples=samples, truths=truth_pv, figname='corners.pdf', save=True)

""" Plot best-fit residuals """ 
median_params = np.median(samples, axis=0)
plot_residuals(relic, median_params, figname='residuals.png', dpi=100, save=True)

print("Done!")
# %%
