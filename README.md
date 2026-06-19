# ReLic: atmospheric Retrievals using spectral Light Curves.

A Python package for exoplanet atmospheric retrieval using spectral light curves.

## Project Structure

```
relic/                           # Python package (source code)
├── __init__.py                  # Package exports
├── core.py                      # Main ReLic retrieval class + ExoIris integration (ReLicExoIris)
├── atmosphere.py                # Atmospheric models (BaseAtmosphere, TP6EqChem, etc.)
├── tslpf.py                     # Time-series LPF (NewTSLPF)
├── white.py                     # White light curve fitting (NewWhiteLPF)
├── utils.py                     # Utilities (SpectrumDownsampler, covariates, etc.)
├── physics.py                   # Physics helpers (calc_teq)
└── plots.py                     # Plotting functions
example_scripts/                 # Executable entry-point scripts
├── benchmark_mcmc.py            # Benchmark script (MCMC)
├── benchmark_ns.py              # Benchmark script (nested sampling)
├── pipeline_mcmc.py             # Pipeline script (MCMC)
├── pipeline_ns.py               # Pipeline script (nested sampling)
└── run.sh                       # Launch script for batch runs
config/                          # Configuration files (TOML/JSON)
tests/                           # Test directory
examples/                        # Example notebooks/scripts
```

## Installation

Install the package in development mode:

```bash
pip install -e .
```

## Usage

Run a retrieval from the project root:

```bash
# MCMC retrieval
python example_scripts/benchmark_mcmc.py -c config/HD209458b-benchmark-r100.toml

# Nested-sampling retrieval
python example_scripts/benchmark_ns.py -c config/WASP39b-PCA-r100-tp6fastchem.toml

# Batch launch
bash scripts/run.sh
```
