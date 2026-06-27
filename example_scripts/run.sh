# nohup bash scripts/run.sh > output.log 2>&1 &
# nohup bash scripts/run.sh > output-pix.log 2>&1 &
# tail -f output.log

set -e

# SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# cd "$PROJECT_DIR" || exit 1

# python scripts/benchmark.py -c "config/HD209458b_benchmark-r100.toml"
# python scripts/benchmark.py -c "config/HD209458b_benchmark-pix.toml"

nohup python pipeline_ns.py -c "/home/ubuntu/work/relic/source/config/HD209458b-joint-r100-tp6fastchem.toml" > output-r100.log 2>&1 &

nohup python pipeline_ns.py -c "/home/ubuntu/work/relic/source/config/HD209458b-joint-pix-tp6fastchem.toml" > output-pix.log 2>&1 &

# nohup python scripts/exec_ns.py -c "config/WASP39b-PCA-r100-tp6fastchem.toml" > output-PCA.log 2>&1 &
# nohup python scripts/exec_ns.py -c "config/WASP39b-PCA-r100-tp6fastchem+SO2.toml" > output-PCA+SO2.log 2>&1 &

# python scripts/exec_ns.py -c "config/HD209458b-jwst-r100-gp-tp6eqc.toml"

# python scripts/exec_ns.py -c "config/HD209458b-hst-fc-tio.toml"
# python scripts/exec_ns.py -c "config/HD209458b-hst-noalkali.toml"
