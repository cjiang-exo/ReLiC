# nohup bash run.sh > output.log 2>&1 &
# nohup bash run.sh > output-pix.log 2>&1 &
# tail -f output.log

set -e
 
# python relic_benchmark.py -c "config/HD209458b_benchmark-r100.toml"
# python relic_benchmark.py -c "config/HD209458b_benchmark-pix.toml"

nohup python relic_exec_ns.py -c "config/HD209458b-jwst-r100-tp6fastchem.toml" > output-r100.log 2>&1 &
nohup python relic_exec_ns.py -c "config/HD209458b-jwst-pix-tp6fastchem.toml" > output-pix.log 2>&1 &

# python relic_exec_ns.py -c "config/HD209458b-jwst-r100-gp-tp6eqc.toml"

# python relic_exec_ns.py -c "config/HD209458b-hst-fc-tio.toml"
# python relic_exec_ns.py -c "config/HD209458b-hst-noalkali.toml"
