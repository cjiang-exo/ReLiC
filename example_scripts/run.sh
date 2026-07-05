# tail -f output.log

set -e
 
nohup python pipeline_ns.py -c "/home/ubuntu/work/relic/source/config/HD209458b-joint-r100-tp6fastchem.toml" > output-r100.log 2>&1 &

nohup python pipeline_ns.py -c "/home/ubuntu/work/relic/source/config/HD209458b-joint-pix-tp6fastchem.toml" > output-pix.log 2>&1 &

# python scripts/exec_ns.py -c "config/HD209458b-jwst-r100-gp-tp6eqc.toml"

# python scripts/exec_ns.py -c "config/HD209458b-hst-fc-tio.toml"
# python scripts/exec_ns.py -c "config/HD209458b-hst-noalkali.toml"
