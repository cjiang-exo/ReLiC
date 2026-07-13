# tail -f output.log

set -e
 
nohup python pipeline_ns.py -c "/home/ubuntu/work/relic/source/config_paper/HD209458b-jwst-pix-fiducial.toml" > output-jwst-fiducial.log 2>&1 &

nohup python pipeline_ns.py -c "/home/ubuntu/work/relic/source/config_paper/HD209458b-joint-pix-fiducial.toml" > output-joint-fiducial.log 2>&1 &

