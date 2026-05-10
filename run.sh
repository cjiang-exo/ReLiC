# nohup bash run.sh > output-r100.log 2>&1 &
# nohup bash run.sh > output-pix.log 2>&1 &
# tail -f output.log

set -e

# python lcr_exec_mcmc.py -c "config/HD209458b_joint.toml"

python lcr_injection_test.py -c "config/HD209458b_benchmark-r100.toml"

# mpiexec -n 30 python lcr_exec_ns.py -c "config/HD209458b_ns.json"

# python lcr_test_injection.py -c "config/HD209458b_mcmc.json"

