# nohup bash run.sh > output.log 2>&1 &
# tail -f output.log

set -e

python lcr_exec_mcmc.py -c "config/HD209458b_joint.toml"

# mpiexec -n 30 python lcr_exec_ns.py -c "config/HD209458b_ns.json"

# python lcr_test_injection.py -c "config/HD209458b_mcmc.json"

