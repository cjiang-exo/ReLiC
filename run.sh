# nohup bash run.sh > output.log 2>&1 &
# nohup bash run.sh > output-pix.log 2>&1 &
# tail -f output.log

set -e

# python lcr_exec_mcmc.py -c "config/HD209458b_joint_flatmass.toml"

# python relic_benchmark.py -c "config/HD209458b_benchmark-r100.toml"
# python relic_benchmark.py -c "config/HD209458b_benchmark-pix.toml"

# mpiexec -n 30 python relic_benchmark_ns.py -c "config/HD209458b_benchmark-r100.toml"
mpiexec -n 30 python relic_benchmark_ns.py -c "config/HD209458b_benchmark-pix.toml"

