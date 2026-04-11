# nohup bash run.sh > output.log 2>&1 &
# tail -f output.log

set -e

# python lcr_exec.py -c "config/HD209458b.json"

mpiexec -n 30 python lcr_exec.py -c "config/HD209458b.json"