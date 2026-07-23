export HDF5_USE_FILE_LOCKING=FALSE
set -e
 
nohup python pipeline_ns.py -c "config.toml" > output.log 2>&1 &
