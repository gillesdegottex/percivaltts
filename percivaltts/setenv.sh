#!/bin/bash

# Set the environment variables before calling the python code
# This file is meant to be modified according to your packages installation.

# You might want to change these two paths according to your CUDA/CONDA install.
CUDA_ROOT=/usr/local/cuda-9.0
. $HOME/miniconda/etc/profile.d/conda.sh
conda activate

# export MKL_THREADING_LAYER=GNU

# To output some information about TensorFlow, change it to 0
export TF_CPP_MIN_LOG_LEVEL=3
# To select one specific GPU on a multiple-GPU machine
# export CUDA_VISIBLE_DEVICES=0
# To hide the GPUs and thus force using the CPU (makes runs repeatable)
# export CUDA_VISIBLE_DEVICES=""
# This should also help to make runs repeatable
# export PYTHONHASHSEED=0

# Add CUDA
#export PATH=$CUDA_ROOT/bin:$PATH
#export CPATH=$CUDA_ROOT/include:$CPATH
#export LIBRARY_PATH=$CUDA_ROOT/lib64:$LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
#export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH


unset LD_PRELOAD

echo "Run command: "$@

# Finally, run the scripts' arguments as a command
$@
