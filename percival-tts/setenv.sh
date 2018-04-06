#!/bin/bash

# Set the environment variables before calling the python code

# This file is meant to be modified according to your packages installation.


# You might want to change these two paths according to your CONDA/CUDA install.
CONDAPATH=$HOME/miniconda
CUDA_ROOT=/usr/local/cuda-9.0

# Use CONDA python libraries first
export PYTHONPATH=$CONDAPATH/lib/python2.7/site-packages:$PYTHONPATH

# Add CUDA
export PATH=$CUDA_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$CONDAPATH/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_ROOT/lib64:$CONDAPATH/lib:$LIBRARY_PATH
export CPATH="$CUDA_ROOT/include:$CONDAPATH/include:$CPATH"

# Basic Thenao flags
export THEANO_FLAGS="floatX=float32,on_unused_input=ignore,"

# Force using of cuDNN. An error will be thrown if cuDNN is not accessible
# export THEANO_FLAGS="optimizer_including=cudnn,"$THEANO_FLAGS

#export THEANO_FLAGS="optimizer_including=cudnn:local_ultra_fast_sigmoid,"$THEANO_FLAGS #local_ultra_fast_sigmoid seems to block training


# For repeatability of 2D convolution layers
# export THEANO_FLAGS="dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic,"$THEANO_FLAGS
# or
# export THEANO_FLAGS="dnn.enabled=False,"$THEANO_FLAGS
# and this might help too
# export THEANO_FLAGS="deterministic=more,"$THEANO_FLAGS


# For maximum speed:
export THEANO_FLAGS="mode=FAST_RUN,device=cuda,"$THEANO_FLAGS
# For debugging, comment the line above and uncomment the line below
# export THEANO_FLAGS="device=cpu,exception_verbosity=high,optimizer=None,"$THEANO_FLAGS

# Theano loads way faster by compiling on shm.
# Hardcode the compute hostnames that have shm here below.
if [ "$HOSTNAME" == "mila" ]; then
    export THEANO_FLAGS="base_compiledir=/dev/shm/$USERNAME,"$THEANO_FLAGS
fi

unset LD_PRELOAD

# Finally, run the scripts' arguments as a python command
python $@
