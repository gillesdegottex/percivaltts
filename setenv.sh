#!/bin/bash

# Set the environment variables before calling the python code

CUDAPATH=/usr/local/cuda-9.0

CONDAPATH=/opt/miniconda2
export PYTHONPATH=$CONDAPATH/lib/python2.7/site-packages:$PYTHONPATH
export PATH=$CUDAPATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDAPATH/lib64:$CONDAPATH/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDAPATH/lib64:$CONDAPATH/lib:$LIBRARY_PATH
export CPATH="$CUDAPATH/include:$CONDAPATH/include:$CPATH"


# Use default Ubuntu CUDA install
export THEANO_FLAGS="cuda.root=$CUDAPATH,floatX=float32,on_unused_input=ignore,"

#export THEANO_FLAGS="optimizer_including=cudnn:local_ultra_fast_sigmoid,"$THEANO_FLAGS #local_ultra_fast_sigmoid seems to block training
# Force use of cuDNN
#export THEANO_FLAGS="optimizer_including=cudnn,"$THEANO_FLAGS
# export THEANO_FLAGS="optimizer_excluding=cudnn,"$THEANO_FLAGS
# Force use of cnMEM
# export THEANO_FLAGS="lib.cnmem=1,"$THEANO_FLAGS


export THEANO_FLAGS="mode=FAST_RUN,device=cuda,"$THEANO_FLAGS
# For debugging, uncomment below and comment above
# export THEANO_FLAGS="device=cpu,exception_verbosity=high,optimizer=None,"$THEANO_FLAGS

# Theano loads way faster by compiling on shm
if [ "$HOSTNAME" == "mila" ]; then
    export THEANO_FLAGS="base_compiledir=/dev/shm/$USERNAME,"$THEANO_FLAGS
fi

unset LD_PRELOAD

python $@
