#!/bin/bash

# Set the environment variables before calling the python code

CONDAPATH=$HOME/miniconda

export PYTHONPATH=$CONDAPATH/lib/python2.7/site-packages:$PYTHONPATH
export PATH=$CONDAPATH/bin:$PATH
export LD_LIBRARY_PATH=$CONDAPATH/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDAPATH/lib:$LIBRARY_PATH
export CPATH="$CONDAPATH/include:$CPATH"

export THEANO_FLAGS="floatX=float32,on_unused_input=ignore,"

# Enable all possible verbosity options to ease debuging if tests fail.
export THEANO_FLAGS="device=cpu,exception_verbosity=high,optimizer=None,"$THEANO_FLAGS

unset LD_PRELOAD

if [ "$1" == "coverage" ] ; then
    $@
else
    python $@
fi
