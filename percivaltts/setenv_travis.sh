#!/bin/bash

# Set the environment variables before calling the python code

CONDAPATH=$HOME/miniconda

export PYTHONPATH=$CONDAPATH/lib/python2.7/site-packages:$PYTHONPATH
export PATH=$CONDAPATH/bin:$PATH
export LD_LIBRARY_PATH=$CONDAPATH/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDAPATH/lib:$LIBRARY_PATH
export CPATH="$CONDAPATH/include:$CPATH"

unset LD_PRELOAD

$@
