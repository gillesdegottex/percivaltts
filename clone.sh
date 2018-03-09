#!/bin/bash

# This script is for cloning the code in an "experiment" directory, and run it.
# By cloning the code, you also keep an original copy of the experiment's code
# while you can work/run other experiments in parallel without losing track of
# what has been changed in between.

# The output of the experiment will appear in a sub-directory "out" along side
# the directory "Code", the clone source code.

# This is also useful for running variants of an experiment by copying the whole
# working directory (the parent dir of "Code" and "out") and modifying only a
# single parameter.


#Copyright(C) 2017 Engineering Department, University of Cambridge, UK.
#
#License
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#Author
#   Gilles Degottex <gad27@cam.ac.uk>
#

WORKDIR=$1

CODEDIR=$(dirname "$0")              # relative
CODEDIR=$( ( cd "$CODEDIR" && pwd ) )
echo -n Cloning \"$CODEDIR\" in \"$WORKDIR\"

# Create the destination directory if access done by NFS
if [[ $WORKDIR = '/'* ]] ; then
    echo " (using file system)"
    mkdir -p $WORKDIR;
    mkdir -p $WORKDIR/out;
else
    echo " (using ssh on ${WORKDIR%:*})"
    ssh ${WORKDIR%:*} /bin/mkdir -p ${WORKDIR#*:}/out;
fi

# Do the actual copy
# cp -fr $CODEDIR $WORKDIR
rsync -qav $CODEDIR/ $WORKDIR/Code/ --exclude .git --exclude .git/


# Copy a short version of the git in the working directory
if [[ -d '.git/' ]] ; then
    rm -fr tmp_clone_gitstatedepth1;
    mkdir -p tmp_clone_gitstatedepth1;
    cd tmp_clone_gitstatedepth1;
    git clone --depth 1 file://$CODEDIR gitstatedepth1;
    rsync -qav gitstatedepth1/.git/ $WORKDIR/Code/.git/;
    cd ..;
    rm -fr tmp_clone_gitstatedepth1;
fi


if [[ "${@:2}" ]]; then

# Go into the working directory
cd $WORKDIR/out

echo Run command: "${@:2}"
# ${@:2} > log 2>&1
${@:2}

fi
