#!/bin/bash

# This script clones the code in an "experiment" directory EXPDIR, and run it.
# By cloning the code, you keep an original copy of the experiment that is
# running while you can continue working on other experiments in parallel
# without losing track of what has been changed in the meantime.

# The output of the experiment will appear in a sub-directory OUTDIR along side
# the CODEDIR directory, the cloned source code.

# This is also useful for running slight variants of an experiment by copying
# the whole experiment directory (the parent dir of CODEDIR and OUTDIR) and
# modifying only a single parameter.

# Example
#  ./clone.sh /data/research/exp1 bash setenv.sh run.py
# which clones the current directory into /data/research/exp1/percival; cd
# into this same directory and run the command "$ bash setenv.sh run.py"

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

# The root experiment directory where CODEDIR and OUTDIR will appear.
EXPDIR=$1

# The directory name where the content of the current directory will be cloned.
CODEDIR=$(basename $(pwd))
# The output directory of the experiment
# (his variables is hard coded bcs arguments after $1 are the command to run)
OUTDIR=out

# The source code is the current directory. Hack CODESRCDIR if you want otherwise.
CODESRCDIR=$(pwd)
echo -n Cloning \"$CODESRCDIR\" in \"$EXPDIR/$CODEDIR\"

# Create the destination directory
if [[ $EXPDIR = '/'* ]] ; then
    echo " (using file system)"
    mkdir -p $EXPDIR;
    mkdir -p $EXPDIR/$OUTDIR;
else
    echo " (using ssh on ${EXPDIR%:*})"
    ssh ${EXPDIR%:*} /bin/mkdir -p ${EXPDIR#*:}/$OUTDIR;
fi

# Do the actual copy using rsync to mirror exactely the source directory and
# avoid re-copying files already copied from any previous cloning.
# cp -fr $CODESRCDIR $EXPDIR
rsync -qav $CODESRCDIR/ $EXPDIR/$CODEDIR/ --exclude .git --exclude .git/


# # Copy a shallow version of any .git directory in the code sub-directory
# # The sub-directory is thus assumed to hold the .git dir !
# # (sub-directory bcs a python module is supposed to be package with module dir
# #  as a sub-module of the package dir. It's okay bcs the package dir becomes
# #  the EXPDIR directory and the git tree is thus still consistent).
# if [[ -d '../.git/' ]] ; then
#     echo "Copy a shallow git in the code directory"
#     rm -fr tmp_clone_gitstatedepth1;
#     mkdir -p tmp_clone_gitstatedepth1;
#     cd tmp_clone_gitstatedepth1;
#     git clone --depth 1 --quiet file://$CODESRCDIR/../ gitstatedepth1;
#     rsync -qav gitstatedepth1/.git/ $EXPDIR/.git/;
#     cd ..;
#     rm -fr tmp_clone_gitstatedepth1;
# fi

# Run the command that is after the clone.sh arguments
if [[ "${@:2}" ]]; then

    if [[ $EXPDIR = '/'* ]] ; then
        echo Run command: "${@:2}" " (using file system)"

        # Go into the output directory
        cd $EXPDIR/$OUTDIR

# (keep it unindented to avoid undesired empty spaces before the command)
# ${@:2} > log 2>&1
${@:2}

    else
        echo Run command: "${@:2}" " (using ssh on ${EXPDIR%:*})"

        ssh ${EXPDIR%:*} "cd ${EXPDIR#*:}/$OUTDIR; ${@:2}"

    fi

fi
