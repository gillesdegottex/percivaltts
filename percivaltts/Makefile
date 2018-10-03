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

FILETORUN=run.py
SETENVSCRIPT=setenv.sh
QSUBCMD="qsub -l gpu=1 -j y -cwd -S /bin/bash"

CODEDIR=$(shell basename $(shell pwd))
CMDTORUN=../${CODEDIR}/$(SETENVSCRIPT) python ../${CODEDIR}/${FILETORUN}
CMDTORUN_CONTINUE=${CMDTORUN} --continue

# Maintenance targets ----------------------------------------------------------

.PHONY: clean describe tests tests_clean

all: run

describe:
	@git describe

clean: tests_clean
	rm -fr tests/slt_arctic_merlin_full/wav_* tests/slt_arctic_merlin_full/label_state_align_*

# Run targets ------------------------------------------------------------------

run:
	mkdir -p ../out; cd ../out; bash ${CMDTORUN}

run_continue:
	mkdir -p ../out; cd ../out; bash ${CMDTORUN_CONTINUE}

run_grid:
	mkdir -p ../out; cd ../out; "$(QSUBCMD)" ${CMDTORUN}

run_grid_continue:
	mkdir -p ../out; cd ../out; "$(QSUBCMD)" ${CMDTORUN_CONTINUE}

clone:
	@test "$(DEST)"
	./clone.sh "$(DEST)"

clone_run:
	@test "$(DEST)"
	./clone.sh "$(DEST)" bash ${CMDTORUN}

clone_run_continue:
	@test "$(DEST)"
	./clone.sh "$(DEST)" bash ${CMDTORUN_CONTINUE}

clone_run_grid:
	@test "$(DEST)"
	./clone.sh "$(DEST)" "$(QSUBCMD)" ${CMDTORUN}

clone_run_grid_continue:
	@test "$(DEST)"
	./clone.sh "$(DEST)" "$(QSUBCMD)" ${CMDTORUN_CONTINUE}

# Run in background
run_back:
	mkdir -p ../out; cd ../out; nohup ${CMDTORUN} </dev/null >> bck.log 2>&1 &

run_back_continue:
	mkdir -p ../out; cd ../out; nohup ${CMDTORUN_CONTINUE} </dev/null >> bck.log 2>&1 &

# Clone on given path and run in background
# TODO This works only with ssh cloning (i.e. by specifying a hostname before
# the path) for obscure nohup-bash-arguments reasons.
clone_run_back:
	./clone.sh "$(DEST)" 'nohup ${CMDTORUN} </dev/null >> bck.log 2>&1 &'

clone_run_back_continue:
	./clone.sh "$(DEST)" 'nohup ${CMDTORUN_CONTINUE} </dev/null >> bck.log 2>&1 &'
