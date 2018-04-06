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

CODEDIR=percivaltts

# Maintenance targets ----------------------------------------------------------

.PHONY: tests tests_clean

all: run

describe:
	@git describe

clean: tests_clean
	rm -fr tests/slt_arctic_merlin_full/wav_* tests/slt_arctic_merlin_full/label_state_align_*

# Run targets ------------------------------------------------------------------

run:
	mkdir -p ../out; cd ../out; bash ../${CODEDIR}/"$(SETENVSCRIPT)" python ../${CODEDIR}/${FILETORUN}

run_continue:
	cd ../out; bash ../${CODEDIR}/"$(SETENVSCRIPT)" python ../${CODEDIR}/${FILETORUN} --continue

run_grid:
	mkdir -p ../out; cd ../out; "$(QSUBCMD)" ../${CODEDIR}/"$(SETENVSCRIPT)" python ../${CODEDIR}/${FILETORUN}

run_grid_continue:
	cd ../out; "$(QSUBCMD)" ../${CODEDIR}/"$(SETENVSCRIPT)" python ../${CODEDIR}/${FILETORUN} --continue

clone:
	@test "$(DEST)"
	./clone.sh "$(DEST)"

clone_run:
	@test "$(DEST)"
	./clone.sh "$(DEST)" bash ../${CODEDIR}/"$(SETENVSCRIPT)" python ../${CODEDIR}/${FILETORUN}

clone_run_grid:
	@test "$(DEST)"
	./clone.sh "$(DEST)" "$(QSUBCMD)" ../${CODEDIR}/"$(SETENVSCRIPT)" python ../${CODEDIR}/${FILETORUN}

# Testing ----------------------------------------------------------------------

# Download the full demo data (~1000 sentences)
tests/slt_arctic_merlin_full.tar.gz:
	git clone git@github.com:gillesdegottex/percivaltts_demodata.git
	mv percivaltts_demodata/* tests/
	rm -fr percivaltts_demodata

# Decompress the full demo data (~1000 sentences)
tests/slt_arctic_merlin_full: tests/slt_arctic_merlin_full.tar.gz
	tar xvzf tests/slt_arctic_merlin_full.tar.gz -C tests/

# Decompress the test data (10 sentences)
tests/slt_arctic_merlin_test: tests/slt_arctic_merlin_test.tar.gz
	tar xvf tests/slt_arctic_merlin_test.tar.gz -C tests/

tests: tests/slt_arctic_merlin_test
	python -m tests.test_base
	python -m tests.test_smoke
	bash "$(SETENVSCRIPT)" python -m tests.test_smoke_theano
	bash "$(SETENVSCRIPT)" python -m tests.test_run

tests_clean:
	rm -fr tests/slt_arctic_merlin_test
	rm -fr tests/test_made__*
