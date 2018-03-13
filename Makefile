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
# SETENVSCRIPT=setenv_travis.sh
QSUBCMD="qsub -l gpu=1 -j y -cwd -S /bin/bash"
CODEDIR=percival

# Maintenance targets ----------------------------------------------------------

.PHONY: test test_clean

all: build

submodule_init:
	git submodule update --init --recursive

build: submodule_init build_pulsemodel

build_pulsemodel: submodule_init
	cd external/pulsemodel; make

describe:
	@git describe

clean: test_clean
	rm -fr test/slt_arctic_merlin_full/wav_* test/slt_arctic_merlin_full/label_state_align_*

distclean: test_clean
	cd external/pulsemodel; $(MAKE) distclean
	# TODO Clean REAPER
	# TODO Clean WORLD
	# TODO Clean sigproc
	find . -name '*.pyc' -delete

# Run targets ------------------------------------------------------------------

run:
	mkdir -p ../out; cd ../out; bash ../${CODEDIR}/"$(SETENVSCRIPT)" ../${CODEDIR}/${FILETORUN}

run_continue:
	cd ../out; bash ../${CODEDIR}/"$(SETENVSCRIPT)" ../${CODEDIR}/${FILETORUN} --continue

run_grid:
	mkdir -p ../out; cd ../out; "$(QSUBCMD)" ../${CODEDIR}/"$(SETENVSCRIPT)" ../${CODEDIR}/${FILETORUN}

run_grid_continue:
	cd ../out; "$(QSUBCMD)" ../${CODEDIR}/"$(SETENVSCRIPT)" ../${CODEDIR}/${FILETORUN} --continue

clone:
	@test "$(DEST)"
	./clone.sh "$(DEST)"

clone_run:
	@test "$(DEST)"
	./clone.sh "$(DEST)" bash ../${CODEDIR}/"$(SETENVSCRIPT)" ../${CODEDIR}/${FILETORUN}

clone_run_grid:
	@test "$(DEST)"
	./clone.sh "$(DEST)" "$(QSUBCMD)" ../${CODEDIR}/"$(SETENVSCRIPT)" ../${CODEDIR}/${FILETORUN}

generate:
	bash "$(SETENVSCRIPT)" generate.py ../out/model.pkl

# Testing ----------------------------------------------------------------------

# Download the full demo data (~1000 sentences)
test/slt_arctic_merlin_full.tar.gz:
	git clone git@github.com:gillesdegottex/percival_demo_data.git
	mv percival_demo_data/* test/
	rm -fr percival_demo_data

# Decompress the full demo data (~1000 sentences)
test/slt_arctic_merlin_full: test/slt_arctic_merlin_full.tar.gz
	tar xvzf test/slt_arctic_merlin_full.tar.gz -C test/

# Decompress the test data (10 sentences)
test/slt_arctic_merlin_test: test/slt_arctic_merlin_test.tar.gz
	tar xvf test/slt_arctic_merlin_test.tar.gz -C test/

run_demo: build test/slt_arctic_merlin_full run


test: build test/slt_arctic_merlin_test
	python test/test_base.py
	python test/test_smoke.py
	bash "$(SETENVSCRIPT)" test/test_smoke_theano.py
	bash "$(SETENVSCRIPT)" test/test_run.py

test_clean:
	rm -fr test/slt_arctic_merlin_test
	rm -fr test/test_made__*
