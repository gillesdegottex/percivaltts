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
#SETENVSCRIPT=setenv_home.sh
#SETENVSCRIPT=setenv_cued.sh
QSUBCMD="qsub -l gpu=1 -j y -cwd -S /bin/bash"
# QSUBCMD="qsubcudagad.sh"

# Maintenance targets ----------------------------------------------------------

.PHONY: test

all: build

submodule_init:
	git submodule update --init --recursive

build: submodule_init build_pulsemodel

build_pulsemodel: submodule_init
	cd external/pulsemodel; make

describe:
	@git describe

distclean:
	cd external/pulsemodel; $(MAKE) distclean
	find . -name '*.pyc' -delete
	rm -fr test/slttest

# Run targets ------------------------------------------------------------------

run:
	mkdir -p ../out; cd ../out; bash ../Code/"$(SETENVSCRIPT)" ../Code/${FILETORUN}

run_continue:
	cd ../out; bash ../Code/"$(SETENVSCRIPT)" ../Code/${FILETORUN} --continue

run_grid:
	mkdir -p ../out; cd ../out; "$(QSUBCMD)" ../Code/"$(SETENVSCRIPT)" ../Code/${FILETORUN}

run_grid_continue:
	cd ../out; "$(QSUBCMD)" ../Code/"$(SETENVSCRIPT)" ../Code/${FILETORUN} --continue

clone:
	@test "$(DEST)"
	./clone.sh "$(DEST)"

clone_run:
	@test "$(DEST)"
	./clone.sh "$(DEST)" bash ../Code/"$(SETENVSCRIPT)" ../Code/${FILETORUN}

clone_run_grid:
	@test "$(DEST)"
	./clone.sh "$(DEST)" "$(QSUBCMD)" ../Code/"$(SETENVSCRIPT)" ../Code/${FILETORUN}

generate:
	bash "$(SETENVSCRIPT)" generate.py ../out/model.pkl


# Testing ----------------------------------------------------------------------

test/slttest: test/slttest.tar.gz
	tar xvf test/slttest.tar.gz -C test/

test: build test/slttest
	python test/test_base.py
	python test/test_smoke.py
	# python test/test_run.py
