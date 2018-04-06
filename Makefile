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

# Maintenance targets ----------------------------------------------------------

.PHONY: test test_clean

all: build

submodule_init:
	git submodule update --init --recursive

build_pulsemodel: submodule_init
	cd percival-tts/external/pulsemodel; make

build: submodule_init build_pulsemodel

describe:
	@git describe

clean: test_clean
	rm -fr percival-tts/tests/slt_arctic_merlin_full/wav_* percival-tts/tests/slt_arctic_merlin_full/label_state_align_*

distclean: test_clean
	cd percival-tts/external/pulsemodel; $(MAKE) distclean
	# TODO Clean REAPER
	# TODO Clean WORLD
	# TODO Clean sigproc
	find . -name '*.pyc' -delete
