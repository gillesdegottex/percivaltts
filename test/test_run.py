# http://pymbook.readthedocs.io/en/latest/testing.html

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import *

import unittest

class TestRun(unittest.TestCase):

    def test_run(self):
        import run

        print('Overwrite the configuration to run a smoke test')    # TODO Test the demo ?
        run.cfg.id_valid_start = 8
        run.cfg.id_valid_nb = 1
        run.cfg.id_test_nb = 1
        run.cfg.train_batchsize = 2
        run.cfg.train_max_nbepochs = 5
        run.cfg.model_hiddensize = 4
        run.cfg.model_nbprelayers = 1
        run.cfg.model_nbcnnlayers = 1
        run.cfg.model_nbfilters = 2
        run.cfg.model_spec_freqlen = 3
        run.cfg.model_nm_freqlen = 3
        run.cfg.model_windur = 0.020

        run.cfg.print_content()

        run.features_extraction()
        run.composition()
        run.training(cont=False)
        run.generate_wavs('model.pkl.last')

if __name__ == '__main__':
    unittest.main()
