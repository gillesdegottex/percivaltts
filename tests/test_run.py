# http://pymbook.readthedocs.io/en/latest/testing.html

import os
from percivaltts import *

import unittest

class TestRun(unittest.TestCase):

    def test_run(self):

        os.system('ls -l')
        os.system('ls -l tests')

        # Fool the demo data with the test data
        # Because running the full demo on travis is not possible
        if not os.path.exists('tests/slt_arctic_merlin_full'):
            os.symlink('slt_arctic_merlin_test', 'tests/slt_arctic_merlin_full')
        os.listdir('tests/slt_arctic_merlin_full')
        os.system('ls -l tests/slt_arctic_merlin_full')

        import percivaltts.run as run

        print('Overwrite the configuration to run a smoke test')
        run.cfg.id_valid_start = 8
        run.cfg.id_valid_nb = 1
        run.cfg.id_test_nb = 1

        run.cfg.train_min_nbepochs = 1
        run.cfg.train_max_nbepochs = 10
        run.cfg.train_cancel_nodecepochs = 3

        run.cfg.train_batch_size = 2
        run.cfg.arch_hiddenwidth = 4
        run.cfg.arch_gen_nbcnnlayers = 1

        run.cfg.print_content()

        run.features_extraction()
        run.contexts_extraction()
        run.training(cont=False)
        run.generate()

if __name__ == '__main__':
    unittest.main()
