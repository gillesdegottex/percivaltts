# http://pymbook.readthedocs.io/en/latest/testing.html

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import unittest

class TestRun(unittest.TestCase):

    def test_run(self):
        import run
        run.features_extraction()
        run.composition()
        # run.training(cont=False)
        # run.generate_wavs()

if __name__ == '__main__':
    unittest.main()
