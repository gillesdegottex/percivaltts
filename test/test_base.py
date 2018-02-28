# http://pymbook.readthedocs.io/en/latest/testing.html

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import unittest

class TestBase(unittest.TestCase):

    def test_base(self):
        import run
        run.features_extraction()
        run.composition()

if __name__ == '__main__':
    unittest.main()
