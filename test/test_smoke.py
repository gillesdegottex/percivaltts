# http://pymbook.readthedocs.io/en/latest/testing.html

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import unittest

import utils

class TestUtils(unittest.TestCase):

    def test_utils(self):
        cfg = utils.configuration()
        utils.print_log('print_log')
        utils.print_tty('print_tty', end='\n')
        print(utils.datetime2str(sec=1519426184))
        print(utils.time2str(sec=1519426184))
        utils.makedirs('/dev/null')
        self.assertTrue(utils.is_int('74324'))
        self.assertFalse(utils.is_int('743.24'))
        memres = utils.proc_memresident()
        print(memres)
        self.assertNotEqual(memres, -1)
        utils.print_sysinfo()
        # utils.print_sysinfo_theano()
        # utils.log_plot_costs(costs_tra, costs_val, worst_val, fname, epochs_modelssaved, costs_discri=[])
        # utils.log_plot_costs(costs, worst_val, fname, epochs_modelssaved)
        # utils.log_plot_samples()

if __name__ == '__main__':
    unittest.main()
