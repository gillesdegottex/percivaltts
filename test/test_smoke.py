# http://pymbook.readthedocs.io/en/latest/testing.html

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import unittest

cptest = 'test/slttest/'

class TestSmoke(unittest.TestCase):
    def test_utils(self):
        import utils
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
        # utils.print_sysinfo_theano() TODO
        # utils.log_plot_costs(costs_tra, costs_val, worst_val, fname, epochs_modelssaved, costs_discri=[]) TODO
        # utils.log_plot_costs(costs, worst_val, fname, epochs_modelssaved) TODO
        # utils.log_plot_samples() TODO

    def test_data(self):
        import data

        fbases = data.loadids(cptest+'file_id_list.scp')

        path, shape = data.getpathandshape('dummy.fwspec')
        self.assertTrue(path=='dummy.fwspec')
        self.assertTrue(shape==None)
        path, shape = data.getpathandshape('dummy.fwspec:(-1,129)')
        self.assertTrue(path=='dummy.fwspec')
        self.assertTrue(shape==(-1,129))
        path, shape = data.getpathandshape('dummy.fwspec:(-1,129)', (-1,12))
        self.assertTrue(path=='dummy.fwspec')
        self.assertTrue(shape==(-1,12))
        path, shape = data.getpathandshape('dummy.fwspec', (-1,12))
        self.assertTrue(path=='dummy.fwspec')
        self.assertTrue(shape==(-1,12))
        dim = data.getlastdim('dummy.fwspec')
        self.assertTrue(dim==1)
        dim = data.getlastdim('dummy.fwspec:(-1,129)')
        self.assertTrue(dim==129)

        Xs = data.load(cptest+'binary_label_601/*.lab:(-1,601)', fbases, shape=None, frameshift=0.005, verbose=1, label='testload ')
        self.assertTrue(len(Xs)==10)
        self.assertTrue(Xs[0].shape==(666, 601))

        self.assertTrue(data.gettotallen(Xs)==5688)

        Xs, Xs = data.cropsize([Xs, Xs]) # TODO Crop against features

        # data.cropsilences TODO
        # data.vstack_masked TODO
        # data.maskify TODO
        # data.addstop TODO
        # data.load_inoutset TODO
        # data.cost_0pred_rmse TODO
        # data.cost_model TODO
        # data.cost_model_prediction_rmse TODO
        # data.prediction_std TODO

if __name__ == '__main__':
    unittest.main()
