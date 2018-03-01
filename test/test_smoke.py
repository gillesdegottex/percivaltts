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

        indir = cptest+'binary_label_601_norm_minmaxm11/*.lab:(-1,601)'
        Xs = data.load(indir, fbases, shape=None, frameshift=0.005, verbose=1, label='Xs: ')
        self.assertTrue(len(Xs)==10)
        self.assertTrue(Xs[0].shape==(666, 601))

        self.assertTrue(data.gettotallen(Xs)==5688)

        outdir = cptest+'wav_cmp_lf0_fwspec65_fwnm17_bndnmnoscale/*.cmp:(-1,83)'
        Ys = data.load(outdir, fbases, shape=None, frameshift=0.005, verbose=1, label='Ys: ')
        self.assertTrue(len(Ys)==10)
        self.assertTrue(Ys[0].shape==(664, 83))

        wdir = cptest+'wav_fwspec65_weights/*.w:(-1,1)'
        Ws = data.load(wdir, fbases, shape=None, frameshift=0.005, verbose=1, label='Ws: ')
        self.assertTrue(len(Ws)==10)

        Xs, Ys, Ws = data.cropsize([Xs, Ys, Ws])

        [Xs, Ys], Ws = data.cropsilences([Xs, Ys], Ws, thresh=0.5)

        # data.vstack_masked TODO
        # data.maskify TODO
        # data.addstop TODO
        # data.load_inoutset TODO

        X_full, MX_full, Y_full, MY_full = data.load_inoutset(indir, outdir, wdir, fbases, length=None, lengthmax=100, maskpadtype='randshift')

        worst_val = data.cost_0pred_rmse(Ys)
        print('worst_val={}'.format(worst_val))
        # data.cost_model TODO
        # data.cost_model_prediction_rmse TODO
        # data.prediction_std TODO

if __name__ == '__main__':
    unittest.main()
