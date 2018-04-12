# http://pymbook.readthedocs.io/en/latest/testing.html

from percivaltts import *

import unittest

import numpy as np
numpy_force_random_seed()

cptest = 'tests/slt_arctic_merlin_test/'
lab_size = 425
spec_size = 65
nm_size = 17


class TestSmoke(unittest.TestCase):
    def test_percivaltts(self):
        import percivaltts

        cfg = percivaltts.configuration()

        text_file = open(cptest+'/info.py', "w")
        text_file.write("fs = 32000\n")
        text_file.write("shift = 0.005\n")
        text_file.close()
        cfg.mergefiles([cptest+'/info.py'])
        cfg.mergefiles(cptest+'/info.py')

        percivaltts.print_log('print_log')

        percivaltts.print_tty('print_tty', end='\n')

        print(percivaltts.datetime2str(sec=1519426184))

        print(percivaltts.time2str(sec=1519426184))

        percivaltts.makedirs('/dev/null')

        self.assertTrue(percivaltts.is_int('74324'))
        self.assertFalse(percivaltts.is_int('743.24'))

        rng = np.random.RandomState(123)
        percivaltts.weights_normal_ortho(32, 64, 1.0, rng, dtype=np.float32)

        memres = percivaltts.proc_memresident()
        print(memres)
        self.assertNotEqual(memres, -1)

        percivaltts.print_sysinfo()

        # percivaltts.print_sysinfo_theano() TODO
        # percivaltts.log_plot_costs(costs_tra, costs_val, worst_val, fname, epochs_modelssaved, costs_discri=[]) TODO
        # percivaltts.log_plot_costs(costs, worst_val, fname, epochs_modelssaved) TODO
        # percivaltts.log_plot_samples() TODO

    def test_data(self):
        import data

        fids = readids(cptest+'file_id_list.scp')

        path, shape = data.getpathandshape('dummy.fwlspec')
        self.assertTrue(path=='dummy.fwlspec')
        self.assertTrue(shape==None)
        path, shape = data.getpathandshape('dummy.fwlspec:(-1,129)')
        self.assertTrue(path=='dummy.fwlspec')
        self.assertTrue(shape==(-1,129))
        path, shape = data.getpathandshape('dummy.fwlspec:(-1,129)', (-1,12))
        self.assertTrue(path=='dummy.fwlspec')
        self.assertTrue(shape==(-1,12))
        path, shape = data.getpathandshape('dummy.fwlspec', (-1,12))
        self.assertTrue(path=='dummy.fwlspec')
        self.assertTrue(shape==(-1,12))
        dim = data.getlastdim('dummy.fwlspec')
        self.assertTrue(dim==1)
        dim = data.getlastdim('dummy.fwlspec:(-1,129)')
        self.assertTrue(dim==129)

        indir = cptest+'binary_label_'+str(lab_size)+'_norm_minmaxm11/*.lab:(-1,'+str(lab_size)+')'
        Xs = data.load(indir, fids, shape=None, frameshift=0.005, verbose=1, label='Xs: ')
        self.assertTrue(len(Xs)==10)
        print(Xs[0].shape)
        self.assertTrue(Xs[0].shape==(667, lab_size))

        print(data.gettotallen(Xs))
        self.assertTrue(data.gettotallen(Xs)==5694)

        outdir = cptest+'wav_cmp_lf0_fwlspec65_fwnm17_bndnmnoscale/*.cmp:(-1,83)'
        Ys = data.load(outdir, fids, shape=None, frameshift=0.005, verbose=1, label='Ys: ')
        print(len(Ys))
        self.assertTrue(len(Ys)==10)
        print(Ys[0].shape)
        self.assertTrue(Ys[0].shape==(666, 83))

        wdir = cptest+'wav_fwlspec65_weights/*.w:(-1,1)'
        Ws = data.load(wdir, fids, shape=None, frameshift=0.005, verbose=1, label='Ws: ')
        self.assertTrue(len(Ws)==10)

        Xs, Ys, Ws = data.croplen([Xs, Ys, Ws])

        [Xs, Ys], Ws = data.croplen_weight([Xs, Ys], Ws, thresh=0.5)

        Xs_w_stop = data.addstop(Xs)

        X_train, MX_train, Y_train, MY_train = data.load_inoutset(indir, outdir, wdir, fids, length=None, lengthmax=100, maskpadtype='randshift', inouttimesync=False)
        X_train, MX_train, Y_train, MY_train = data.load_inoutset(indir, outdir, wdir, fids, length=None, lengthmax=100, maskpadtype='randshift')

        worst_val = data.cost_0pred_rmse(Ys)
        print('worst_val={}'.format(worst_val))

        worst_val = data.cost_0pred_rmse(Ys[0])
        print('worst_val={}'.format(worst_val))

        def data_cost_model_mfn(Xs, Ys):
            return np.std(Ys) # TODO More usefull
        X_vals = data.load(indir, fids)
        Y_vals = data.load(outdir, fids)
        X_vals, Y_vals = data.croplen([X_vals, Y_vals])
        cost = data.cost_model_mfn(data_cost_model_mfn, [X_vals, Y_vals])
        print(cost)

        class SmokyModel:
            def predict(self, Xs):
                return np.zeros([1, Xs.shape[1], 83])
        mod = SmokyModel()
        cost = data.cost_model_prediction_rmse(mod, [Xs], Ys)
        print(cost)

        std = data.prediction_mstd(mod, [Xs])
        print(std)

        rms = data.prediction_rms(mod, [Xs])
        print(rms)

    def test_compose(self):
        import data
        import compose

        fids = readids(cptest+'/file_id_list.scp')

        wav_dir = 'wav'
        f0_path = cptest+wav_dir+'_lf0/*.lf0'
        spec_path = cptest+wav_dir+'_fwlspec'+str(spec_size)+'/*.fwlspec'
        nm_path = cptest+wav_dir+'_fwnm'+str(nm_size)+'/*.fwnm'

        compose.compose([cptest+'binary_label_'+str(lab_size)+'/*.lab:(-1,'+str(lab_size)+')'], fids, 'tests/test_made__smoke_compose_compose_lab0/*.lab', id_valid_start=8, normfn=None, do_finalcheck=True, wins=[], dropzerovardims=False)

        compose.compose([cptest+'binary_label_'+str(lab_size)+'/*.lab:(-1,'+str(lab_size)+')'], fids, 'tests/test_made__smoke_compose_compose_lab1/*.lab', id_valid_start=8, normfn=compose.normalise_minmax, do_finalcheck=True, wins=[], dropzerovardims=False)

        path2, shape2 = data.getpathandshape('tests/test_made__smoke_compose_compose_lab1/*.lab:(mean.dat,'+str(lab_size)+')')

        compose.compose([cptest+'binary_label_'+str(lab_size)+'/*.lab:(-1,'+str(lab_size)+')'], fids, 'tests/test_made__smoke_compose_compose_lab2/*.lab', id_valid_start=8, normfn=compose.normalise_minmax, do_finalcheck=True, wins=[], dropzerovardims=True)

        compose.compose([f0_path, spec_path+':(-1,'+str(spec_size)+')', nm_path+':(-1,'+str(nm_size)+')'], fids, 'tests/test_made__smoke_compose_compose2_cmp1/*.cmp', id_valid_start=8, normfn=compose.normalise_minmax, do_finalcheck=True, wins=[])

        compose.compose([f0_path, spec_path+':(-1,'+str(spec_size)+')', nm_path+':(-1,'+str(nm_size)+')'], fids, 'tests/test_made__smoke_compose_compose2_cmp2/*.cmp', id_valid_start=8, normfn=compose.normalise_meanstd, do_finalcheck=True, wins=[])

        compose.compose([f0_path, spec_path+':(-1,'+str(spec_size)+')', nm_path+':(-1,'+str(nm_size)+')'], fids, 'tests/test_made__smoke_compose_compose2_cmp3/*.cmp', id_valid_start=8, normfn=compose.normalise_meanstd_nmminmaxm11, do_finalcheck=True, wins=[])

        compose.compose([f0_path, spec_path+':(-1,'+str(spec_size)+')', nm_path+':(-1,'+str(nm_size)+')'], fids, 'tests/test_made__smoke_compose_compose2_cmp3_deltas/*.cmp', id_valid_start=8, normfn=compose.normalise_meanstd_nmminmaxm11, do_finalcheck=True, wins=[[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]])

        compose.compose([f0_path, spec_path+':(-1,'+str(spec_size)+')', nm_path+':(-1,'+str(nm_size)+')'], fids, 'tests/test_made__smoke_compose_compose2_cmp4/*.cmp', id_valid_start=8, normfn=compose.normalise_meanstd_nmnoscale, do_finalcheck=True, wins=[])

        compose.compose([f0_path, spec_path+':(-1,'+str(spec_size)+')', nm_path+':(-1,'+str(nm_size)+')'], fids, 'tests/test_made__smoke_compose_compose2_cmp_deltas/*.cmp', id_valid_start=8, normfn=compose.normalise_meanstd_nmnoscale, do_finalcheck=True, wins=[[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]])

        compose.create_weights(spec_path+':(-1,'+str(spec_size)+')', fids, 'tests/test_made__smoke_compose_compose2_w1/*.w', spec_type='fwlspec', thresh=-32)


if __name__ == '__main__':
    unittest.main()
