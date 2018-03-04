# http://pymbook.readthedocs.io/en/latest/testing.html

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import unittest

class TestSmokeTheano(unittest.TestCase):
    def test_model(self):
        class ModelSmoke(model.Model):
            def train(self, params, indir, outdir, outwdir, fid_lst_tra, fid_lst_val, X_vals, Y_vals, cfg, params_savefile, trialstr='', cont=None):
                raise ValueError('That\'s a smoky model that doesn\'t train anything')

        mod = ModelSmoke()

    def test_utils_theano(self):
        import utils_theano

        print(utils_theano.th_memfree())
        print(utils_theano.nvidia_smi_current_gpu())
        print(utils_theano.nvidia_smi_proc_memused())

        # th_print(msg, op) # TODO
        # paramss_count(paramss) # TODO
        # linear_and_bndnmoutput_deltas_tanh(x, specsize, nmsize) # TODO
        # linear_nmsigmoid(x, specsize, nmsize) # TODO
        # nonlin_tanh_saturated(x, coef=1.01) # TODO
        # nonlin_tanh_bysigmoid(x) # TODO
        # nonlin_tanhcm11(x) # TODO
        # nonlin_saturatedsigmoid(x, coef=1.01) # TODO
        # nonlin_sigmoidbinary(x) # TODO
        # nonlin_softsign(x) # TODO
        # nonlin_sigmoidparm(x, c=0.0, f=1.0) # TODO


if __name__ == '__main__':
    unittest.main()
