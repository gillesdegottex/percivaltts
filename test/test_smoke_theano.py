# http://pymbook.readthedocs.io/en/latest/testing.html

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import unittest

from utils import *


class TestSmokeTheano(unittest.TestCase):
    def test_model(self):
        makedirs('test/test_made__smoke_theano_model')

        import model
        class ModelSmoke(model.Model):
            def train(self, params, indir, outdir, outwdir, fid_lst_tra, fid_lst_val, X_vals, Y_vals, cfg, params_savefile, trialstr='', cont=None):
                raise ValueError('That\'s a smoky model that doesn\'t train anything')
        mod = ModelSmoke()


        import model_gan
        modgan = model_gan.ModelGAN(601, 65, 17)

        print("modgan.nbParams={}".format(modgan.nbParams()))

        cfg = configuration() # Init configuration structure
        cfg.smokyconfiguration = 64
        cfg.train_hypers = []
        cost_val = 67.43
        modgan.saveAllParams('test/test_made__smoke_theano_model/smokymodelparams.pkl')
        modgan.saveAllParams('test/test_made__smoke_theano_model/smokymodelparams.pkl', cfg=cfg, extras={'cost_val':cost_val})
        modgan.saveAllParams('test/test_made__smoke_theano_model/smokymodelparams.pkl', cfg=cfg, extras={'cost_val':cost_val})

        cfg_loaded, extras_loaded = modgan.loadAllParams('test/test_made__smoke_theano_model/smokymodelparams.pkl')
        self.assertEqual(cfg, cfg_loaded)
        self.assertEqual({'cost_val':cost_val}, extras_loaded)


        modgan.saveTrainingState('test/test_made__smoke_theano_model/smokytrainingstate.pkl', cfg=cfg, extras={'cost_val':cost_val})

        cfg_loaded, extras_loaded = modgan.loadTrainingState('test/test_made__smoke_theano_model/smokytrainingstate.pkl', cfg=cfg)
        self.assertEqual(cfg, cfg_loaded)
        self.assertEqual({'cost_val':cost_val}, extras_loaded)

        cfg, hyperstr = modgan.randomize_hyper(cfg)
        print('randomize_hyper: hyperstr='+hyperstr)
        cfg.print_content()

        cfg.train_hypers = [('train_learningrate_log10', -6.0, -2.0), ('train_adam_beta1', 0.8, 1.0)] # For ADAM
        cfg_hyprnd1, hyperstr1 = modgan.randomize_hyper(cfg)
        print('randomize_hyper: hyperstr1='+hyperstr1)
        cfg_hyprnd1.print_content()
        cfg_hyprnd2, hyperstr2 = modgan.randomize_hyper(cfg)
        print('randomize_hyper: hyperstr2='+hyperstr2)
        cfg_hyprnd2.print_content()
        self.assertNotEqual(cfg_hyprnd1, cfg_hyprnd2)

        # def train_multipletrials(self, indir, outdir, outwdir, fid_lst_tra, fid_lst_val, params, params_savefile, cfgtomerge=None, cont=None, **kwargs) # TODO

        # def generate(self, params_savefile, outsuffix, cfg, do_objmeas=True, do_resynth=True, indicestosynth=None
        #         , spec_comp='fwspec'
        #         , spec_size=129
        #         , nm_size=33
        #         , do_mlpg=False
        #         , pp_mcep=False
        #         , pp_spec_pf_coef=-1 # Common value is 1.2
        #         , pp_spec_extrapfreq=-1
        #         ) # TODO

        # cfg = configuration() # Init configuration structure
        # cfg.print_content()
        # modgan.train_multipletrials(cfg.indir, cfg.outdir, cfg.outwdir, fid_lst_tra, fid_lst_val, mod.params_trainable, 'smokymodelparams.pkl', cfgtomerge=cfg, cont=cont) # TODO


    def test_utils_theano(self):
        import utils_theano

        import theano.tensor as T

        # Test the following if CUDA is available: (won't be tested on travis anyway since no GPU are available on travis)
        if utils_theano.th_cuda_available():
            print('th_cuda_memfree={}'.format(utils_theano.th_cuda_memfree())) # Can't test it because needs CUDA
            print('nvidia_smi_current_gpu={}'.format(nvidia_smi_current_gpu()))  # Can't test it because needs CUDA
            print('nvidia_smi_gpu_memused={}'.format(nvidia_smi_gpu_memused())) # Can't test it because needs CUDA

        # th_print(msg, op) # TODO
        # paramss_count(paramss) # TODO
        # linear_and_bndnmoutput_deltas_tanh(x, specsize, nmsize) # TODO
        # linear_nmsigmoid(x, specsize, nmsize) # TODO

        x = T.ftensor3('x')

        y = utils_theano.nonlin_tanh_saturated(x, coef=1.01)
        y = utils_theano.nonlin_tanh_bysigmoid(x)
        y = utils_theano.nonlin_tanhcm11(x)
        y = utils_theano.nonlin_saturatedsigmoid(x, coef=1.01)
        y = utils_theano.nonlin_sigmoidbinary(x)
        y = utils_theano.nonlin_softsign(x)
        y = utils_theano.nonlin_sigmoidparm(x, c=0.0, f=1.0)


if __name__ == '__main__':
    unittest.main()
