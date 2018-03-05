# http://pymbook.readthedocs.io/en/latest/testing.html

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import unittest

class TestSmokeTheano(unittest.TestCase):
    def test_model(self):

        import model
        class ModelSmoke(model.Model):
            def train(self, params, indir, outdir, outwdir, fid_lst_tra, fid_lst_val, X_vals, Y_vals, cfg, params_savefile, trialstr='', cont=None):
                raise ValueError('That\'s a smoky model that doesn\'t train anything')

        mod = ModelSmoke()

        # def nbParams(self) # TODO

        # def saveAllParams(self, fmodel, cfg=None, extras=dict(), printfn=print) # TODO

        # def loadAllParams(self, fmodel, printfn=print) # TODO

        # def saveTrainingState(self, fstate, cfg=None, extras=dict(), printfn=print) # TODO

        # def loadTrainingState(self, fstate, cfg, printfn=print) # TODO

        # def randomize_hyper(self, cfg) # TODO

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

        import model_gan

        modgan = model_gan.ModelGAN(601, 65, 17)

        # cfg = configuration() # Init configuration structure
        # cfg.print_content()
        # modgan.train_multipletrials(cfg.indir, cfg.outdir, cfg.outwdir, fid_lst_tra, fid_lst_val, mod.params_trainable, 'smokymodelparams.pkl', cfgtomerge=cfg, cont=cont) # TODO


    def test_utils_theano(self):
        import utils_theano

        import theano.tensor as T

        # TODO Test if CUDA is available and test the following if yes:
        # print(utils_theano.th_memfree()) # Can't test it because needs CUDA
        # print(utils_theano.nvidia_smi_current_gpu())  # Can't test it because needs CUDA
        # print(utils_theano.nvidia_smi_proc_memused()) # Can't test it because needs CUDA

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
