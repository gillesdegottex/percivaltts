# http://pymbook.readthedocs.io/en/latest/testing.html

from percivaltts import *

import unittest

import data

global cfg
cfg = configuration() # Init configuration structure

cptest = 'tests/slt_arctic_merlin_test/' # The main directory where the data of the voice is stored
lab_size = 425
spec_size = 65
nm_size = 17
cfg.vocoder_shift = 0.005   # Time shift between 2 frames
cfg.vocoder_fs = 16000      # Sampling frequency of the samples used for testing

cfg.fileids = cptest+'/file_id_list.scp'
fid_lst = readids(cfg.fileids)
cfg.id_valid_start = 8
cfg.id_valid_nb = 1
cfg.id_test_nb = 1
cfg.indir = cptest+'binary_label_'+str(lab_size)+'_norm_minmaxm11/*.lab:(-1,'+str(lab_size)+')'
cfg.outdir = cptest+'wav_cmp_lf0_fwlspec65_fwnm17_bndnmnoscale/*.cmp:(-1,83)'
cfg.wdir = cptest+'wav_fwlspec65_weights/*.w:(-1,1)'

cfg.train_batch_size = 2
cfg.train_cancel_nodecepochs = 2

fid_lst_tra = fid_lst[:cfg.id_train_nb()]
fid_lst_val = fid_lst[cfg.id_valid_start:cfg.id_valid_start+cfg.id_valid_nb]

cfg.dummyattribute = -1

class TestSmokeTheano(unittest.TestCase):
    def test_model(self):
        makedirs('tests/test_made__smoke_theano_model')
        makedirs('tests/test_made__smoke_theano_model_train')
        makedirs('tests/test_made__smoke_theano_model_train_vocoder_WORLD')

        fid_lst = readids(cfg.fileids)

        import vocoders
        vocoder = vocoders.VocoderPML(cfg.vocoder_fs, cfg.vocoder_shift, spec_size, nm_size)

        # Most basic model
        import models_basic
        model = models_basic.ModelFC(lab_size, vocoder, mlpg_wins=[], hiddensize=4, nblayers=2)
        print("modgan.nbParams={}".format(model.nbParams()))
        self.assertEqual(model.nbParams(), 2163)

        # LSE optimizer
        import optimizer
        cfg.train_max_nbepochs = 5
        cfg.train_nbtrials = 1        # Just run one training only
        cfg.train_hypers = []
        cfg.cropmode = 'begend'
        optigan = optimizer.Optimizer(model, errtype='LSE')
        optigan.train_multipletrials(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, model.params_trainable, 'tests/test_made__smoke_theano_model_train/smokymodelparams.pkl', cfgtomerge=cfg, cont=False)
        model.saveAllParams('tests/test_made__smoke_theano_model/smokymodelparams.pkl')

        # Generate waveforms
        model.generate_wav(cfg.indir, cfg.outdir, fid_lst, 'tests/test_made__smoke_theano_model_train/smokymodelparams-snd', cfg, vocoder, wins=[], do_objmeas=True, do_resynth=True)


        # Extra test along side the simple test above
        # Saving ...
        global cfg
        cost_val = 67.43
        model.saveAllParams('tests/test_made__smoke_theano_model/smokymodelparams.pkl')
        model.saveAllParams('tests/test_made__smoke_theano_model/smokymodelparams.pkl', cfg=cfg, extras={'cost_val':cost_val})

        # Loading ...
        cfg_loaded, extras_loaded = model.loadAllParams('tests/test_made__smoke_theano_model/smokymodelparams.pkl')
        self.assertEqual(cfg, cfg_loaded)
        self.assertEqual({'cost_val':cost_val}, extras_loaded)

        # Save training state
        import optimizer
        optigan.saveTrainingState('tests/test_made__smoke_theano_model/smokytrainingstate.pkl', cfg=cfg, extras={'cost_val':cost_val})

        # Load training state
        cfg_loaded, extras_loaded, rngstate = optigan.loadTrainingState('tests/test_made__smoke_theano_model/smokytrainingstate.pkl', cfg=cfg)
        self.assertEqual(cfg, cfg_loaded)
        self.assertEqual({'cost_val':cost_val}, extras_loaded)

        # Test empty hyper parameters
        cfg, hyperstr = optigan.randomize_hyper(cfg)
        print('randomize_hyper: hyperstr='+hyperstr)
        cfg.print_content()

        # Test multi hyper parameters
        cfg.train_hypers = [('train_learningrate_log10', -6.0, -2.0), ('train_adam_beta1', 0.8, 1.0), ('train_batch_size', 1, 4)] # For ADAM
        cfg_hyprnd1, hyperstr1 = optigan.randomize_hyper(cfg)
        print('randomize_hyper: hyperstr1='+hyperstr1)
        cfg_hyprnd1.print_content()
        cfg_hyprnd2, hyperstr2 = optigan.randomize_hyper(cfg)
        print('randomize_hyper: hyperstr2='+hyperstr2)
        cfg_hyprnd2.print_content()
        self.assertNotEqual(cfg_hyprnd1, cfg_hyprnd2)

        # Train on multiple trials
        cfg.train_max_nbepochs = 5
        cfg.train_nbtrials = 5        # Just run one training only
        optigan.train_multipletrials(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, model.params_trainable, 'tests/test_made__smoke_theano_model_train/smokymodelparams.pkl', cfgtomerge=cfg, cont=False)


        # Go back to single trial for the next tests
        cfg.train_nbtrials = 1
        cfg.train_hypers = []

        # Change a few configuration values
        cfg.train_max_nbepochs = 10
        cfg.newdummyattribute = -1
        delattr(cfg, 'dummyattribute')

        # Try to continue the last training
        optigan.train_multipletrials(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, model.params_trainable, 'tests/test_made__smoke_theano_model_train/smokymodelparams.pkl', cfgtomerge=cfg, cont=True)
        model.saveAllParams('tests/test_made__smoke_theano_model_train/smokymodelparams.pkl')

        model.generate_cmp(cfg.indir, 'tests/test_made__smoke_theano_model_train/smokymodelparams-cmp', fid_lst_val)

        model.generate_wav(cfg.indir, cfg.outdir, fid_lst, 'tests/test_made__smoke_theano_model_train/smokymodelparams-snd', cfg, vocoder, wins=[], do_objmeas=True, do_resynth=True)
        model.generate_wav(cfg.indir, cfg.outdir, fid_lst, 'tests/test_made__smoke_theano_model_train/smokymodelparams-snd-pp_spec_extrapfreq', cfg, vocoder, wins=[], do_objmeas=True, do_resynth=True, pp_spec_extrapfreq=8000)
        model.generate_wav(cfg.indir, cfg.outdir, fid_lst, 'tests/test_made__smoke_theano_model_train/smokymodelparams-snd-pp_spec_pf_coef', cfg, vocoder, wins=[], do_objmeas=True, do_resynth=True, pp_spec_pf_coef=1.2)


        # Test WORLD vocoder
        import vocoders
        vocoder = vocoders.VocoderWORLD(cfg.vocoder_fs, cfg.vocoder_shift, spec_size, _aper_size=nm_size)
        import models_basic
        model = models_basic.ModelFC(lab_size, vocoder, mlpg_wins=[], hiddensize=4, nblayers=2)
        cfg.train_max_nbepochs = 5
        cfg.train_nbtrials = 1        # Just run one training only
        cfg.train_hypers = []
        cfg.cropmode = 'begend'
        cfg.outdir = 'tests/test_made__smoke_compose_compose2_cmp_WORLD/*.cmp:(-1,84)'
        optilse = optimizer.Optimizer(model, errtype='LSE')
        optilse.train_multipletrials(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, model.params_trainable, 'tests/test_made__smoke_theano_model_train_vocoder_WORLD/smokymodelparams.pkl', cfgtomerge=cfg, cont=False)
        model.saveAllParams('tests/test_made__smoke_theano_model_train_vocoder_WORLD/smokymodelparams.pkl')
        model.generate_wav(cfg.indir, cfg.outdir, fid_lst, 'tests/test_made__smoke_theano_model_train_vocoder_WORLD/smokymodelparams-snd', cfg, vocoder, wins=[], do_objmeas=True, do_resynth=True)
        cfg.outdir = cptest+'wav_cmp_lf0_fwlspec65_fwnm17_bndnmnoscale/*.cmp:(-1,83)'



        # Test MLPG
        mlpg_wins = [[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]]
        modelwdeltas = models_basic.ModelFC(lab_size, vocoder, mlpg_wins=mlpg_wins, hiddensize=4, nblayers=2)
        # Use the MLPG features
        cfg.outdir = 'tests/test_made__smoke_compose_compose2_cmp_deltas/*.cmp:(-1,249)'
        optiganwdeltas = optimizer.Optimizer(modelwdeltas, errtype='LSE')
        optiganwdeltas.train_multipletrials(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, modelwdeltas.params_trainable, 'tests/test_made__smoke_theano_model_train/smokymodelparams_wdeltas.pkl', cfgtomerge=cfg, cont=False)
        modelwdeltas.saveAllParams('tests/test_made__smoke_theano_model_train/smokymodelparams_wdeltas.pkl')
        modelwdeltas.generate_wav(cfg.indir, cfg.outdir, fid_lst, 'tests/test_made__smoke_theano_model_train/smokymodelparams_wdeltas-snd', cfg, vocoder, wins=mlpg_wins, do_objmeas=True, do_resynth=True)

        # Restore the non-MLPG features
        cfg.outdir = cptest+'wav_cmp_lf0_fwlspec65_fwnm17_bndnmnoscale/*.cmp:(-1,83)'


        # Now test the various models available

        model = models_basic.ModelBGRU(lab_size, vocoder, mlpg_wins=[], hiddensize=4, nblayers=1)
        modelwdeltas = models_basic.ModelBGRU(lab_size, vocoder, mlpg_wins=mlpg_wins, hiddensize=4, nblayers=1)
        optigan = optimizer.Optimizer(model, errtype='LSE')
        optigan.train_multipletrials(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, model.params_trainable, 'tests/test_made__smoke_theano_model_train/smokymodelparams.pkl', cfgtomerge=cfg, cont=False)
        # model.generate_wav('test/test_made__smoke_theano_model_train/smokymodelparams-snd', fid_lst, cfg, do_objmeas=True, do_resynth=True, indicestosynth=None, spec_comp='fwlspec', spec_size=spec_size, nm_size=nm_size)

        model = models_basic.ModelBLSTM(lab_size, vocoder, mlpg_wins=[], hiddensize=4, nblayers=1)
        modelwdeltas = models_basic.ModelBLSTM(lab_size, vocoder, mlpg_wins=mlpg_wins, hiddensize=4, nblayers=1)
        optigan = optimizer.Optimizer(model, errtype='LSE')
        optigan.train_multipletrials(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, model.params_trainable, 'tests/test_made__smoke_theano_model_train/smokymodelparams.pkl', cfgtomerge=cfg, cont=False)
        # model.generate_wav('test/test_made__smoke_theano_model_train/smokymodelparams-snd', fid_lst, cfg, do_objmeas=True, do_resynth=True, indicestosynth=None, spec_comp='fwlspec', spec_size=spec_size, nm_size=nm_size)

        model = models_basic.ModelGeneric(lab_size, vocoder, mlpg_wins=[], layertypes=['FC', 'BLSTM'], hiddensize=4)
        optigan = optimizer.Optimizer(model, errtype='LSE')
        optigan.train_multipletrials(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, model.params_trainable, 'tests/test_made__smoke_theano_model_train/smokymodelparams.pkl', cfgtomerge=cfg, cont=False)

        import models_cnn
        model = models_cnn.ModelCNN(lab_size, vocoder, hiddensize=4, nbcnnlayers=1, nbfilters=2, spec_freqlen=3, noise_freqlen=3, windur=0.020)
        optigan = optimizer.Optimizer(model, errtype='LSE')
        optigan.train_multipletrials(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, model.params_trainable, 'tests/test_made__smoke_theano_model_train/smokymodelparams.pkl', cfgtomerge=cfg, cont=False)
        # # model.generate_wav('test/test_made__smoke_theano_model_train/smokymodelparams-snd', fid_lst, cfg, do_objmeas=True, do_resynth=True, indicestosynth=None, spec_comp='fwlspec', spec_size=spec_size, nm_size=nm_size)

        optigan = optimizer.Optimizer(model, errtype='WGAN')
        cfg.train_LScoef = 0.0
        optigan.train_multipletrials(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, model.params_trainable, 'tests/test_made__smoke_theano_model_train/smokymodelparams.pkl', cfgtomerge=cfg, cont=False)
        cfg.train_LScoef = 0.25
        optigan.train_multipletrials(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, model.params_trainable, 'tests/test_made__smoke_theano_model_train/smokymodelparams.pkl', cfgtomerge=cfg, cont=False)
        # model.generate_wav('test/test_made__smoke_theano_model_train/smokymodelparams-snd', fid_lst, cfg, do_objmeas=True, do_resynth=True, indicestosynth=None, spec_comp='fwlspec', spec_size=spec_size, nm_size=nm_size)



    def test_backend_theano(self):
        import backend_theano

        import theano.tensor as T

        # Test the following if CUDA is available: (won't be tested on travis anyway since no GPU are available on travis)
        if backend_theano.th_cuda_available():
            print('nvidia_smi_current_gpu={}'.format(nvidia_smi_current_gpu()))  # Can't test it because needs CUDA
            print('nvidia_smi_gpu_memused={}'.format(nvidia_smi_gpu_memused())) # Can't test it because needs CUDA

        x = T.ftensor3('x')

        y = backend_theano.th_print('smoky debug message', x)

        y = backend_theano.nonlin_tanh_saturated(x, coef=1.01)
        y = backend_theano.nonlin_tanh_bysigmoid(x)
        y = backend_theano.nonlin_tanhcm11(x)
        y = backend_theano.nonlin_saturatedsigmoid(x, coef=1.01)
        y = backend_theano.nonlin_softsign(x)
        y = backend_theano.nonlin_sigmoidparm(x, c=0.0, f=1.0)


if __name__ == '__main__':
    unittest.main()
