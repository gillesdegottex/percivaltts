# http://pymbook.readthedocs.io/en/latest/testing.html

from percivaltts import *

import unittest

global cfg
cfg = configuration() # Init configuration structure

cptest = 'tests/slt_arctic_merlin_test/' # The main directory where the data of the voice is stored
lab_size = 425
spec_size = 65
nm_size = 17
cfg.vocoder_shift = 0.005   # Time shift between 2 frames
cfg.vocoder_fs = 16000      # Sampling frequency of the samples used for testing

cfg.fileids = cptest+'/file_id_list.scp'
fid_lst = percivaltts.readids(cfg.fileids)
cfg.id_valid_start = 8
cfg.id_valid_nb = 1
cfg.id_test_nb = 1
cfg.indir = cptest+'binary_label_'+str(lab_size)+'_norm_minmaxm11/*.lab:(-1,'+str(lab_size)+')'
cfg.outdir = cptest+'wav_cmp_lf0_fwlspec65_fwnm17_bndnmnoscale/*.cmp:(-1,83)'
cfg.wdir = cptest+'wav_fwlspec65_weights/*.w:(-1,1)'

cfg.arch_hiddenwidth = 4

cfg.train_batch_size = 2
cfg.train_min_nbepochs = 1
cfg.train_cancel_nodecepochs = 2

fid_lst_tra = fid_lst[:cfg.id_train_nb()]
fid_lst_val = fid_lst[cfg.id_valid_start:cfg.id_valid_start+cfg.id_valid_nb]

cfg.dummyattribute = -1

class TestSmokeTensorflowKeras(unittest.TestCase):
    def test_model(self):
        makedirs('tests/test_made__smoke_tfkeras_model')
        makedirs('tests/test_made__smoke_tfkeras_model_train')
        makedirs('tests/test_made__smoke_tfkeras_model_train_vocoder_WORLD')
        makedirs('tests/test_made__smoke_tfkeras_model_train_vocoder_WORLD_mlpg')

        fid_lst = readids(cfg.fileids)
        fid_lst = fid_lst[:3]   ## Just synthesize the first 3

        import percivaltts.vocoders
        vocoder = percivaltts.vocoders.VocoderPML(cfg.vocoder_fs, cfg.vocoder_shift, spec_size, nm_size)

        import percivaltts.modeltts_common
        model = percivaltts.modeltts_common.Generic(lab_size, vocoder, layertypes=['FC', 'FC', 'FC'], cfgarch=cfg)
        print("model.count_params={}".format(model.count_params()))
        self.assertEqual(model.count_params(), 2195)

        # LSE optimizer
        import percivaltts.optimizertts
        cfg.train_max_nbepochs = 5
        cfg.train_nbtrials = 1        # Just run one training only
        cfg.train_hypers = []
        cfg.cropmode = 'begend'
        optigan = percivaltts.optimizertts.OptimizerTTS(cfg,model)

        optigan.train(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams.pkl', cont=False)
        model.save('tests/test_made__smoke_tfkeras_model/smokymodelparams.pkl')

        # Generate waveforms
        model.generate_wav(cfg.indir, cfg.outdir, fid_lst, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams-snd', do_objmeas=True, do_resynth=True)

        # Extra test along side the simple test above
        # Saving ...
        global cfg
        cost_val = 67.43
        model.save('tests/test_made__smoke_tfkeras_model/smokymodelparams.pkl')
        model.save('tests/test_made__smoke_tfkeras_model/smokymodelparams.pkl', cfg=cfg, extras={'cost_val':cost_val})

        # Loading ...
        cfg_loaded, extras_loaded = model.load('tests/test_made__smoke_tfkeras_model/smokymodelparams.pkl')
        self.assertEqual(cfg, cfg_loaded)
        self.assertEqual({'cost_val':cost_val}, extras_loaded)

        # # Save training state TODO Doesn't work anymore for GAN-based training, should use TensorFlow checkpoints instead
        # # import optimizertts
        # optigan.saveTrainingState('tests/test_made__smoke_tfkeras_model/smokytrainingstate.pkl', extras={'cost_val':cost_val})
        #
        # # Load training state
        # cfg_loaded, extras_loaded, rngstate = optigan.loadTrainingState('tests/test_made__smoke_tfkeras_model/smokytrainingstate.pkl')
        # self.assertEqual(cfg, cfg_loaded)
        # self.assertEqual({'cost_val':cost_val}, extras_loaded)

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
        optigan.train(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams.pkl', cont=False)

        # Go back to single trial for the next tests
        cfg.train_nbtrials = 1
        cfg.train_hypers = []

        # Change a few configuration values
        cfg.train_max_nbepochs = 10
        cfg.newdummyattribute = -1
        delattr(cfg, 'dummyattribute')

        # Try to continue the last training
        optigan.train(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams.pkl', cont=True)
        model.save('tests/test_made__smoke_tfkeras_model_train/smokymodelparams.pkl')

        model.generate_cmp(cfg.indir, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams-cmp', fid_lst_val)

        model.generate_wav(cfg.indir, cfg.outdir, fid_lst, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams-snd', do_objmeas=True, do_resynth=True)
        model.generate_wav(cfg.indir, cfg.outdir, fid_lst, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams-snd-pp_spec_extrapfreq', do_objmeas=True, do_resynth=True, pp_spec_extrapfreq=8000)
        model.generate_wav(cfg.indir, cfg.outdir, fid_lst, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams-snd-pp_spec_pf_coef', do_objmeas=True, do_resynth=True, pp_spec_pf_coef=1.2)

        # Test MLPG
        vocoder = percivaltts.vocoders.VocoderPML(cfg.vocoder_fs, cfg.vocoder_shift, spec_size, nm_size, mlpg_wins=[[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]])
        modelwdeltas = percivaltts.modeltts_common.Generic(lab_size, vocoder, layertypes=['FC', 'FC'], cfgarch=cfg)
        # Use the MLPG features
        cfg.outdir = 'tests/test_made__smoke_compose_compose2_cmp_deltas/*.cmp:(-1,249)'
        optiganwdeltas = percivaltts.optimizertts.OptimizerTTS(cfg, modelwdeltas)
        optiganwdeltas.train(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams_wdeltas.pkl', cont=False)
        modelwdeltas.save('tests/test_made__smoke_tfkeras_model_train/smokymodelparams_wdeltas.pkl')
        modelwdeltas.generate_wav(cfg.indir, cfg.outdir, fid_lst, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams_wdeltas-snd', do_objmeas=True, do_resynth=True)
        # Restore the non-MLPG features
        cfg.outdir = cptest+'wav_cmp_lf0_fwlspec65_fwnm17_bndnmnoscale/*.cmp:(-1,83)'

        # Test WORLD vocoder
        vocoder_world = percivaltts.vocoders.VocoderWORLD(cfg.vocoder_fs, cfg.vocoder_shift, spec_size, aper_size=nm_size)
        model = percivaltts.modeltts_common.Generic(lab_size, vocoder_world, layertypes=['FC', 'FC', 'FC'], cfgarch=cfg)
        cfg.train_max_nbepochs = 5
        cfg.train_nbtrials = 1        # Just run one training only
        cfg.train_hypers = []
        cfg.cropmode = 'begend'
        cfg.outdir = 'tests/test_made__smoke_compose_compose2_cmp_WORLD/*.cmp:(-1,84)'
        optilse = percivaltts.optimizertts.OptimizerTTS(cfg, model)
        optilse.train(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, 'tests/test_made__smoke_tfkeras_model_train_vocoder_WORLD/smokymodelparams.pkl', cont=False)
        model.save('tests/test_made__smoke_tfkeras_model_train_vocoder_WORLD/smokymodelparams.pkl')
        model.generate_wav(cfg.indir, cfg.outdir, fid_lst, 'tests/test_made__smoke_tfkeras_model_train_vocoder_WORLD/smokymodelparams-snd', do_objmeas=True, do_resynth=True)
        # Test MLPG
        vocoder_world = percivaltts.vocoders.VocoderWORLD(cfg.vocoder_fs, cfg.vocoder_shift, spec_size, aper_size=nm_size, mlpg_wins=[[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]])
        modelwdeltas = percivaltts.modeltts_common.Generic(lab_size, vocoder_world, layertypes=['FC', 'FC', 'FC'], cfgarch=cfg)
        # Use the MLPG features
        cfg.outdir = 'tests/test_made__smoke_compose_compose2_cmp_WORLD_mlpg/*.cmp:(-1,252)'
        optiganwdeltas = percivaltts.optimizertts.OptimizerTTS(cfg, modelwdeltas)
        optiganwdeltas.train(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, 'tests/test_made__smoke_tfkeras_model_train_vocoder_WORLD_mlpg/smokymodelparams_wdeltas.pkl', cont=False)
        modelwdeltas.save('tests/test_made__smoke_tfkeras_model_train_vocoder_WORLD_mlpg/smokymodelparams_wdeltas.pkl')
        modelwdeltas.generate_wav(cfg.indir, cfg.outdir, fid_lst, 'tests/test_made__smoke_tfkeras_model_train_vocoder_WORLD_mlpg/smokymodelparams_wdeltas-snd', do_objmeas=True, do_resynth=True)

        # Restore PML and non-MLPG features
        cfg.outdir = cptest+'wav_cmp_lf0_fwlspec65_fwnm17_bndnmnoscale/*.cmp:(-1,83)'


        # Now test the various combinations of layers
        vocoder = percivaltts.vocoders.VocoderPML(cfg.vocoder_fs, cfg.vocoder_shift, spec_size, nm_size)

        model = percivaltts.modeltts_common.Generic(lab_size, vocoder, layertypes=['GRU', 'BGRU'], cfgarch=cfg)
        optilse = percivaltts.optimizertts.OptimizerTTS(cfg, model)
        optilse.train(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams.pkl', cont=False)

        model = percivaltts.modeltts_common.Generic(lab_size, vocoder, layertypes=['FC', ['BLSTM', False]], cfgarch=cfg)
        optilse = percivaltts.optimizertts.OptimizerTTS(cfg, model)
        optilse.train(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams.pkl', cont=False)

        # model = percivaltts.modeltts_common.Generic(lab_size, vocoder, layertypes=['BLSTM', 'BLSTM', 'BLSTM'], cfgarch=cfg)
        # optilse = percivaltts.optimizertts.OptimizerTTS(cfg, model)
        # optilse.train(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams.pkl', cont=False)

        # The DCNN model of the paper
        cfg.arch_hiddenwidth = 2
        cfg.arch_ctx_nbcnnlayers = 2
        cfg.arch_ctx_winlen = 3
        cfg.arch_gen_nbcnnlayers = 2
        cfg.arch_gen_nbfilters = 2
        cfg.arch_gen_winlen = 3
        cfg.arch_spec_freqlen = 3
        model = percivaltts.modeltts_common.DCNNF0SpecNoiseFeatures(lab_size, vocoder, cfg)
        optilse = percivaltts.optimizertts.OptimizerTTS(cfg, model)
        optilse.train(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams.pkl', cont=False)
        # model.generate_wav('test/test_made__smoke_tfkeras_model_train/smokymodelparams-snd', fid_lst, cfg, do_objmeas=True, do_resynth=True, indicestosynth=None, spec_comp='fwlspec', spec_size=spec_size, nm_size=nm_size)

        import percivaltts.optimizertts_wgan
        import percivaltts.networks_critic

        optilse = percivaltts.optimizertts_wgan.OptimizerTTSWGAN(cfg, model, errtype='WLSWGAN', critic=percivaltts.networks_critic.Critic(vocoder, lab_size, cfg))
        optilse.train(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams.pkl', cont=False)

        optilse = percivaltts.optimizertts_wgan.OptimizerTTSWGAN(cfg, model, errtype='WGAN', critic=percivaltts.networks_critic.Critic(vocoder, lab_size, cfg))
        optilse.train(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, 'tests/test_made__smoke_tfkeras_model_train/smokymodelparams.pkl', cont=False)


    # def test_backend_tensorflowkeras(self): # TODO TODO TODO
    #     import percivaltts.backend_tensorflow
    #
    #     import theano.tensor as T
    #
    #     # Test the following if CUDA is available: (won't be tested on travis anyway since no GPU are available on travis)
    #     if percivaltts.backend_tensorflow.tf_cuda_available():
    #         print('nvidia_smi_current_gpu={}'.format(nvidia_smi_current_gpu()))  # Can't test it because needs CUDA
    #         print('nvidia_smi_gpu_memused={}'.format(nvidia_smi_gpu_memused())) # Can't test it because needs CUDA
    #
    #     x = T.ftensor3('x')
    #
    #     # y = backend_tensorflow.nonlin_tanh_saturated(x, coef=1.01)
    #     # y = backend_tensorflow.nonlin_tanh_bysigmoid(x)
    #     # y = backend_tensorflow.nonlin_tanhcm11(x)
    #     # y = backend_tensorflow.nonlin_saturatedsigmoid(x, coef=1.01)
    #     y = percivaltts.backend_tensorflow.nonlin_softsign(x)
    #     y = percivaltts.backend_tensorflow.nonlin_sigmoidparm(x, c=0.0, f=1.0)


if __name__ == '__main__':
    unittest.main()
