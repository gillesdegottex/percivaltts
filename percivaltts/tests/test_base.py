# http://pymbook.readthedocs.io/en/latest/testing.html

import os
from percivaltts import *

import unittest

print_sysinfo()

print_log('Global configurations')
cfg = configuration() # Init configuration structure

# Corpus/Voice(s) options
cp = 'tests/slt_arctic_merlin_test/' # The main directory where the data of the voice is stored
cfg.fileids = cp+'/file_id_list.scp'
cfg.id_valid_start = 8
cfg.id_valid_nb = 1
cfg.id_test_nb = 1

# Input text labels
label_state_align_path = cp+'label_state_align/*.lab'
in_size = 425 # 601
label_dir = 'binary_label_'+str(in_size)
label_path = cp+label_dir+'/*.lab'
cfg.indir = cp+label_dir+'_norm_minmaxm11/*.lab:(-1,'+str(in_size)+')' # Merlin-minmaxm11 eq.

# Output features
cfg.vocoder_fs = 32000
cfg.vocoder_shift = 0.005
f0_min, f0_max = 60, 600
spec_size = 65
nm_size = 17
out_size = 1+spec_size+nm_size
wav_dir = 'wav'
wav_path = cp+wav_dir+'/*.wav'
f0_path = cp+wav_dir+'_lf0/*.lf0'
spec_fw_path = cp+wav_dir+'_fwlspec'+str(spec_size)+'/*.fwlspec'
spec_fwcep_path = cp+wav_dir+'_fwcep'+str(spec_size)+'/*.mcep'
nm_path = cp+wav_dir+'_fwnm'+str(nm_size)+'/*.fwnm'
cfg.outdir = cp+wav_dir+'_cmp_lf0_fwlspec'+str(spec_size)+'_fwnm'+str(nm_size)+'_bndnmnoscale/*.cmp:(-1,'+str(out_size)+')'
cfg.wdir = cp+wav_dir+'_fwlspec'+str(spec_size)+'_weights/*.w:(-1,1)'

cfg.print_content()

class TestBase(unittest.TestCase):

    def test_contexts_features_extractions_and_composition(self):

        from external.merlin.label_normalisation import HTSLabelNormalisation
        label_normaliser = HTSLabelNormalisation(question_file_name='external/merlin/questions-radio_dnn_416.hed', add_frame_features=True, subphone_feats='full')

        fids = readids(cfg.fileids)

        makedirs(os.path.dirname(label_path))
        for fid in fids:
            label_normaliser.perform_normalisation([label_state_align_path.replace('*',fid)], [label_path.replace('*',fid)])

        import vocoders
        vocoder_pml = vocoders.VocoderPML(cfg.vocoder_fs, cfg.vocoder_shift, spec_size, nm_size)
        vocoder_world = vocoders.VocoderWorld(cfg.vocoder_fs, cfg.vocoder_shift, spec_size, _aper_size=nm_size)

        for fid in fids:
            print('Extracting features from: '+fid)
            vocoder_pml.analysisfid(fid, wav_path, cfg.vocoder_f0_min, cfg.vocoder_f0_max, {'f0':f0_path, 'spec':spec_fw_path, 'noise':nm_path)
            vocoder_world.analysisfid(fid, wav_path, cfg.vocoder_f0_min, cfg.vocoder_f0_max, {'f0':cp+wav_dir+'_world_lf0/*.lf0', 'spec':cp+wav_dir+'_world_fwlspec/*.fwlspec', 'noise':cp+wav_dir+'_world_fwdbaper/*.fwdbaper', 'vuv':cp+wav_dir+'_world_vuv/*.vuv'})
            # pulsemodel.analysisf(wav_path.replace('*',fid), f0_min=f0_min, f0_max=f0_max, ff0=f0_path.replace('*',fid), f0_log=True,
            # fspec=spec_fw_path.replace('*',fid), spec_nbfwbnds=spec_size, fnm=nm_path.replace('*',fid), nm_nbfwbnds=nm_size, verbose=1)


        import compose

        # Compose the inputs
        # The input files are binary labels, as the come from the NORMLAB Process of Merlin TTS pipeline https://github.com/CSTR-Edinburgh/merlin
        compose.compose([label_path+':(-1,'+str(in_size)+')'], fids, cfg.indir, id_valid_start=cfg.id_valid_start, normfn=compose.normalise_minmax, do_finalcheck=True, wins=[])

        # Compose the outputs
        compose.compose([f0_path, spec_fw_path+':(-1,'+str(spec_size)+')', nm_path+':(-1,'+str(nm_size)+')'], fids, cfg.outdir, id_valid_start=cfg.id_valid_start, normfn=compose.normalise_meanstd_nmnoscale)

        # Create time weights (column vector in [0,1]). The frames at begining or end of
        # each file whose weights are smaller than 0.5 will be ignored by the training
        compose.create_weights_spec(spec_fw_path+':(-1,'+str(spec_size)+')', fids, cfg.wdir, spec_type='mcep')    # Wrong data, just to smoke it
        compose.create_weights_spec(spec_fw_path+':(-1,'+str(spec_size)+')', fids, cfg.wdir, spec_type='fwcep')   # Wrong data, just to smoke it
        compose.create_weights_spec(spec_fw_path+':(-1,'+str(spec_size)+')', fids, cfg.wdir, spec_type='fwlspec')  # Overwrite with the good one

        import data
        fid_lst_val = fids[cfg.id_valid_start:cfg.id_valid_start+cfg.id_valid_nb]
        X_vals = data.load(cfg.indir, fid_lst_val, verbose=1, label='Context labels: ')
        worst_val = data.cost_0pred_rmse(X_vals) # RMSE
        print("    X 0-pred validation RMSE = {} (100%)".format(worst_val))

        Y_vals = data.load(cfg.outdir, fid_lst_val, verbose=1, label='Output features: ')
        # X_vals, Y_vals = data.croplen([X_vals, Y_vals])
        worst_val = data.cost_0pred_rmse(Y_vals) # RMSE
        print("    Y 0-pred validation RMSE = {} (100%)".format(worst_val))

    # def test_vocoder_pulsemodel_features_extraction_and_composition_fwcep(self):
    #     from external import pulsemodel
    #     with open(cfg.fileids) as f:
    #         fids = filter(None, [x for x in map(str.strip, f.readlines()) if x])
    #         for fid in fids:
    #             print('Extracting features from: '+fid)
    #             pulsemodel.analysisf(wav_path.replace('*',fid), f0_min=f0_min, f0_max=f0_max, ff0=f0_path.replace('*',fid), f0_log=True,
    #             fspec=spec_fwcep_path.replace('*',fid), spec_fwceporder=spec_size, fnm=nm_path.replace('*',fid), nm_nbfwbnds=nm_size, verbose=1)
    #
    #     import compose
    #
    #     # Compose the outputs for fwcep
    #     compose.compose([f0_path, spec_fwcep_path+':(-1,'+str(spec_size)+')', nm_path+':(-1,'+str(nm_size)+')'], cfg.fileids, cp+wav_dir+'_cmp_lf0_fwcep'+str(spec_size)+'_fwnm'+str(nm_size)+'_bndnmnoscale/*.cmp:(-1,'+str(out_size)+')', id_valid_start=cfg.id_valid_start, normfn=compose.normalise_meanstd_nmnoscale)
    #
    #     # Create time weights (column vector in [0,1]). The frames at begining or end of
    #     # each file whose weights are smaller than 0.5 will be ignored by the training
    #     compose.create_weights(spec_fwcep_path+':(-1,'+str(spec_size)+')', cfg.fileids, cfg.wdir, spec_type='fwcep')

if __name__ == '__main__':
    unittest.main()
