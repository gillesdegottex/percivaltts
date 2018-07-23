'''
This script coordinates the overall pipeline execution:
* Feature extraction
* Data composition/preparation (e.g. output composition, normalisation)
* Training
* Generation
If you want to skip a step, it's very complicate: comment the lines concerned at
the very end of this script.

This file is meant to be savagely modified depending on the experiment you run.

Copyright(C) 2017 Engineering Department, University of Cambridge, UK.

License
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

Author
    Gilles Degottex <gad27@cam.ac.uk>
'''

print('')

from percivaltts import *  # Always include this first to setup a few things for percival
import data
import vocoders
import compose
import models_cnn
import models_generic
import optimizer
print_sysinfo()

print_log('Global configurations')
cfg = configuration() # Init configuration structure

# Corpus/Voice(s) options
cp = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tests/slt_arctic_merlin_full/') # The main directory where the data of the voice is stored
cfg.fileids = cp+'/file_id_list.scp'
cfg.id_valid_start = 1032
cfg.id_valid_nb = 50
cfg.id_test_nb = 50
fids = readids(cfg.fileids)

# Input text labels
lab_type = 'state'  # 'state' or 'phone'
lab_dir = 'label_'+lab_type+'_align'
lab_path = cp+lab_dir+'/*.lab'
lab_questions = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'external/merlin/questions-radio_dnn_416.hed')
in_size = 416+9
labbin_path = cp+lab_dir+'_bin'+str(in_size)+'/*.lab'
cfg.inpath = os.path.dirname(labbin_path)+'_norm_minmaxm11/*.lab:(-1,'+str(in_size)+')' # Merlin-minmaxm11 eq.
labs_wpath = cp+lab_dir+'_weights/*.w:(-1,1)' # Ignore silences based on labs

# Output features
cfg.vocoder_fs = 16000
cfg.vocoder_shift = 0.005
cfg.vocoder_f0_min, cfg.vocoder_f0_max = 70, 600

vocoder = vocoders.VocoderPML(cfg.vocoder_fs, cfg.vocoder_shift, _spec_size=129, _nm_size=33)
# vocoder = vocoders.VocoderWORLD(cfg.vocoder_fs, cfg.vocoder_shift, _spec_size=129, _aper_size=33)

use_WGAN = True # Switch it to False and it will turn Merling's post-processing on (see below)

do_mlpg = False # Switch it to True and deltas will be added in the input and the MLPG will be used during generation.
pp_mcep = not use_WGAN # Set to True to apply Merlin's post-processing to enhance formants. You need mcep command line from SPTK.

mlpg_wins = []
if do_mlpg: mlpg_wins = [[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]]
out_size = vocoder.featuressize()*(len(mlpg_wins)+1)
wav_dir = 'wav'
wav_path = cp+wav_dir+'/*.wav'
feats_dir = ''
feats_dir+='_'+vocoder.name()
f0_path = cp+wav_dir+feats_dir+'_lf0/*.lf0'
spec_path = cp+wav_dir+feats_dir+'_fwlspec'+str(vocoder.specsize())+'/*.fwlspec'
feats_wpath = cp+wav_dir+feats_dir+'_fwlspec'+str(vocoder.specsize())+'_weights/*.w:(-1,1)' # Ignore silences based on spec energy
if isinstance(vocoder, vocoders.VocoderPML):    noisetag='fwnm'
elif isinstance(vocoder, vocoders.VocoderWORLD):noisetag='fwdbap'   # pragma: no cover
noise_path = cp+wav_dir+feats_dir+'_'+noisetag+str(vocoder.noisesize())+'/*.'+noisetag
vuv_path = cp+wav_dir+feats_dir+'_vuv1/*.vuv'

if do_mlpg: feats_dir+='_mlpg'
cfg.outpath = cp+wav_dir+feats_dir+'_cmp_lf0_fwlspec'+str(vocoder.specsize())+'_'+noisetag+str(vocoder.noisesize())
if isinstance(vocoder, vocoders.VocoderPML):     cfg.outpath+='_nmnoscale'
elif isinstance(vocoder, vocoders.VocoderWORLD): cfg.outpath+='_vuv'    # pragma: no cover
cfg.outpath+='/*.cmp:(-1,'+str(out_size)+')'

# Model architecture options
cfg.model_ctx_nblayers = 1
cfg.model_ctx_nbfilters = 4     # 4 seems enough
cfg.model_ctx_winlen = 21       # Too big (>41,200ms) seems to bring noise in the synth (might lack data)
cfg.model_hiddensize = 256      # For all arch 256
cfg.model_nbcnnlayers = 8       # CNN only 8
cfg.model_nbfilters = 16        # CNN only 16
cfg.model_spec_freqlen = 5      # [bins] CNN only 5
cfg.model_noise_freqlen = 5     # [bins] CNN only 5
cfg.model_windur = 0.025        # [s] 0.025/0.005=5 frames. CNN only 0.025

# Training options
cfg.fparams_fullset = 'model.pkl'
# The ones below will overwrite default options in model.py:train_multipletrials(.)
cfg.train_batch_size = 5
cfg.train_batch_lengthmax = int(2.0/0.005) # [frames] Maximum duration of each batch through time
                                           # Has to be short enough to avoid plowing up the GPU's memory and long enough to allow modelling of LT dependences by LSTM layers.
cfg.wpath = labs_wpath # labs_wpath or feats_wpath. By def. ignore silences according to input labels.
cfg.train_LScoef = 0.25         # LS loss weights 0.25 and WGAN for the rest (even though LS loss is in [0,oo) whereas WGAN loss is on (-oo,+oo))
cfg.train_min_nbepochs = 200
cfg.train_max_nbepochs = 300    # (Can stop much earlier for 3 stacked BLSTM or 6 stacked FC)
cfg.train_cancel_nodecepochs = 50 # (Can reduce it for 3 stacked BLSTM or 6 stacked FC)
if not use_WGAN:
    cfg.train_min_nbepochs = 20
    cfg.train_max_nbepochs = 50
    cfg.train_cancel_nodecepochs = 10

# cfg.train_hypers = [('train_D_learningrate', 0.01, 0.00001), ('train_D_adam_beta1', 0.0, 0.9), ('train_D_adam_beta2', 0.8, 0.9999), ('train_G_learningrate', 0.01, 0.00001), ('train_G_adam_beta1', 0.0, 0.9), ('train_G_adam_beta2', 0.8, 0.9999)]
# cfg.train_nbtrials = 12

cfg.print_content()

# Processes --------------------------------------------------------------------

def pfs_map_vocoder(fid): return vocoder.analysisfid(fid, wav_path, cfg.vocoder_f0_min, cfg.vocoder_f0_max, {'f0':f0_path, 'spec':spec_path, 'noise':noise_path, 'vuv':vuv_path})
def features_extraction():

    # Use this tool for parallel extraction of the acoustic features ...
    from external import pfs
    pfs.map(pfs_map_vocoder, fids)

    # ... or uncomment these line to extract them file by file.
    # for fid in fids: pfs_map_vocoder(fid)

    # Create time weights (column vector in [0,1]). The frames at begining or end of
    # each file whose weights are smaller than 0.5 will be ignored by the training
    compose.create_weights_spec(spec_path+':(-1,'+str(vocoder.specsize())+')', fids, feats_wpath)

    # Compose the outputs
    outpaths = [f0_path, spec_path+':(-1,'+str(vocoder.specsize())+')', noise_path+':(-1,'+str(vocoder.noisesize())+')']
    normfn = compose.normalise_meanstd
    if isinstance(vocoder, vocoders.VocoderPML):        normfn=compose.normalise_meanstd_nmnoscale
    elif isinstance(vocoder, vocoders.VocoderWORLD):    outpaths.append(vuv_path)   # pragma: no cover
    compose.compose(outpaths, fids, cfg.outpath, id_valid_start=cfg.id_valid_start, normfn=normfn, wins=mlpg_wins)


def contexts_extraction():
    # Let's use Merlin's code for this
    from external.merlin.label_normalisation import HTSLabelNormalisation
    label_normaliser = HTSLabelNormalisation(question_file_name=lab_questions, add_frame_features=True, subphone_feats='full' if lab_type else 'coarse_coding') # coarse_coding or full
    makedirs(os.path.dirname(labbin_path))
    for fid in readids(cfg.fileids):
        label_normaliser.perform_normalisation([lab_path.replace('*',fid)], [labbin_path.replace('*',fid)], label_type='state_align' if lab_type else 'phone_align') # phone_align or state_align

    compose.create_weights_lab(lab_path, cfg.fileids, labs_wpath, silencesymbol='sil', shift=cfg.vocoder_shift)

    # Compose the inputs
    # The input files are binary labels, as they come from the NORMLAB Process of Merlin TTS pipeline https://github.com/CSTR-Edinburgh/merlin
    compose.compose([labbin_path+':(-1,'+str(in_size)+')'], fids, cfg.inpath, id_valid_start=cfg.id_valid_start, normfn=compose.normalise_minmax, wins=[], do_finalcheck=False)


def build_model():
    mod = models_cnn.ModelCNN(in_size, vocoder, hiddensize=cfg.model_hiddensize, ctx_nblayers=cfg.model_ctx_nblayers, ctx_nbfilters=cfg.model_ctx_nbfilters, ctx_winlen=cfg.model_ctx_winlen, nbcnnlayers=cfg.model_nbcnnlayers, nbfilters=cfg.model_nbfilters, spec_freqlen=cfg.model_spec_freqlen, noise_freqlen=cfg.model_noise_freqlen, windur=cfg.model_windur)

    # mod = models_generic.ModelGeneric(in_size, vocoder, mlpg_wins=mlpg_wins, layertypes=['FC', 'FC', 'FC', 'FC', 'FC', 'FC'], hiddensize=cfg.model_hiddensize)
    # mod = models_generic.ModelGeneric(in_size, vocoder, mlpg_wins=mlpg_wins, layertypes=['BLSTM', 'BLSTM', 'BLSTM'], hiddensize=cfg.model_hiddensize)
    # mod = models_generic.ModelGeneric(in_size, vocoder, mlpg_wins=mlpg_wins, layertypes=[['CNN',cfg.model_ctx_nbfilters,cfg.model_ctx_winlen], 'FC', 'FC', 'FC', 'FC'], hiddensize=cfg.model_hiddensize)

    return mod


def training(cont=False):
    fid_lst_tra = fids[:cfg.id_train_nb()]
    fid_lst_val = fids[cfg.id_valid_start:cfg.id_valid_start+cfg.id_valid_nb]

    mod = build_model()

    opti = optimizer.Optimizer(mod, errtype='WGAN' if use_WGAN else 'LSE') # 'WGAN' or 'LSE'
    opti.train_multipletrials(cfg.inpath, cfg.outpath, cfg.wpath, fid_lst_tra, fid_lst_val, mod.params_trainable, cfg.fparams_fullset, cfgtomerge=cfg, cont=cont)


def generate(fparams=cfg.fparams_fullset):

    mod = build_model()           # Rebuild the model from scratch
    mod.loadAllParams(fparams)    # Load the model's parameters

    # Generate the network outputs (without any decomposition), for potential re-use for another network's input
    # mod.generate_cmp(cfg.inpath, os.path.splitext(fparams)[0]+'-gen/*.cmp', fids)

    fid_lst_test = fids[cfg.id_valid_start+cfg.id_valid_nb:cfg.id_valid_start+cfg.id_valid_nb+cfg.id_test_nb]

    demostart = cfg.id_test_demostart if hasattr(cfg, 'id_test_demostart') else 0
    mod.generate_wav(cfg.inpath, cfg.outpath, fid_lst_test[demostart:demostart+10], os.path.splitext(fparams)[0]+'-demo-snd', vocoder, wins=mlpg_wins, do_objmeas=True, do_resynth=True, pp_mcep=pp_mcep)

    # And generate all of them for listening tests
    # mod.generate_wav(cfg.inpath, cfg.outpath, fid_lst_test, os.path.splitext(fparams)[0]+'-snd', vocoder, wins=mlpg_wins, do_objmeas=True, do_resynth=True, pp_mcep=pp_mcep)


if  __name__ == "__main__" :                                 # pragma: no cover
    features_extraction()
    contexts_extraction()
    training(cont='--continue' in sys.argv)
    generate()
