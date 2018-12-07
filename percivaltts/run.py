'''
This script coordinates the overall pipeline execution:
* Feature extraction
* Context label extraction
* Training
* Generation
If you want to skip a step: comment the corresponding lines at the end of this file.

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

# The following select a GPU available, so that only one GPU is used for one training script running
import os
import external.GPUtil
if len(external.GPUtil.getAvailable())>0: os.environ["CUDA_VISIBLE_DEVICES"]=str(external.GPUtil.getAvailable()[0])
# os.environ["CUDA_VISIBLE_DEVICES"]=""

print('')

from percivaltts import *  # Always include this first to setup a few things for percival
import backend_tensorflow
import modeltts_common
import networks_critic
import optimizertts
import optimizertts_wgan
import data
import vocoders
import compose
import modeltts_common
import networks_critic
import optimizertts
import optimizertts_wgan
print_sysinfo()

from functools import partial

from tensorflow import keras


print_log('Global configurations')
cfg = configuration() # Init configuration structure
# `cfg` is the dirty global variable that is carried around here and there in percival's code.
# This is usually very bad practice as it prevents encapsulating functionalities
# in robust functions and creates extra parameters that do not appear in the functions argument.
# It is however extremely convenient for prototyping and it can be used to store and vary the hyperparameters.

# Corpus/Voice(s) options
cp = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../tests/slt_arctic_merlin_full/') # The main directory where the data of the voice is stored
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
ctxsize = 416+9
labbin_path = cp+lab_dir+'_bin'+str(ctxsize)+'/*.lab'
cfg.inpath = os.path.dirname(labbin_path)+'_norm_minmaxm11/*.lab:(-1,'+str(ctxsize)+')' # Merlin-minmaxm11 eq.
labs_wpath = cp+lab_dir+'_weights/*.w:(-1,1)' # Ignore silences based on labs

# Output features
cfg.vocoder_fs = 16000
cfg.vocoder_shift = 0.005
cfg.vocoder_f0_min, cfg.vocoder_f0_max = 70, 600


# mlpg_wins = [[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]]
mlpg_wins = None
vocoder = vocoders.VocoderPML(cfg.vocoder_fs, cfg.vocoder_shift, spec_size=129, nm_size=33, mlpg_wins=mlpg_wins)
# vocoder = vocoders.VocoderWORLD(cfg.vocoder_fs, cfg.vocoder_shift, spec_size=129, aper_size=33, mlpg_wins=mlpg_wins)

errtype = 'WLSWGAN' # Switch it to 'LSE', 'WGAN', 'WLSWGAN'

out_size = vocoder.featuressize()
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

if not mlpg_wins is None: feats_dir+='_mlpg'
cfg.outpath = cp+wav_dir+feats_dir+'_cmp_lf0_fwlspec'+str(vocoder.specsize())+'_'+noisetag+str(vocoder.noisesize())
if isinstance(vocoder, vocoders.VocoderPML):     cfg.outpath+='_nmnoscale'
elif isinstance(vocoder, vocoders.VocoderWORLD): cfg.outpath+='_vuv'    # pragma: no cover
cfg.outpath+='/*.cmp:(-1,'+str(out_size)+')'

# Model architecture options
cfg.arch_hiddenwidth = 256 # seems enough, more adds noise   Tacotron2:512 (5 chars)
cfg.arch_ctx_nbcnnlayers = 1
cfg.arch_ctx_winlen = 21       # Too big (>41,200ms) seems to bring noise in the synth (might lack data, or just training`)
cfg.arch_gen_nbcnnlayers = 8
cfg.arch_gen_nbfilters = 4
cfg.arch_gen_winlen = 5        # [frames] 5*0.005=0.025s.
cfg.arch_spec_freqlen = 5      # [bins] CNN only 5

# Training options
cfg.fparams_fullset = 'model.h5'
# The ones below will overwrite default options in model.py:train_multipletrials(.)
cfg.train_batch_size = 10 # Faster than 5, and not muffled as with 20 This seems good for avoiding local overfitting and remove whistlings/musical sounds.
cfg.train_batch_lengthmax = int(2.0/0.005) # [frames] Maximum duration of each batch through time
                                           # Has to be short enough to avoid plowing up the GPU's memory and long enough to allow modelling of LT dependences by LSTM layers.
cfg.wpath = labs_wpath # TODO TODO TODO labs_wpath or feats_wpath. By def. ignore silences according to input labels.
cfg.train_wgan_LScoef = 0.25         # LS loss weights 0.25 and WGAN for the rest (even though LS loss is in [0,oo) whereas WGAN loss is on (-oo,+oo)). Increasing it can help reducing spurious artefacts (e.g. musical sounds)
cfg.train_min_nbepochs = 250  # TODO TODO TODO
cfg.train_max_nbepochs = 300  # TODO TODO TODO
cfg.train_cancel_nodecepochs = 25 # (Can reduce it for 3 stacked BLSTM or 6 stacked FC)  # TODO TODO TODO
# cfg.train_wgan_critic_LSweighting = True
cfg.train_wgan_critic_LSWGANtransfreqcutoff = 4000
cfg.train_wgan_critic_LSWGANtranscoef = 1.0/8.0
cfg.train_wgan_critic_use_WGAN_incnoisefeature = False   # TODO TODO TODO


# cfg.train_hypers = [('train_wgan_critic_learningrate_log10', -5.0, -1.0), ('train_wgan_critic_adam_beta1', 0.0, 0.9), ('train_wgan_critic_adam_beta2', 0.8, 0.9999), ('train_wgan_gen_learningrate_log10', -5.0, -1.0), ('train_wgan_gen_adam_beta1', 0.0, 0.9), ('train_wgan_gen_adam_beta2', 0.8, 0.9999), ('train_wgan_gen_adam_beta2', 0.8, 0.9999)]
# cfg.train_nbtrials = 36

cfg.print_content()

# Processes --------------------------------------------------------------------

def pfs_map_vocoder(fid): return vocoder.analysisfid(fid, wav_path, cfg.vocoder_f0_min, cfg.vocoder_f0_max, {'f0':f0_path, 'spec':spec_path, 'noise':noise_path, 'vuv':vuv_path})
def features_extraction():

    # Use this tool for parallel extraction of the acoustic features ...
    from external import pfs
    pfs.map(pfs_map_vocoder, fids)

    # ... or uncomment this line below to extract them file by file.
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
    compose.compose([labbin_path+':(-1,'+str(ctxsize)+')'], fids, cfg.inpath, id_valid_start=cfg.id_valid_start, normfn=compose.normalise_minmax, wins=[], do_finalcheck=False)


def build_model():
    # mod = modeltts_common.Generic(ctxsize, vocoder, layertypes=['FC', 'FC', 'FC', 'FC', 'FC', 'FC'], cfgarch=cfg) # 6 stacked FC
    # mod = modeltts_common.Generic(ctxsize, vocoder, layertypes=['BLSTM', 'BLSTM', 'BLSTM'], cfgarch=cfg)          # 3 stacked BLSTM

    # mod = modeltts_common.Generic(ctxsize, vocoder, layertypes=[['RNDUNI', 100], ['CNN1D',cfg.arch_hiddenwidth,cfg.arch_ctx_winlen], 'FC', 'FC', 'FC', 'FC', 'FC'], cfgarch=cfg)    # CNNFC in the paper
    mod = modeltts_common.DCNNF0SpecNoiseFeatures(ctxsize, vocoder, cfg)    # DCNN in the paper

    return mod


def training(cont=False):
    fid_lst_tra = fids[:cfg.id_train_nb()]
    fid_lst_val = fids[cfg.id_valid_start:cfg.id_valid_start+cfg.id_valid_nb]

    mod = build_model()

    if errtype=='LSE': opti=optimizertts.OptimizerTTS(cfg, mod)
    else:              opti=optimizertts_wgan.OptimizerTTSWGAN(cfg, mod, errtype=errtype, critic=networks_critic.Critic(vocoder, ctxsize, cfg))

    opti.train(cfg.inpath, cfg.outpath, cfg.wpath, fid_lst_tra, fid_lst_val, cfg.fparams_fullset, cont=cont)

    del mod


def generate(fparams=cfg.fparams_fullset):

    mod = build_model()           # Rebuild the model from scratch
    mod.load(fparams)    # Load the model's parameters

    # Generate the network outputs (without any decomposition), for potential re-use for another network's input
    # mod.generate_cmp(cfg.inpath, os.path.splitext(fparams)[0]+'-gen/*.cmp', fids)

    fid_lst_test = fids[cfg.id_valid_start+cfg.id_valid_nb:cfg.id_valid_start+cfg.id_valid_nb+cfg.id_test_nb]

    demostart = cfg.id_test_demostart if hasattr(cfg, 'id_test_demostart') else 0
    mod.generate_wav(cfg.inpath, cfg.outpath, fid_lst_test[demostart:demostart+10], os.path.splitext(fparams)[0]+'-demo-snd', do_objmeas=True, do_resynth=True, pp_mcep=False)
    # mod.generate_wav(cfg.inpath, cfg.outpath, fid_lst_test[demostart:demostart+10], os.path.splitext(fparams)[0]+'-demo-pp-snd', do_objmeas=False, do_resynth=False, pp_mcep=True) # Need SPTK

    # And generate all of them for listening tests
    # mod.generate_wav(cfg.inpath, cfg.outpath, fid_lst_test, os.path.splitext(fparams)[0]+'-snd', do_objmeas=True, do_resynth=True, pp_mcep=False)
    # mod.generate_wav(cfg.inpath, cfg.outpath, fid_lst_test, os.path.splitext(fparams)[0]+'-pp-snd', do_objmeas=True, do_resynth=True, pp_mcep=True)


if  __name__ == "__main__" :                                 # pragma: no cover
    features_extraction()
    contexts_extraction()
    training(cont='--continue' in sys.argv)
    generate()
