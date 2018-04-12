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
print_sysinfo()

print_log('Global configurations')
cfg = configuration() # Init configuration structure

# Corpus/Voice(s) options
cp = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tests/slt_arctic_merlin_full/') # The main directory where the data of the voice is stored
cfg.fileids = cp+'/file_id_list.scp'
cfg.id_valid_start = 1032
cfg.id_valid_nb = 50
cfg.id_test_nb = 50

# Input text labels
lab_dir = 'label_state_align'
lab_path = cp+lab_dir+'/*.lab'
lab_questions = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'external/merlin/questions-radio_dnn_416.hed')
in_size = 416+9
labbin_path = cp+lab_dir+'_bin'+str(in_size)+'/*.lab'
cfg.indir = cp+lab_dir+'_bin'+str(in_size)+'_norm_minmaxm11/*.lab:(-1,'+str(in_size)+')' # Merlin-minmaxm11 eq.


# Output features
cfg.fs = 16000
cfg.f0_min, cfg.f0_max = 70, 600
spec_size = 129
nm_size = 33
out_size = 1+spec_size+nm_size
cfg.shift = 0.005
wav_dir = 'wav'
wav_path = cp+wav_dir+'/*.wav'
f0_path = cp+wav_dir+'_lf0/*.lf0'
spec_path = cp+wav_dir+'_fwlspec'+str(spec_size)+'/*.fwlspec'
nm_path = cp+wav_dir+'_fwnm'+str(nm_size)+'/*.fwnm'
cfg.outdir = cp+wav_dir+'_cmp_lf0_fwlspec'+str(spec_size)+'_fwnm'+str(nm_size)+'_nmnoscale/*.cmp:(-1,'+str(out_size)+')'
cfg.wdir = cp+wav_dir+'_fwlspec'+str(spec_size)+'_weights/*.w:(-1,1)'

# Model architecture options
cfg.model_hiddensize = 256      # All arch
cfg.model_nbcnnlayers = 8       # CNN only
cfg.model_nbfilters = 16        # CNN only
cfg.model_spec_freqlen = 5      # [bins] CNN only
cfg.model_nm_freqlen = 5        # [bins] CNN only
cfg.model_windur = 0.025        # [s] 0.025/0.005=5 frames. CNN only

# Training options
cfg.fparams_fullset = 'model.pkl'
# The ones below will overwrite default options in model.py:train_multipletrials(.)
cfg.train_batchsize = 5
cfg.train_batch_lengthmax = int(2.0/0.005) # [frames] Maximum duration of each batch through time
                                           # Has to be short enough to avoid plowing up the GPU's memory and long enough to allow modelling of LT dependences by LSTM layers.
cfg.train_LScoef = 0.25         # For WGAN mixed with LS [def. 0.25]
cfg.train_max_nbepochs = 300    # Can stop much earlier with 3 stacked BLSTM or 6 stacked FC
cfg.train_cancel_nodecepochs = 50

# cfg.train_hypers = [('train_D_learningrate', 0.01, 0.00001), ('train_D_adam_beta1', 0.0, 0.9), ('train_D_adam_beta2', 0.8, 0.9999), ('train_G_learningrate', 0.01, 0.00001), ('train_G_adam_beta1', 0.0, 0.9), ('train_G_adam_beta2', 0.8, 0.9999)]
# cfg.train_nbtrials = 12

cfg.print_content()



# Feature extraction -----------------------------------------------------------
from external import pulsemodel

def pml_analysis(fid):              # pragma: no cover  coverage not detected
    print('Extracting features from: '+fid)
    pulsemodel.analysisf(wav_path.replace('*',fid), f0_min=cfg.f0_min, f0_max=cfg.f0_max, ff0=f0_path.replace('*',fid), f0_log=True, fspec=spec_path.replace('*',fid), spec_nbfwbnds=spec_size, fnm=nm_path.replace('*',fid), nm_nbfwbnds=nm_size, verbose=1)

def features_extraction():
    fids = readids(cfg.fileids)

    # Use this tool for parallel extraction of the acoustic features ...
    from external import pfs
    pfs.map(pml_analysis, fids, processes=7)   # Change number of processes

    # ... or uncomment these line to extract them file by file.
    # for fid in fids:
    #     pulsemodel.analysisf(wav_path.replace('*',fid), f0_min=cfg.f0_min, f0_max=cfg.f0_max, ff0=f0_path.replace('*',fid), f0_log=True,
    #     fspec=spec_path.replace('*',fid), spec_nbfwbnds=spec_size, fnm=nm_path.replace('*',fid), nm_nbfwbnds=nm_size, verbose=1)


def contexts_extraction():
    # Let's use Merlin's code for this

    from external.merlin.label_normalisation import HTSLabelNormalisation
    label_normaliser = HTSLabelNormalisation(question_file_name=lab_questions, add_frame_features=True, subphone_feats='full')

    makedirs(os.path.dirname(labbin_path))
    for fid in readids(cfg.fileids):
        label_normaliser.perform_normalisation([lab_path.replace('*',fid)], [labbin_path.replace('*',fid)])


# DNN data composition ---------------------------------------------------------
def composition_normalisation():
    fids = readids(cfg.fileids)

    import compose

    # Compose the inputs
    # The input files are binary labels, as the come from the NORMLAB Process of Merlin TTS pipeline https://github.com/CSTR-Edinburgh/merlin
    compose.compose([labbin_path+':(-1,'+str(in_size)+')'], fids, cfg.indir, id_valid_start=cfg.id_valid_start, normfn=compose.normalise_minmax, wins=[])

    # Compose the outputs
    compose.compose([f0_path, spec_path+':(-1,'+str(spec_size)+')', nm_path+':(-1,'+str(nm_size)+')'], fids, cfg.outdir, id_valid_start=cfg.id_valid_start, normfn=compose.normalise_meanstd_nmnoscale)

    # Create time weights (column vector in [0,1]). The frames at begining or end of
    # each file whose weights are smaller than 0.5 will be ignored by the training
    compose.create_weights(spec_path+':(-1,'+str(spec_size)+')', fids, cfg.wdir)


def build_model():
    # Build the model
    import models_cnn
    model = models_cnn.ModelCNN(in_size, spec_size, nm_size, hiddensize=cfg.model_hiddensize, nbcnnlayers=cfg.model_nbcnnlayers, nbfilters=cfg.model_nbfilters, spec_freqlen=cfg.model_spec_freqlen, nm_freqlen=cfg.model_nm_freqlen, windur=cfg.model_windur)

    # import models_basic
    # model = models_basic.ModelFC(in_size, 1+spec_size+nm_size, spec_size, nm_size, hiddensize=cfg.model_hiddensize, nblayers=6)
    # model = models_basic.ModelBLSTM(in_size, 1+spec_size+nm_size, spec_size, nm_size, hiddensize=cfg.model_hiddensize, nblayers=3)

    return model

# Training ---------------------------------------------------------------------
def training(cont=False):
    print('\nData profile')
    fid_lst = data.readids(cfg.fileids)
    in_size = data.getlastdim(cfg.indir)
    out_size = data.getlastdim(cfg.outdir)
    print('    in_size={} out_size={}'.format(in_size,out_size))
    fid_lst_tra = fid_lst[:cfg.id_train_nb()]
    fid_lst_val = fid_lst[cfg.id_valid_start:cfg.id_valid_start+cfg.id_valid_nb]
    print('    {} validation files; ratio of validation data over training data: {:.2f}%'.format(len(fid_lst_val), 100.0*float(len(fid_lst_val))/len(fid_lst_tra)))

    model = build_model()

    import optimizer
    optigan = optimizer.Optimizer(model, errtype='WGAN') # 'WGAN' or 'LSE'
    optigan.train_multipletrials(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, model.params_trainable, cfg.fparams_fullset, cfgtomerge=cfg, cont=cont)


def generate(fparams=cfg.fparams_fullset):

    model = build_model()           # Rebuild the model from scratch
    model.loadAllParams(fparams)    # Load the model's parameters

    fid_lst = data.readids(cfg.fileids)

    # Generate the network outputs (without any decomposition), for potential re-use for another network's input
    # model.generate_cmp(cfg.indir, os.path.splitext(fparams)[0]+'-gen/*.cmp', fid_lst)

    fid_lst_test = fid_lst[cfg.id_valid_start+cfg.id_valid_nb:cfg.id_valid_start+cfg.id_valid_nb+cfg.id_test_nb]

    demostart = cfg.id_test_demostart if hasattr(cfg, 'id_test_demostart') else 0
    model.generate_wav(os.path.splitext(fparams)[0]+'-demo-snd', fid_lst_test[demostart:demostart+10], cfg, spec_size=spec_size, nm_size=nm_size, do_objmeas=True, do_resynth=True)

    # And generate all of them for listening tests
    model.generate_wav(os.path.splitext(fparams)[0]+'-snd', fid_lst_test, cfg, spec_size=spec_size, nm_size=nm_size, do_objmeas=True, do_resynth=False)


if  __name__ == "__main__" :                                 # pragma: no cover
    features_extraction()
    contexts_extraction()
    composition_normalisation()
    training(cont='--continue' in sys.argv)
    generate()
