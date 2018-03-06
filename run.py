'''
This script coordinates the overall pipeline execution:
* Feature extraction
* Data composition/preparation (e.g. output composition, normalisation)
* Training
* Generation
If you want to skip a step, it's very complicate: comment the lines concerned at
the very end of this script.

This file is meant to be widely modified depending on the experiment you run.

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
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/external/')
from utils import *
print_sysinfo()

print_log('Global configurations')
cfg = configuration() # Init configuration structure

# Corpus/Voice(s) options
cp = 'test/slttest/' # The main directory where the data of the voice is stored # TODO Use demo data not test data
cfg.fileids = cp+'/file_id_list.scp'
cfg.id_valid_start = 160
cfg.id_valid_nb = 20
cfg.id_test_nb = 20

# Input text labels
in_size = 601
label_dir = 'binary_label_'+str(in_size)
label_path = cp+label_dir+'/*.lab'
cfg.indir = cp+label_dir+'_norm_minmaxm11/*.lab:(-1,'+str(in_size)+')' # Merlin-minmaxm11 eq.

# Output features
cfg.fs = 32000
f0_min, f0_max = 60, 600
spec_size = 65
nm_size = 17
out_size = 1+spec_size+nm_size
cfg.shift = 0.005
wav_dir = 'wav'
wav_path = cp+wav_dir+'/*.wav'
f0_path = cp+wav_dir+'_lf0/*.lf0'
spec_path = cp+wav_dir+'_fwspec'+str(spec_size)+'/*.fwspec'
nm_path = cp+wav_dir+'_fwnm'+str(nm_size)+'/*.fwnm'
cfg.outdir = cp+wav_dir+'_cmp_lf0_fwspec'+str(spec_size)+'_fwnm'+str(nm_size)+'_bndnmnoscale/*.cmp:(-1,'+str(out_size)+')'
cfg.wdir = cp+wav_dir+'_fwspec'+str(spec_size)+'_weights/*.w:(-1,1)'

# Model options
cfg.model_hiddensize = 512
cfg.model_nbprelayers = 2
cfg.model_nbcnnlayers = 4
cfg.model_nbfilters = 8
cfg.model_spec_freqlen = 13
cfg.model_nm_freqlen = 7
cfg.model_windur = 0.100

# Training options
fparams_fullset = 'model.pkl'
# The ones below will overwrite default options in model.py:train_multipletrials(.)
cfg.train_batchsize = 5
cfg.train_batch_lengthmax = int(3.0/0.005) # Maximum duration [frames] of each batch
cfg.train_nbtrials = 1        # Just run one training only

cfg.print_content()



# Feature extraction -----------------------------------------------------------
def features_extraction():
    import pulsemodel
    with open(cfg.fileids) as f:
        fids = filter(None, [x for x in map(str.strip, f.readlines()) if x])
        for fid in fids:
            print('Extracting features from: '+fid)
            pulsemodel.analysisf(wav_path.replace('*',fid), f0_min=f0_min, f0_max=f0_max, f0_file=f0_path.replace('*',fid), f0_log=True,
            spec_file=spec_path.replace('*',fid), spec_nbfwbnds=spec_size, nm_file=nm_path.replace('*',fid), nm_nbfwbnds=nm_size, verbose=1)


# DNN data composition ---------------------------------------------------------
def composition():
    import compose

    # Compose the inputs
    # The input files are binary labels, as the come from the NORMLAB Process of Merlin TTS pipeline https://github.com/CSTR-Edinburgh/merlin
    compose.compose([label_path+':(-1,'+str(in_size)+')'], cfg.fileids, cfg.indir, id_valid_start=cfg.id_valid_start, normfn=compose.normalise_minmax, do_finalcheck=True, wins=[])

    # Compose the outputs
    compose.compose([f0_path, spec_path+':(-1,'+str(spec_size)+')', nm_path+':(-1,'+str(nm_size)+')'], cfg.fileids, cfg.outdir, id_valid_start=cfg.id_valid_start, normfn=compose.normalise_meanstd_bndnmnoscale)

    # Create time weights (column vector in [0,1]). The frames at begining or end of
    # each file whose weights are smaller than 0.5 will be ignored by the training
    compose.create_weights(spec_path+':(-1,'+str(spec_size)+')', cfg.fileids, cfg.wdir)


# Training ---------------------------------------------------------------------
def training(cont=False):
    print('\nData profile')
    import data
    fid_lst = data.loadids(cfg.fileids)
    in_size = data.getlastdim(cfg.indir)
    out_size = data.getlastdim(cfg.outdir)
    print('    in_size={} out_size={}'.format(in_size,out_size))
    fid_lst_tra = fid_lst[:cfg.id_train_nb()]
    fid_lst_val = fid_lst[cfg.id_valid_start:cfg.id_valid_start+cfg.id_valid_nb]
    print('    {} validation files; ratio of validation data over training data: {:.2f}%'.format(len(fid_lst_val), 100.0*float(len(fid_lst_val))/len(fid_lst_tra)))

    # Build the model
    import models_cnn
    model = models_cnn.ModelCNN(601, spec_size, nm_size, hiddensize=cfg.model_hiddensize, nbprelayers=cfg.model_nbprelayers, nbcnnlayers=cfg.model_nbcnnlayers, nbfilters=cfg.model_nbfilters, spec_freqlen=cfg.model_spec_freqlen, nm_freqlen=cfg.model_nm_freqlen, windur=cfg.model_windur)

    # Here you can load pre-computed weights, or just do nothing and start
    # from fully random weights.

    # Here you can select a subset of the parameters to train, while keeping
    # the other ones frozen.
    params = model.params_trainable # Train all the model's parameters, you can make a selection here

    import optimizer
    optigan = optimizer.Optimizer(model, errtype='WGAN')
    optigan.train_multipletrials(cfg.indir, cfg.outdir, cfg.wdir, fid_lst_tra, fid_lst_val, params, fparams_fullset, cfgtomerge=cfg, cont=cont)

    # Here you can save a subset of parameters to save in a different file
    # import cPickle
    #params = cPickle.load(open(fparams_fullset, 'rb'))[0]
    #print([p[0] for p in params])
    # Save the parameters of the first layer
    #model.saveParams('bottleneck.pkl', params[:2])
    #btl = cPickle.load(open('bottleneck.pkl', 'rb')) # For verification


def generate_wavs():

    # Rebuild the model
    import model_gan
    mod = model_gan.ModelGAN(in_size, spec_size, nm_size)

    demostart = 0
    if hasattr(cfg, 'id_test_demostart'): demostart=cfg.id_test_demostart
    indicestosynth = range(demostart,demostart+10) # Just generate 10 of them
    mod.generate(fparams_fullset, '-demo-snd', cfg, spec_size=spec_size, nm_size=nm_size, do_objmeas=True, do_resynth=True, indicestosynth=indicestosynth)
    mod.generate(fparams_fullset, '-snd', cfg, spec_size=spec_size, nm_size=nm_size, do_objmeas=True, do_resynth=False)

if  __name__ == "__main__" :
    features_extraction()
    composition()
    training(cont='--continue' in sys.argv)
    generate_wavs()
