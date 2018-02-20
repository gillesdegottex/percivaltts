'''
This script coordinates the pipeline execution flow.
It is the only one that knows all of the file paths.
If don't need to run some part of it, juste comment the lines you want to skip.
'''

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))+'/external/')

import runpy

cp = '/home/daffy/CUED_local/db/SLT32/' # The main directory where the data of the voice is stored
f0_min, f0_max = 60, 600
cfg = runpy.run_path(cp+'/info.py') # this contains various variables that are dependent on the data content (e.g. id_valid_start)

# Feature extraction -----------------------------------------------------------
wavdir = 'testwav'
specdim = 129
nmdim = 33
wavpath = cp+wavdir+'/*.wav'
f0path = cp+wavdir+'_lf0/*.lf0'
specpath = cp+wavdir+'_fwspec'+str(specdim)+'/*.fwspec'
nmpath = cp+wavdir+'_fwnm'+str(nmdim)+'/*.fwnm'
import pulsemodel

# with open(cp+'file_id_list.scp') as f:
#     fids = filter(None, [x for x in map(str.strip, f.readlines()) if x])
#     for fid in fids:
#         print('Extracting features from: '+fid)
#         pulsemodel.analysisf(wavpath.replace('*',fid), f0_min=f0_min, f0_max=f0_max, f0_file=f0path.replace('*',fid), f0_log=True,
#         spec_file=specpath.replace('*',fid), spec_nbfwbnds=specdim, nm_file=nmpath.replace('*',fid), nm_nbfwbnds=nmdim, verbose=1)


# DNN data Preparation ---------------------------------------------------------
import compose

# Compose the inputs
# The input files are binary labels, as the come from the NORMLAB Process of Merlin TTS pipeline https://github.com/CSTR-Edinburgh/merlin
compose.compose([cp+'binary_label_601/*.lab:(-1,601)'], cp+'file_id_list.scp', cp+'binary_label_601_norm_minmaxm11/*.lab', id_valid_start=cfg['id_valid_start'], normfn=compose.normalise_minmax, do_finalcheck=True, wins=[])

# Compose the outputs
compose.compose([f0path, specpath+':(-1,'+str(specdim)+')', nmpath+':(-1,'+str(nmdim)+')'], cp+'file_id_list.scp', cp+wavdir+'_cmp_lf0_fwspec'+str(specdim)+'_fwnm'+str(nmdim)+'_bndnmnoscale/*.cmp', id_valid_start=cfg['id_valid_start'], normfn=compose.normalise_meanstd_bndnmnoscale)

# Create time weights (column vector in [0,1]). The frames at begining or end of
# each file whose weights are smaller than 0.5 will be ignored by the training
compose.create_weights(specpath+':(-1,'+str(specdim)+')', cp+'file_id_list.scp', cp+wavdir+'_fwspec'+str(specdim)+'_weights/*.w')


# Training ---------------------------------------------------------------------

# TODO
