'''
The base model class to derive the others from.

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

from __future__ import print_function

from percivaltts import *  # Always include this first to setup a few things

import sys
import os
import cPickle
from functools import partial

import numpy as np
numpy_force_random_seed()

print('\nLoading TensorFlow')
from backend_tensorflow import *
print_sysinfo_backend()

import tensorflow as tf

import data
import vocoders

class ModelTTS:

    # Network variables
    ctxsize = -1
    vocoder = None

    kerasmodel = None

    def __init__(self, ctxsize, vocoder, kerasmodel=None):
        # Force additional random inputs is using anyform of GAN
        print("Building the TTS-dedicated model")

        self.ctxsize = ctxsize

        self.vocoder = vocoder

        if not kerasmodel is None:
            self.kerasmodel = kerasmodel
            # self.ctxsize = # TODO
            self.kerasmodel.summary()


    def predict(self, x):
        return self.kerasmodel.predict(x)


    def count_params(self):
        return self.kerasmodel.count_params()

    def save(self, fmodel, cfg=None, extras=None, printfn=print, infostr=''):
        if extras is None: extras=dict()
        printfn('    saving parameters in {} ...'.format(fmodel), end='')
        sys.stdout.flush()
        tf.keras.models.save_model(self._kerasmodel, fmodel, include_optimizer=False)
        DATA = [cfg, extras]
        cPickle.dump(DATA, open(fmodel+'.cfgextras.pkl', 'wb'))
        print(' done '+infostr)
        sys.stdout.flush()

    def load(self, fmodel, printfn=print, compile=True):
        printfn('    reloading parameters from {} ...'.format(fmodel), end='')
        sys.stdout.flush()
        self._kerasmodel = tf.keras.models.load_model(fmodel, compile=compile)
        DATA = cPickle.load(open(fmodel+'.cfgextras.pkl', 'rb'))
        print(' done')
        sys.stdout.flush()
        return DATA


    def generate_cmp(self, inpath, outpath, fid_lst):

        if not os.path.isdir(os.path.dirname(outpath)): os.mkdir(os.path.dirname(outpath))

        X = data.load(inpath, fid_lst, verbose=1)

        for vi in xrange(len(fid_lst)):
            CMP = self.predict(np.reshape(X[vi],[1]+[s for s in X[vi].shape]))  # Generate them one by one to avoid blowing up the memory
            CMP.astype('float32').tofile(outpath.replace('*',fid_lst[vi]))


    def generate_wav(self, inpath, outpath, fid_lst, syndir, do_objmeas=True, do_resynth=True
            , pp_mcep=False
            , pp_spec_pf_coef=-1 # Common value is 1.2
            , pp_spec_extrapfreq=-1
            ):
        from external.pulsemodel import sigproc as sp

        print('Reloading output stats')
        # Assume mean/std normalisation of the output
        Ymean = np.fromfile(os.path.dirname(outpath)+'/mean4norm.dat', dtype='float32')
        Ystd = np.fromfile(os.path.dirname(outpath)+'/std4norm.dat', dtype='float32')

        print('\nLoading generation data at once ...')
        X_test = data.load(inpath, fid_lst, verbose=1)
        if do_objmeas:
            y_test = data.load(outpath, fid_lst, verbose=1)
            X_test, y_test = data.croplen((X_test, y_test))

        def denormalise(CMP, mlpg_ignore=False):

            CMP = CMP*np.tile(Ystd, (CMP.shape[0], 1)) + np.tile(Ymean, (CMP.shape[0], 1)) # De-normalise

            # TODO Should go in the vocoder, but there is Ystd to put as argument ...
            #      Though, the vocoder is not taking care of the deltas composition during data composition either.
            if (not self.vocoder.mlpg_wins is None) and len(self.vocoder.mlpg_wins)>0:    # If MLPG is used
                if mlpg_ignore:
                    CMP = CMP[:,:self.vocoder.featuressizeraw()]
                else:
                    # Apply MLPG
                    from external.merlin.mlpg_fast import MLParameterGenerationFast as MLParameterGeneration
                    mlpg_algo = MLParameterGeneration(delta_win=self.vocoder.mlpg_wins[0], acc_win=self.vocoder.mlpg_wins[1])
                    var = np.tile(Ystd**2,(CMP.shape[0],1)) # Simplification!
                    CMP = mlpg_algo.generation(CMP, var, self.vocoder.featuressizeraw())

            return CMP

        if not os.path.isdir(syndir): os.makedirs(syndir)
        if do_resynth and (not os.path.isdir(syndir+'-resynth')): os.makedirs(syndir+'-resynth')

        for vi in xrange(len(X_test)):

            print('Generating {}/{} fid={} ...'.format(1+vi, len(X_test), fid_lst[vi]))
            print('    Predict ...')

            if do_resynth:
                CMP = denormalise(y_test[vi], mlpg_ignore=True)
                resyn = self.vocoder.synthesis(self.vocoder.fs, CMP, pp_mcep=False)
                sp.wavwrite(syndir+'-resynth/'+fid_lst[vi]+'.wav', resyn, self.vocoder.fs, norm_abs=True, force_norm_abs=True, verbose=1)

            CMP = self.predict(np.reshape(X_test[vi],[1]+[s for s in X_test[vi].shape]))
            CMP = CMP[0,:,:]

            CMP = denormalise(CMP)
            syn = self.vocoder.synthesis(self.vocoder.fs, CMP, pp_mcep=pp_mcep)
            sp.wavwrite(syndir+'/'+fid_lst[vi]+'.wav', syn, self.vocoder.fs, norm_abs=True, force_norm_abs=True, verbose=1)

            if do_objmeas: self.vocoder.objmeasures_add(CMP, y_test[vi])

        if do_objmeas: self.vocoder.objmeasures_stats()

        print_log('Generation finished')
