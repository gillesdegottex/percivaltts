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

def layer_final(l_in, vocoder, mlpg_wins):
    layers_toconcat = []

    if isinstance(vocoder, vocoders.VocoderPML):
        l_out_f0spec = keras.layers.Dense(1+vocoder.spec_size, activation=None, name='lo_f0spec')(l_in)
        l_out_nm = keras.layers.Dense(vocoder.nm_size, activation=keras.activations.sigmoid, name='lo_nm')(l_in)
        # l_out_f0spec = keras.layers.TimeDistributed(keras.layers.Dense(1+vocoder.spec_size, activation=None, name='lo_f0spec'))(l_in) # TODO TODO TODO ???
        # l_out_nm = keras.layers.TimeDistributed(keras.layers.Dense(vocoder.nm_size, activation=keras.activations.sigmoid, name='lo_nm'))(l_in)
        layers_toconcat.extend([l_out_f0spec, l_out_nm])
        if len(mlpg_wins)>0:
            l_out_delta_f0spec = keras.layers.Dense(1+vocoder.spec_size, activation=None, name='lo_delta_f0spec')(l_in)
            l_out_delta_nm = keras.layers.Dense(vocoder.nm_size, activation=keras.activations.tanh, name='lo_delta_nm')(l_in)
            layers_toconcat.extend([l_out_delta_f0spec, l_out_delta_nm])
            if len(mlpg_wins)>1:
                l_out_deltadelta_f0spec = keras.layers.Dense(1+vocoder.spec_size, activation=None, name='lo_deltadelta_f0spec')(l_in)
                l_out_deltadelta_nm = keras.layers.Dense(vocoder.nm_size, activation=keras.activations.tanh, name='lo_deltadelta_nm')(l_in) # TODO TODO TODO keras.activations.tanh/partial(nonlin_tanh_saturated, coef=2.0)
                layers_toconcat.extend([l_out_deltadelta_f0spec, l_out_deltadelta_nm])

    elif isinstance(vocoder, vocoders.VocoderWORLD):
        l_out = keras.layers.Dense(vocoder.featuressize(), name='lo_f0specaper')(l_in)
        if len(mlpg_wins)>0:
            l_out_delta = keras.layers.Dense(vocoder.featuressize(), activation=None, name='lo_delta_f0specaper')(l_in)
            if len(mlpg_wins)>1:
                l_out_deltadelta = keras.layers.Dense(vocoder.featuressize(), activation=None, name='lo_deltadelta_f0specaper')(l_in)

    if len(layers_toconcat)==1: l_out = layers_toconcat[0]
    else:                       l_out = keras.layers.Concatenate(axis=-1, name='lo_concatenation')(layers_toconcat)

    return l_out


class ModelTTS:

    # Network variables
    ctxsize = -1
    vocoder = None

    cfgarch = None
    _kerasmodel = None

    def __init__(self, ctxsize, vocoder, cfgarch=None, kerasmodel=None):
        # Force additional random inputs is using anyform of GAN
        print("Building the model")

        self.ctxsize = ctxsize

        self.vocoder = vocoder
        self.cfgarch = cfgarch

        if not kerasmodel is None:
            self._kerasmodel = kerasmodel
            # self.ctxsize = # TODO
            self._kerasmodel.summary()


    def predict(self, x):
        return self._kerasmodel.predict(x)


    def count_params(self):
        return self._kerasmodel.count_params()

    def saveAllParams(self, fmodel, cfg=None, extras=None, printfn=print, infostr=''):
        if extras is None: extras=dict()
        printfn('    saving parameters in {} ...'.format(fmodel), end='')
        sys.stdout.flush()
        tf.keras.models.save_model(self._kerasmodel, fmodel, include_optimizer=False)
        DATA = [cfg, extras]
        cPickle.dump(DATA, open(fmodel+'.cfgextras.pkl', 'wb'))
        print(' done '+infostr)
        sys.stdout.flush()

    def loadAllParams(self, fmodel, printfn=print, compile=True):
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


    def generate_wav(self, inpath, outpath, fid_lst, syndir, vocoder, wins=[[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]], do_objmeas=True, do_resynth=True
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

        def denormalise(CMP, wins=[[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]]):

            CMP = CMP*np.tile(Ystd, (CMP.shape[0], 1)) + np.tile(Ymean, (CMP.shape[0], 1)) # De-normalise

            if len(wins)>0:
                # Apply MLPG
                from external.merlin.mlpg_fast import MLParameterGenerationFast as MLParameterGeneration
                mlpg_algo = MLParameterGeneration(delta_win=wins[0], acc_win=wins[1])
                var = np.tile(Ystd**2,(CMP.shape[0],1)) # Simplification!
                CMP = mlpg_algo.generation(CMP, var, len(Ymean)/3)
            else:
                CMP = CMP[:,:vocoder.featuressize()]

            return CMP

        if not os.path.isdir(syndir): os.makedirs(syndir)
        if do_resynth and (not os.path.isdir(syndir+'-resynth')): os.makedirs(syndir+'-resynth')

        for vi in xrange(len(X_test)):

            print('Generating {}/{} ...'.format(1+vi, len(X_test)))
            print('    Predict ...')

            if do_resynth:
                CMP = denormalise(y_test[vi], wins=[])
                resyn = vocoder.synthesis(vocoder.fs, CMP, pp_mcep=False)
                sp.wavwrite(syndir+'-resynth/'+fid_lst[vi]+'.wav', resyn, vocoder.fs, norm_abs=True, force_norm_abs=True, verbose=1)

            CMP = self.predict(np.reshape(X_test[vi],[1]+[s for s in X_test[vi].shape]))
            CMP = CMP[0,:,:]

            CMP = denormalise(CMP, wins=wins)
            syn = vocoder.synthesis(vocoder.fs, CMP, pp_mcep=pp_mcep)
            sp.wavwrite(syndir+'/'+fid_lst[vi]+'.wav', syn, vocoder.fs, norm_abs=True, force_norm_abs=True, verbose=1)

            if do_objmeas: vocoder.objmeasures_add(CMP, y_test[vi])

        if do_objmeas: vocoder.objmeasures_stats()

        print_log('Generation finished')
