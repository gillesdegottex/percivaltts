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

import numpy as np
numpy_force_random_seed()

print('\nLoading Theano')
from backend_theano import *
print_sysinfo_theano()
import theano
import theano.tensor as T
import lasagne
# lasagne.random.set_rng(np.random)

import data

class Model:

    # lasagne.nonlinearities.rectify, lasagne.nonlinearities.leaky_rectify, lasagne.nonlinearities.very_leaky_rectify, lasagne.nonlinearities.elu, lasagne.nonlinearities.softplus, lasagne.nonlinearities.tanh, networks.nonlin_softsign
    _nonlinearity = lasagne.nonlinearities.very_leaky_rectify

    # Network and Prediction variables

    insize = -1
    _input_values = None # Input contextual values (e.g. text, labels)
    inputs = None   # All the inputs of prediction function

    params_all = None # All possible parameters, including non trainable, running averages of batch normalisation, etc.
    params_trainable = None # Trainable parameters
    updates = []

    _hiddensize = 256

    vocoder = None
    net_out = None  # Network output
    outputs = None  # Outputs of prediction function

    predict = None  # Prection function

    def __init__(self, insize, _vocoder, hiddensize=256):
        # Force additional random inputs is using anyform of GAN
        print("Building the model")

        self.insize = insize

        self.vocoder = _vocoder
        self._hiddensize = hiddensize

        self._input_values = T.ftensor3('input_values')


    def init_finish(self, net_out):

        self.net_out = net_out

        print('    architecture:')
        for li, l in enumerate(lasagne.layers.get_all_layers(self.net_out)):
            print('        {}: {}({})'.format(li, l.name, l.output_shape))

        self.params_all = lasagne.layers.get_all_params(self.net_out)
        self.params_trainable = lasagne.layers.get_all_params(self.net_out, trainable=True)

        print('    params={}'.format(self.params_all))

        print('    compiling prediction function ...')
        self.inputs = [self._input_values]

        predicted_values = lasagne.layers.get_output(self.net_out, deterministic=True)
        self.outputs = predicted_values
        self.predict = theano.function(self.inputs, self.outputs, updates=self.updates)

        print('')


    def nbParams(self):
        return paramss_count(self.params_all)

    def saveAllParams(self, fmodel, cfg=None, extras=None, printfn=print, infostr=''):
        if extras is None: extras=dict()
        # https://github.com/Lasagne/Lasagne/issues/159
        printfn('    saving parameters in {} ...'.format(fmodel), end='')
        sys.stdout.flush()
        paramsvalues = [(str(p), p.get_value()) for p in self.params_all]
        DATA = [paramsvalues, cfg, extras]
        cPickle.dump(DATA, open(fmodel, 'wb'))
        print(' done '+infostr)
        sys.stdout.flush()

    def loadAllParams(self, fmodel, printfn=print):
        # https://github.com/Lasagne/Lasagne/issues/159
        printfn('    reloading parameters from {} ...'.format(fmodel), end='')
        sys.stdout.flush()
        DATA = cPickle.load(open(fmodel, 'rb'))
        for p, v in zip(self.params_all, DATA[0]): p.set_value(v[1])
        print(' done')
        sys.stdout.flush()
        return DATA[1:]


    def generate_cmp(self, inpath, outpath, fid_lst):

        if not os.path.isdir(os.path.dirname(outpath)): os.mkdir(os.path.dirname(outpath))

        X = data.load(inpath, fid_lst, verbose=1)

        for vi in xrange(len(fid_lst)):
            CMP = self.predict(np.reshape(X[vi],[1]+[s for s in X[vi].shape]))  # Generate them one by one to avoid blowing up the memory
            CMP.astype('float32').tofile(outpath.replace('*',fid_lst[vi]))


    def generate_wav(self, inpath, outpath, fid_lst, syndir, cfg, vocoder, wins=[[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]], do_objmeas=True, do_resynth=True
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

        for vi in xrange(len(X_test)):

            print('Generating {}/{} ...'.format(1+vi, len(X_test)))
            print('    Predict ...')

            if do_resynth:
                CMP = denormalise(y_test[vi], wins=[])
                resyn = vocoder.synthesis(cfg.vocoder_fs, CMP, pp_mcep=False)
                sp.wavwrite(syndir+'/'+fid_lst[vi]+'-resynth.wav', resyn, cfg.vocoder_fs, norm_abs=True, force_norm_abs=True, verbose=1)

            CMP = self.predict(np.reshape(X_test[vi],[1]+[s for s in X_test[vi].shape]))
            CMP = CMP[0,:,:]

            CMP = denormalise(CMP, wins=wins)
            syn = vocoder.synthesis(cfg.vocoder_fs, CMP, pp_mcep=pp_mcep)
            sp.wavwrite(syndir+'/'+fid_lst[vi]+'.wav', syn, cfg.vocoder_fs, norm_abs=True, force_norm_abs=True, verbose=1)

            if do_objmeas: vocoder.objmeasures_add(CMP, y_test[vi])

        if do_objmeas: vocoder.objmeasures_stats()

        print_log('Generation finished')
