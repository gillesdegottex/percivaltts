'''
Definition of the Fully Connected (FC) and Recurrent Neural Networks (RNN) networks.

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
numpy_force_random_seed()
from backend_tensorflow import *

import warnings

from external.pulsemodel import sigproc as sp

import modeltts

from tensorflow import keras

import tensorflow.keras.layers as kl

import networks


class ModelGeneric(modeltts.ModelTTS):
    def __init__(self, ctxsize, vocoder, layertypes=['FC', 'FC', 'FC'], nameprefix=None, cfgarch=None):
        bn_axis=-1
        modeltts.ModelTTS.__init__(self, ctxsize, vocoder)

        if nameprefix is None: nameprefix=''

        l_in = kl.Input(shape=(None, ctxsize), name=nameprefix+'input.conditional')
        l_out = l_in

        for layi in xrange(len(layertypes)):
            layerstr = nameprefix+'l'+str(1+layi)+'_{}'.format(layertypes[layi])

            if layertypes[layi]=='network_context_preproc':
                l_out = networks.network_context_preproc(l_out, cfgarch, use_bn=True)
            elif layertypes[layi]=='network_generator_residual':
                l_out = networks.network_generator_residual(l_out, vocoder, cfgarch, use_bn=True)
            elif layertypes[layi]=='DO':
                l_out = kl.Dropout(0.2, noise_shape=(5, 1, None))(l_out)  # TODO 5 => l_out.input_shape[0] ?
            elif layertypes[layi]=='FC':
                l_out = kl.Dense(cfgarch.arch_hiddensize, use_bias=False, name=layerstr)(l_out) # TODO , kernel_initializer='normal' blows up the values
                l_out = kl.BatchNormalization(axis=bn_axis)(l_out)
                l_out = kl.LeakyReLU(alpha=0.3)(l_out)                    # TODO TODO TODO Using nonlinearity doesn't work! Need to clone it maybe
            elif layertypes[layi]=='GRU' or layertypes[layi]=='BGRU':
                l_gru = kl.GRU(cfgarch.arch_hiddensize, activation='tanh', use_bias=True, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, reset_after=False, name=layerstr)
                if layertypes[layi]=='BGRU':    l_out=kl.Bidirectional(l_gru)(l_out)
                else:                           l_out=l_gru(l_out)
            elif layertypes[layi]=='LSTM' or layertypes[layi]=='BLSTM':
                # l_lstm = kl.LSTM(cfgarch.arch_hiddensize, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, name=layerstr)
                l_lstm = kl.CuDNNLSTM(cfgarch.arch_hiddensize, return_sequences=True)
                if layertypes[layi]=='BLSTM':   l_out=kl.Bidirectional(l_lstm)(l_out)
                else:                           l_out=l_lstm(l_out)
                # TODO TODO TODO Test batch normalisation
                # l_out = kl.LeakyReLU(alpha=0.3)(l_out) # TODO Makes it unstable. tanh works
            elif isinstance(layertypes[layi], list):
                if layertypes[layi][0]=='CNN':
                    nbfilters = layertypes[layi][1]
                    winlen = layertypes[layi][2]
                    l_out = kl.Conv1D(nbfilters, winlen, strides=1, padding='same', dilation_rate=1, activation=None, use_bias=False)(l_out)
                    # TODO TODO TODO Test dilation_rate>1
                    l_out = kl.BatchNormalization(axis=bn_axis)(l_out)
                    l_out = kl.LeakyReLU(alpha=0.3)(l_out)
                elif layertypes[layi][0]=='DilCNN':
                    nbfilters = layertypes[layi][1]
                    winlen = layertypes[layi][2]
                    dilation_rate = layertypes[layi][3]
                    l_out = kl.Conv1D(nbfilters, winlen, strides=1, padding='same', dilation_rate=dilation_rate, activation=None, use_bias=False)(l_out)
                    # TODO TODO TODO Test dilation_rate>1
                    l_out = kl.BatchNormalization(axis=bn_axis)(l_out)
                    l_out = kl.LeakyReLU(alpha=0.3)(l_out)
                # if layertypes[layi][0]=='GCNN':
                elif layertypes[layi][0]=='FC':
                    hiddensize = layertypes[layi][1]
                    l_out = kl.Dense(hiddensize, use_bias=False)(l_out)
                    l_out = kl.BatchNormalization(axis=bn_axis)(l_out)
                    l_out = kl.LeakyReLU(alpha=0.3)(l_out)

        l_out = networks.layer_final(l_out, vocoder, mlpg_wins=[])

        self._kerasmodel = keras.Model(inputs=l_in, outputs=l_out)
        self._kerasmodel.summary()

        # from keras.utils import plot_model
        # plot_model(self._kerasmodel, to_file='model.png')



class ModelCNNF0SpecNoiseFeatures(modeltts.ModelTTS):
    def __init__(self, ctxsize, vocoder, cfgarch, nameprefix=None):
        bn_axis=-1
        modeltts.ModelTTS.__init__(self, ctxsize, vocoder)

        if nameprefix is None: nameprefix=''

        l_in = kl.Input(shape=(None, ctxsize), name=nameprefix+'input.conditional')

        l_ctx = networks.network_context_preproc(l_in, cfgarch, use_bn=True)

        # l_f0 = kl.Lambda(lambda x: x[:,:,:1])(l_ctx)
        # l_spec = kl.Lambda(lambda x: x[:,:,1:1+vocoder.specsize()])(l_ctx)
        # l_nm = kl.Lambda(lambda x: x[:,:,1+vocoder.specsize():1+vocoder.specsize()+vocoder.noisesize()])(l_ctx)

        # F0
        l_f0 = l_ctx
        l_f0 = kl.Conv1D(cfgarch.arch_nbfilters, int(0.5*200/vocoder.shift)*2+1, strides=1, padding='same', dilation_rate=1, use_bias=False)(l_f0)
        l_f0 = kl.BatchNormalization(axis=bn_axis)(l_f0)
        l_f0 = kl.LeakyReLU(alpha=0.3)(l_f0)
        l_f0 = kl.Dense(1, activation=None, use_bias=True)(l_f0)

        # Spec
        l_spec = l_ctx
        l_spec = kl.Dense(cfgarch.arch_hiddensize, use_bias=False)(l_spec)
        l_spec = kl.BatchNormalization(axis=bn_axis)(l_spec)
        l_spec = kl.LeakyReLU(alpha=0.3)(l_spec)
        l_spec = kl.Dense(cfgarch.arch_hiddensize, use_bias=False)(l_spec)
        l_spec = kl.BatchNormalization(axis=bn_axis)(l_spec)
        l_spec = kl.LeakyReLU(alpha=0.3)(l_spec)

        l_spec = kl.Dense(vocoder.specsize(), use_bias=True)(l_spec)   # Projection
        l_spec = kl.Reshape([-1,vocoder.specsize(), 1])(l_spec) # Add the channels after the spectral dimension
        for _ in xrange(cfgarch.arch_nbcnnlayers):
            l_spec = kl.Conv2D(cfgarch.arch_nbfilters, [cfgarch.arch_winlen,cfgarch.arch_spec_freqlen], strides=(1, 1), padding='same', dilation_rate=(1, 1), use_bias=False, data_format='channels_last')(l_spec)
            l_spec = kl.BatchNormalization(axis=bn_axis)(l_spec)
            l_spec = kl.LeakyReLU(alpha=0.3)(l_spec)
        l_spec = kl.Conv2D(1, [cfgarch.arch_winlen,cfgarch.arch_spec_freqlen], strides=(1, 1), padding='same', dilation_rate=(1, 1), use_bias=True, activation=None, data_format='channels_last')(l_spec)
        l_spec = kl.Reshape([-1,l_spec.shape[-2]])(l_spec)

        # NM
        if 0:
            l_nm = kl.Dense(cfgarch.arch_hiddensize, use_bias=False)(l_nm)
            l_nm = kl.BatchNormalization(axis=bn_axis)(l_nm)
            l_nm = kl.LeakyReLU(alpha=0.3)(l_nm)
            l_nm = kl.Dense(vocoder.noisesize(), use_bias=True)(l_ctx)   # Projection
            l_nm = kl.Reshape([-1,vocoder.noisesize(), 1])(l_nm) # Add the channels after the spectral dimension
            for _ in xrange(cfgarch.arch_nbcnnlayers/2):
                l_nm = kl.Conv2D(cfgarch.arch_nbfilters, [cfgarch.arch_winlen,cfgarch.arch_noise_freqlen], strides=(1, 1), padding='same', dilation_rate=(1, 1), use_bias=False, data_format='channels_last')(l_nm)
                l_nm = kl.BatchNormalization(axis=bn_axis)(l_nm)
                l_nm = kl.LeakyReLU(alpha=0.3)(l_nm)
            # l_nm = kl.Dense(vocoder.noisesize(), activation='sigmoid')(l_nm)  # TODO Sigmoid is for PML !
            l_nm = kl.Conv2D(1, [cfgarch.arch_winlen,cfgarch.arch_noise_freqlen], strides=(1, 1), padding='same', dilation_rate=(1, 1), use_bias=True, activation='sigmoid', data_format='channels_last')(l_nm)
            l_nm = kl.Reshape([-1,l_nm.shape[-2]])(l_nm)
        else:
            l_nm = l_ctx
            for _ in xrange(cfgarch.arch_nbcnnlayers/2):
                l_nm = kl.Dense(cfgarch.arch_hiddensize, use_bias=False)(l_nm)   # Projection
                l_nm = kl.BatchNormalization(axis=bn_axis)(l_nm)
                l_nm = kl.LeakyReLU(alpha=0.3)(l_nm)
            l_nm = kl.Dense(vocoder.noisesize(), activation='sigmoid')(l_nm)  # TODO Sigmoid is for PML !

        l_out = kl.Concatenate(axis=-1)([l_f0, l_spec, l_nm])

        self._kerasmodel = keras.Model(inputs=l_in, outputs=l_out)
        self._kerasmodel.summary()
