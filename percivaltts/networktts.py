'''
Functions returning some outpout(s) given input(s), without extra objects created.
It is meant to be dedicated to TTS.

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

import numpy as np
from functools import partial

from tensorflow import keras
import tensorflow.keras.layers as kl
from tensorflow.keras import backend as K
from keras.layers import Activation

from backend_tensorflow import nonlin_tanh_saturated, NonLin_Tanh_Saturated

import vocoders

class GaussianNoiseInput(kl.Layer):

    def __init__(self, stddev=1.0, width=100, **kwargs):
        super(GaussianNoiseInput, self).__init__(**kwargs)
        self.stddev = stddev
        self.width = width

    def call(self, inputs, training=None):
        noiseshape = K.shape(inputs)
        noiseshape = (noiseshape[0], noiseshape[1], self.width)
        noise = K.random_normal(shape=noiseshape, mean=0., stddev=self.stddev)
        return K.concatenate((inputs, noise), axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[:-1], input_shape[-1]+self.width)

    def get_config(self):
        config = super(GaussianNoiseInput,self).get_config()
        config['stddev'] = self.stddev
        config['width'] = self.width
        return config


def pFC(input, width, bn=True, **kwargs):
    output = keras.layers.Dense(width, use_bias=not bn, **kwargs)(input)
    if bn: output=keras.layers.BatchNormalization(axis=-1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    return output

def pDO(input, rate=0.2, batch_size=5):
    # noise_shape = K.shape(input)
    # noise_shape = (noise_shape[0], 1, noise_shape[2]) # TODO Just make save/load to crash
    noise_shape = (batch_size, 1, None)
    output = kl.Dropout(rate=rate, noise_shape=noise_shape)(input)
    return output

def pLSTM(input, width, bn=False, cudnn=True, **kwargs):
    if bn: print('WARNING: Batch normalisation can be unstable with LSTM layers')
    # TODO Test batch normalisation: Does not always work
    if cudnn:
        output = kl.CuDNNLSTM(width, return_sequences=True, **kwargs)(input)
    else:
        output = kl.LSTM(width, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, **kwargs)(input)   # For compatibility with CuDNNLSTM
    # l_out = kl.LeakyReLU(alpha=0.3)(l_out) # TODO Makes it unstable. tanh works though
    return output

def pRawLSTM(input, width, bn=False, **kwargs):
    return pLSTM(input, width, bn=bn, cudnn=False, **kwargs)

def pBLSTM(input, width, bn=False, cudnn=True, **kwargs):
    if bn: print('WARNING: Batch normalisation can be unstable with BLSTM layers')
    # TODO Test batch normalisation: Does not always work
    if cudnn:
        l_lstm = kl.CuDNNLSTM(width, return_sequences=True, **kwargs)
    else:
        l_lstm = kl.LSTM(width, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, **kwargs)   # For compatibility with CuDNNLSTM
    output = kl.Bidirectional(l_lstm)(input)
    # l_out = kl.LeakyReLU(alpha=0.3)(l_out) # TODO Makes it unstable. tanh works though
    return output

def pRawBLSTM(input, width, bn=False, **kwargs):
    return pBLSTM(input, width, bn=bn, cudnn=False, **kwargs)

def pGRU(input, width, bn=False, **kwargs):
    if bn: print('WARNING: Batch normalisation is not working for GRU layers (bug?)')
    # TODO Test batch normalisation: Not able to make it work
    output = kl.GRU(width, activation='tanh', use_bias=True, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, reset_after=False)(input)
    # l_out = kl.LeakyReLU(alpha=0.3)(l_out) # TODO Makes it unstable. tanh works though
    return output

def pBGRU(input, width, bn=False, **kwargs):
    if bn: print('WARNING: Batch normalisation is not working for BGRU layers (bug?)')
    # TODO Test batch normalisation: Not able to make it work
    l_gru = kl.GRU(width, activation='tanh', use_bias=True, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, reset_after=False)
    output = kl.Bidirectional(l_gru)(input)
    # l_out = kl.LeakyReLU(alpha=0.3)(l_out) # TODO Makes it unstable. tanh works though
    return output

def pCNN1D(input, nbfilters, winlen, bn=True, **kwargs):
    output = kl.Conv1D(nbfilters, winlen, strides=1, padding='same', dilation_rate=1, activation=None, use_bias=not bn, **kwargs)(input)
    if bn: output=kl.BatchNormalization(axis=-1)(output)
    output = kl.LeakyReLU(alpha=0.3)(output)
    return output

def pDilCNN1D(input, nbfilters, winlen, dil, bn=True, **kwargs):
    output = kl.Conv1D(nbfilters, winlen, strides=1, padding='same', dilation_rate=dil, activation=None, use_bias=not bn, **kwargs)(input)
    if bn: output=kl.BatchNormalization(axis=-1)(output)
    output = kl.LeakyReLU(alpha=0.3)(output)
    return output


def network_generic(input, layertypes=['FC', 'FC', 'FC'], bn=True, cfgarch=None):
    bn_axis=-1

    l_out = input
    for layi in xrange(len(layertypes)):

        if layertypes[layi]=='FC':
            l_out = pFC(l_out, width=cfgarch.arch_hiddenwidth, bn=bn)
        elif layertypes[layi]=='DO':
            l_out = pDO(l_out, 0.2, batch_size=cfgarch.train_batch_size)
        elif layertypes[layi]=='LSTM':
            l_out = pLSTM(l_out, width=cfgarch.arch_hiddenwidth, bn=bn)
        elif layertypes[layi]=='RawLSTM':
            l_out = pRawLSTM(l_out, width=cfgarch.arch_hiddenwidth, bn=bn)
        elif layertypes[layi]=='BLSTM':
            l_out = pBLSTM(l_out, width=cfgarch.arch_hiddenwidth, bn=bn)
        elif layertypes[layi]=='RawBLSTM':
            l_out = pRawBLSTM(l_out, width=cfgarch.arch_hiddenwidth, bn=bn)
        elif layertypes[layi]=='GRU':
            l_out = pGRU(l_out, width=cfgarch.arch_hiddenwidth, bn=bn)
        elif layertypes[layi]=='BGRU':
            l_out = pBGRU(l_out, width=cfgarch.arch_hiddenwidth, bn=bn)
        elif layertypes[layi]=='RND':
            l_out = GaussianNoiseInput(width=cfgarch.arch_hiddenwidth)(l_out)

        elif isinstance(layertypes[layi], list):
            if layertypes[layi][0]=='FC':
                l_out = pFC(l_out, layertypes[layi][1], bn=bn)
            elif layertypes[layi][0]=='CNN1D':
                l_out = pCNN1D(l_out, layertypes[layi][1], layertypes[layi][2], bn=bn)
            elif layertypes[layi][0]=='DilCNN1D':
                l_out = pDilCNN1D(l_out, layertypes[layi][1], layertypes[layi][2], layertypes[layi][3], bn=bn)
            elif layertypes[layi][0]=='RND':
                l_out = GaussianNoiseInput(width=layertypes[layi][1])(l_out)
            else:
                raise ValueError('Unknown layer type '+str(layertypes[layi]))

        elif callable(layertypes[layi]):
            l_out = layertypes[layi](l_out)

        else:
            raise ValueError('Unknown layer type '+str(layertypes[layi]))

    return l_out


# def network_context_preproc(input, winlen, cfgarch, bn=True):
#
#     output = input
#     for _ in xrange(cfgarch.arch_ctx_nbcnnlayers):
#         output = pCNN1D(output, cfgarch.arch_hiddenwidth, cfgarch.arch_ctx_winlen, bn=bn)
#
#     output = pBLSTM(output, cfgarch.arch_hiddenwidth, bn=bn)
#
#     return output

def network_final(l_in, vocoder, mlpg_wins=None):
    layers_toconcat = []

    if isinstance(vocoder, vocoders.VocoderPML):
        l_out_f0spec = keras.layers.Dense(1+vocoder.spec_size, activation=None, name='lo_f0spec')(l_in)
        l_out_nm = keras.layers.Dense(vocoder.nm_size, activation=keras.activations.sigmoid, name='lo_nm')(l_in)
        # l_out_f0spec = keras.layers.TimeDistributed(keras.layers.Dense(1+vocoder.spec_size, activation=None, name='lo_f0spec'))(l_in) # TODO ???
        # l_out_nm = keras.layers.TimeDistributed(keras.layers.Dense(vocoder.nm_size, activation=keras.activations.sigmoid, name='lo_nm'))(l_in)
        layers_toconcat.extend([l_out_f0spec, l_out_nm])
        if (not mlpg_wins is None) and len(mlpg_wins)>0:
            l_out_delta_f0spec = keras.layers.Dense(1+vocoder.spec_size, activation=None, name='lo_delta_f0spec')(l_in)
            l_out_delta_nm = keras.layers.Dense(vocoder.nm_size, activation=keras.activations.tanh, name='lo_delta_nm')(l_in)
            layers_toconcat.extend([l_out_delta_f0spec, l_out_delta_nm])
            if (not mlpg_wins is None) and len(mlpg_wins)>1:
                l_out_deltadelta_f0spec = keras.layers.Dense(1+vocoder.spec_size, activation=None, name='lo_deltadelta_f0spec')(l_in)
                # l_out_deltadelta_nm = keras.layers.Dense(vocoder.nm_size, activation=keras.activations.tanh, name='lo_deltadelta_nm')(l_in)
                l_out_deltadelta_nm = keras.layers.Dense(vocoder.nm_size, activation= NonLin_Tanh_Saturated(partial(nonlin_tanh_saturated, coef=2.0)), name='lo_deltadelta_nm')(l_in)
                # l_out_deltadelta_nm = keras.layers.Dense(vocoder.nm_size, activation= NonLin_Tanh_Saturated(nonlin_tanh_saturated), name='lo_deltadelta_nm')(l_in)
                layers_toconcat.extend([l_out_deltadelta_f0spec, l_out_deltadelta_nm])

    elif isinstance(vocoder, vocoders.VocoderWORLD):
        l_out_f0specaper = keras.layers.Dense(vocoder.featuressizeraw(), name='lo_f0specaper')(l_in)
        layers_toconcat.extend([l_out_f0specaper])
        if (not mlpg_wins is None) and len(mlpg_wins)>0:
            l_out_delta_f0specaper = keras.layers.Dense(vocoder.featuressizeraw(), activation=None, name='lo_delta_f0specaper')(l_in)
            layers_toconcat.extend([l_out_delta_f0specaper])
            if (not mlpg_wins is None) and len(mlpg_wins)>1:
                l_out_deltadelta_f0specaper = keras.layers.Dense(vocoder.featuressizeraw(), activation=None, name='lo_deltadelta_f0specaper')(l_in)
                layers_toconcat.extend([l_out_deltadelta_f0specaper])

    if len(layers_toconcat)==1: l_out = layers_toconcat[0]
    else:                       l_out = keras.layers.Concatenate(axis=-1, name='lo_concatenation')(layers_toconcat)

    return l_out
