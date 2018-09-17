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

from tensorflow import keras
import tensorflow.keras.layers as kl

import networks

# import gan_normalization2

class Critic:

    input_features = None
    input_ctx = None
    output = None

    vocoder = None
    ctxsize = -1
    cfgarch = None

    def __init__(self, vocoder, ctxsize, cfgarch, use_LSweighting=True, LSWGANtransfreqcutoff=4000, LSWGANtranscoef=1.0/8.0, use_WGAN_incnoisefeature=False):
        nonlinearity=keras.layers.LeakyReLU(alpha=0.3) # TODO Move to cfgarch

        self.vocoder = vocoder
        self.ctxsize = ctxsize
        self.cfgarch = cfgarch

        self.input_features = keras.layers.Input(shape=(None, vocoder.featuressize()), name='input_features')

        use_bias = True     # TODO , use_bias=False

        # l_f0 = kl.Lambda(lambda x: x[:,:,:1])(self.input_features)
        l_spec = kl.Lambda(lambda x: x[:,:,1:1+vocoder.specsize()])(self.input_features)
        # l_nm = kl.Lambda(lambda x: x[:,:,1+vocoder.specsize():1+vocoder.specsize()+vocoder.noisesize()])(self.input_features)

        # l_spec = kl.Reshape([-1,vocoder.specsize(), 1])(l_spec)
        #
        # # l_spec = kl.Conv2D(cfgarch.arch_nbfilters, [cfgarch.arch_winlen,cfgarch.arch_spec_freqlen], strides=(1, 2), padding='same', dilation_rate=(1, 1), use_bias=use_bias, data_format='channels_last')(l_spec)
        # # l_spec = nonlinearity(l_spec)
        # #
        # # l_spec = kl.Conv2D(cfgarch.arch_nbfilters, [cfgarch.arch_winlen,cfgarch.arch_spec_freqlen], strides=(1, 2), padding='same', dilation_rate=(1, 1), use_bias=use_bias, data_format='channels_last')(l_spec)
        # # l_spec = nonlinearity(l_spec)
        #
        # # l_spec = kl.Conv2D(cfgarch.arch_nbfilters, [cfgarch.arch_winlen,cfgarch.arch_spec_freqlen], strides=(1, 1), padding='same', dilation_rate=(1, 1), use_bias=use_bias, data_format='channels_last')(l_spec)
        # # l_spec = nonlinearity(l_spec)
        # #
        # # l_spec = kl.Conv2D(cfgarch.arch_nbfilters, [cfgarch.arch_winlen,cfgarch.arch_spec_freqlen], strides=(1, 1), padding='same', dilation_rate=(1, 1), use_bias=use_bias, data_format='channels_last')(l_spec)
        # # l_spec = nonlinearity(l_spec)
        #
        # l_spec = kl.Conv2D(1, [cfgarch.arch_winlen,cfgarch.arch_spec_freqlen], strides=(1, 1), padding='same', dilation_rate=(1, 1), use_bias=True, data_format='channels_last')(l_spec)
        # l_spec = nonlinearity(l_spec)
        # l_spec = kl.Reshape([-1,l_spec.shape[-2]])(l_spec)

        l_pre = l_spec

        l_pre = kl.Dense(self.cfgarch.arch_hiddensize, activation=None, use_bias=use_bias)(l_pre)
        # l_pre = gan_normalization2.GANBatchNormalization(axis=-1)(l_pre)  # TODO find a proper normalisation for the critic
        l_pre = nonlinearity(l_pre)

        l_pre = kl.Dense(self.cfgarch.arch_hiddensize, activation=None, use_bias=use_bias)(l_pre)
        # l_pre = gan_normalization2.GANBatchNormalization(axis=-1)(l_pre)
        l_pre = nonlinearity(l_pre)

        l_pre = kl.Dense(self.cfgarch.arch_hiddensize, activation=None, use_bias=use_bias)(l_pre)
        # l_pre = gan_normalization2.GANBatchNormalization(axis=-1)(l_pre)
        l_pre = nonlinearity(l_pre)

        self.input_ctx = keras.layers.Input(shape=(None, self.ctxsize), name='input_ctx')

        l_ctx = networks.network_context_preproc(self.input_ctx, cfgarch, use_bn=False)

        l_post = kl.Concatenate(axis=-1, name='lo_concatenation')([l_pre, l_ctx])

        l_post = kl.Dense(self.cfgarch.arch_hiddensize, activation=None, use_bias=use_bias)(l_post)
        # l_post = gan_normalization2.GANBatchNormalization(axis=-1)(l_post)    # TODO Crashes
        l_post = nonlinearity(l_post)

        l_post = kl.Dense(self.cfgarch.arch_hiddensize, activation=None, use_bias=use_bias)(l_post)
        # l_post = gan_normalization2.GANBatchNormalization(axis=-1)(l_post)
        l_post = nonlinearity(l_post)

        l_post = kl.Dense(self.cfgarch.arch_hiddensize, activation=None, use_bias=use_bias)(l_post)
        # l_post = gan_normalization2.GANBatchNormalization(axis=-1)(l_post)
        l_post = nonlinearity(l_post)

        # l_post = kl.Dense(self.cfgarch.arch_hiddensize, activation=None, use_bias=use_bias)(l_post)
        # # l_post = gan_normalization2.GANBatchNormalization(axis=-1)(l_post)
        # l_post = nonlinearity(l_post)

        l_post = kl.Dense(1, activation=None)(l_post)
        self.output = nonlinearity(l_post)
