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

from external.pulsemodel import sigproc as sp

from tensorflow import keras
import tensorflow.keras.layers as kl

from networktts import *

class Critic:

    input_features = None
    input_ctx = None
    output = None

    vocoder = None
    ctxsize = -1
    cfgarch = None

    def __init__(self, vocoder, ctxsize, cfgarch, use_LSspectralweighting=True, LSWGANtransfreqcutoff=4000, LSWGANtranscoef=1.0/8.0, use_WGAN_incnoisefeature=False):

        self.vocoder = vocoder
        self.ctxsize = ctxsize
        self.cfgarch = cfgarch

        bn = False

        self.input_features = keras.layers.Input(shape=(None, vocoder.featuressize()), name='input_features')

        l_toconcat = []

        # Spectrum
        # l_f0 = kl.Lambda(lambda x: x[:,:,:1])(self.input_features)
        l_spec = kl.Lambda(lambda x: x[:,:,1:1+vocoder.specsize()])(self.input_features)
        # l_nm = kl.Lambda(lambda x: x[:,:,1+vocoder.specsize():1+vocoder.specsize()+vocoder.noisesize()])(self.input_features)

        #TODO Add spectral weighting here

        l_spec = kl.Reshape([-1,vocoder.specsize(), 1])(l_spec)

        for _ in xrange(cfgarch.arch_gen_nbcnnlayers):
            l_spec = kl.Conv2D(cfgarch.arch_gen_nbfilters, [cfgarch.arch_critic_ctx_winlen,cfgarch.arch_spec_freqlen], strides=(1, 1), padding='same', dilation_rate=(1, 1), data_format='channels_last')(l_spec)
            l_spec = keras.layers.LeakyReLU(alpha=0.3)(l_spec)

        l_spec = kl.Reshape([-1,l_spec.shape[-2]*l_spec.shape[-1]])(l_spec)

        l_toconcat.append(l_spec)

        self.input_ctx = keras.layers.Input(shape=(None, self.ctxsize), name='input_ctx')
        l_ctx = self.input_ctx
        for _ in xrange(cfgarch.arch_ctx_nbcnnlayers):
            l_ctx = pCNN1D(l_ctx, cfgarch.arch_ctx_nbfilters, cfgarch.arch_critic_ctx_winlen, bn=bn)
        l_spec = pFC(l_spec, self.cfgarch.arch_hiddenwidth, bn=bn)
        l_spec = pFC(l_spec, self.cfgarch.arch_hiddenwidth, bn=bn)

        l_toconcat.append(l_ctx)

        l_post = kl.Concatenate(axis=-1, name='lo_concatenation')(l_toconcat)

        l_post = pFC(l_post, self.cfgarch.arch_hiddenwidth, bn=bn)
        l_post = pFC(l_post, self.cfgarch.arch_hiddenwidth, bn=bn)
        l_post = pFC(l_post, self.cfgarch.arch_hiddenwidth, bn=bn)
        l_post = pFC(l_post, self.cfgarch.arch_hiddenwidth, bn=bn)
        l_post = pFC(l_post, self.cfgarch.arch_hiddenwidth, bn=bn)
        l_post = pFC(l_post, self.cfgarch.arch_hiddenwidth, bn=bn)

        self.output = kl.Dense(1, activation=None)(l_post)
