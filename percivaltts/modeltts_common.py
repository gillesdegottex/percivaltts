'''
Definitions of specific models (e.g. the DCNN used in Percival paper)

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

import modeltts
import networktts

from tensorflow import keras
import tensorflow.keras.layers as kl


class Generic(modeltts.ModelTTS):
    def __init__(self, ctxsize, vocoder, fmodel=None, layertypes=['FC', 'FC', 'FC'], nameprefix=None, cfgarch=None):
        modeltts.ModelTTS.__init__(self, ctxsize, vocoder)

        if not fmodel is None:
            self.load(fmodel)
            self.kerasmodel.summary()
            self.ctxsize = self.kerasmodel.layers[0].input_shape[-1]
            return

        if nameprefix is None: nameprefix=''

        l_in = kl.Input(shape=(None, ctxsize), name=nameprefix+'input.conditional')

        l_out = networktts.network_generic(l_in, layertypes=layertypes, cfgarch=cfgarch)

        l_out = networktts.network_final(l_out, vocoder, mlpg_wins=vocoder.mlpg_wins)

        self.kerasmodel = keras.Model(inputs=l_in, outputs=l_out)
        self.kerasmodel.summary()

        # from keras.utils import plot_model
        # plot_model(self._kerasmodel, to_file='model.png')


class DCNNF0SpecNoiseFeatures(modeltts.ModelTTS):
    def __init__(self, ctxsize, vocoder, cfgarch, nameprefix=None):
        bn_axis=-1
        modeltts.ModelTTS.__init__(self, ctxsize, vocoder)

        if nameprefix is None: nameprefix=''

        l_in = kl.Input(shape=(None, ctxsize), name=nameprefix+'input.conditional')

        # l_ctx = networktts.network_context_preproc(l_in, cfgarch.arch_ctx_winlen, cfgarch, bn=True)
        # l_ctx = networktts.GaussianNoiseInput(width=100)(l_ctx)
        l_ctx = l_in
        for _ in xrange(cfgarch.arch_ctx_nbcnnlayers):
            l_ctx = networktts.pCNN1D(l_ctx, cfgarch.arch_hiddenwidth, cfgarch.arch_ctx_winlen)
        l_ctx = networktts.pFC(l_ctx, cfgarch.arch_hiddenwidth)
        l_ctx = networktts.pFC(l_ctx, cfgarch.arch_hiddenwidth)

        # F0
        l_f0 = l_ctx
        # l_f0 = networktts.pCNN1D(l_f0, cfgarch.arch_hiddenwidth, int(0.5*0.200/vocoder.shift)*2+1) # TODO TODO TODO hardcoded 0.200s
        l_f0 = networktts.pBLSTM(l_f0, width=cfgarch.arch_hiddenwidth) # TODO TODO TODO Use BLSTM as in the paper
        l_f0 = kl.Dense(1, activation=None, use_bias=True)(l_f0)

        # Spec
        l_spec = l_ctx
        # l_spec = kl.Dense(cfgarch.arch_hiddenwidth, use_bias=False)(l_spec)
        # l_spec = kl.BatchNormalization(axis=bn_axis)(l_spec)
        # l_spec = kl.LeakyReLU(alpha=0.3)(l_spec)
        # l_spec = kl.Dense(cfgarch.arch_hiddenwidth, use_bias=False)(l_spec)
        # l_spec = kl.BatchNormalization(axis=bn_axis)(l_spec)
        # l_spec = kl.LeakyReLU(alpha=0.3)(l_spec)

        l_spec = kl.Dense(vocoder.specsize(), use_bias=True)(l_spec)   # Projection
        l_spec = kl.Reshape([-1,vocoder.specsize(), 1])(l_spec) # Add the channels after the spectral dimension
        for _ in xrange(cfgarch.arch_gen_nbcnnlayers):
            l_spec = networktts.pCNN2D(l_spec, cfgarch.arch_gen_nbfilters, cfgarch.arch_gen_winlen, cfgarch.arch_spec_freqlen)
            # l_spec = networktts.pGCNN2D(l_spec, cfgarch.arch_gen_nbfilters, cfgarch.arch_gen_winlen, cfgarch.arch_spec_freqlen)
        l_spec = kl.Conv2D(1, [cfgarch.arch_gen_winlen,cfgarch.arch_spec_freqlen], strides=(1, 1), padding='same', dilation_rate=(1, 1), use_bias=True, activation=None, data_format='channels_last')(l_spec)
        l_spec = kl.Reshape([-1,l_spec.shape[-2]])(l_spec)

        # NM
        if 0:
            l_nm = kl.Dense(cfgarch.arch_hiddenwidth, use_bias=False)(l_nm)
            l_nm = kl.BatchNormalization(axis=bn_axis)(l_nm)
            l_nm = kl.LeakyReLU(alpha=0.3)(l_nm)
            l_nm = kl.Dense(vocoder.noisesize(), use_bias=True)(l_ctx)   # Projection
            l_nm = kl.Reshape([-1,vocoder.noisesize(), 1])(l_nm) # Add the channels after the spectral dimension
            for _ in xrange(cfgarch.arch_gen_nbcnnlayers/2):
                l_nm = kl.Conv2D(cfgarch.arch_gen_nbfilters, [cfgarch.arch_gen_winlen,cfgarch.arch_noise_freqlen], strides=(1, 1), padding='same', dilation_rate=(1, 1), use_bias=False, data_format='channels_last')(l_nm)
                l_nm = kl.BatchNormalization(axis=bn_axis)(l_nm)
                l_nm = kl.LeakyReLU(alpha=0.3)(l_nm)
            # l_nm = kl.Dense(vocoder.noisesize(), activation='sigmoid')(l_nm)  # TODO Sigmoid is for PML !
            l_nm = kl.Conv2D(1, [cfgarch.arch_gen_winlen,cfgarch.arch_noise_freqlen], strides=(1, 1), padding='same', dilation_rate=(1, 1), use_bias=True, activation='sigmoid', data_format='channels_last')(l_nm)
            l_nm = kl.Reshape([-1,l_nm.shape[-2]])(l_nm)
        else:
            l_nm = l_ctx
            for _ in xrange(cfgarch.arch_gen_nbcnnlayers/2):
                l_nm = networktts.pFC(l_nm, cfgarch.arch_hiddenwidth)
            l_nm = kl.Dense(vocoder.noisesize(), activation='sigmoid')(l_nm)  # TODO Sigmoid is for PML !

        l_out = kl.Concatenate(axis=-1)([l_f0, l_spec, l_nm])

        self.kerasmodel = keras.Model(inputs=l_in, outputs=l_out)
        self.kerasmodel.summary()
