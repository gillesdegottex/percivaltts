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
# from models_cnn import CstMulLayer
# from models_cnn import TileLayer
# from models_cnn import layer_GatedConv2DLayer
# from models_cnn import layer_context

from tensorflow import keras


class ModelGeneric(modeltts.ModelTTS):
    def __init__(self, ctxsize, vocoder, mlpg_wins=[], layertypes=['FC', 'FC', 'FC'], hiddensize=256, nonlinearity=keras.layers.LeakyReLU(alpha=0.3), bn_axis=None, nameprefix=None):
        if bn_axis is None: bn_axis=-1
        modeltts.ModelTTS.__init__(self, ctxsize, vocoder, hiddensize)

        if nameprefix is None: nameprefix=''

        l_in = keras.layers.Input(shape=(None, ctxsize), name=nameprefix+'input.conditional')
        l_out = l_in

        for layi in xrange(len(layertypes)):
            layerstr = nameprefix+'l'+str(1+layi)+'_{}{}'.format(layertypes[layi], hiddensize)

            if layertypes[layi]=='FC':
                l_out = keras.layers.Dense(hiddensize, use_bias=False, name=layerstr)(l_out)
                l_out = keras.layers.BatchNormalization(axis=bn_axis)(l_out)
                l_out = keras.layers.LeakyReLU(alpha=0.3)(l_out)                  # TODO TODO TODO Using nonlinearity doesn't work! Need to clone it maybe
            if layertypes[layi]=='GRU' or layertypes[layi]=='BGRU':
                l_gru = keras.layers.GRU(hiddensize, activation='tanh', use_bias=True, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, reset_after=False, name=layerstr)
                if layertypes[layi]=='BGRU':    l_out=keras.layers.Bidirectional(l_gru)(l_out)
                else:                           l_out=l_gru(l_out)
            if layertypes[layi]=='LSTM' or layertypes[layi]=='BLSTM':
                # TODO TODO TODO Try CuDNNLSTM
                # l_out = keras.layers.SimpleRNN(hiddensize, activation='tanh', return_sequences=True)(l_out)
                l_lstm = keras.layers.LSTM(hiddensize, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, name=layerstr)
                if layertypes[layi]=='BLSTM':   l_out=keras.layers.Bidirectional(l_lstm)(l_out)
                else:                           l_out=l_lstm(l_out)
                # TODO TODO TODO Test batch normalisation
                # l_out = keras.layers.LeakyReLU(alpha=0.3)(l_out) # TODO Makes it unstable. tanh works
            elif isinstance(layertypes[layi], list):
                if layertypes[layi][0]=='CNN':
                    nbfilters = layertypes[layi][1]
                    winlen = layertypes[layi][2]
                    l_out = keras.layers.Conv1D(nbfilters*hiddensize, winlen, strides=1, padding='same', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros')(l_out)
                    # TODO TODO TODO Test dilation_rate>1 # TODO TODO TODO nbfilters*hiddensize
                    l_out = keras.layers.LeakyReLU(alpha=0.3)(l_out)                  # TODO only tanh works
                # if layertypes[layi][0]=='GCNN':

        l_out = modeltts.layer_final(l_out, vocoder, mlpg_wins=[])

        self._kerasmodel = keras.Model(inputs=l_in, outputs=l_out)
        self._kerasmodel.summary()

        # from keras.utils import plot_model
        # plot_model(self._kerasmodel, to_file='model.png')

    def build_critic(self, nonlinearity=nonlin_very_leaky_rectify, postlayers_nb=6, use_LSweighting=True, LSWGANtransfreqcutoff=4000, LSWGANtranscoef=1.0/8.0, use_WGAN_incnoisefeature=False):

        l_in = keras.layers.Input(shape=(None, self.vocoder.featuressize()), name='input')

        l_in_ctx = keras.layers.Input(shape=(None, self.ctxsize), name='input_ctx')   # TODO TODO TODO

        l_out = keras.layers.Concatenate(axis=-1, name='lo_concatenation')([l_in, l_in_ctx])
        # l_out = l_in

        l_out = keras.layers.Dense(self.hiddensize, activation=None)(l_out)    # TODO , use_bias=False
        l_out = keras.layers.LeakyReLU(alpha=0.3)(l_out)           # TODO TODO TODO Using nonlinearity doesn't work! Need to clone it maybe

        l_out = keras.layers.Dense(self.hiddensize, activation=None)(l_out)    # TODO , use_bias=False
        l_out = keras.layers.LeakyReLU(alpha=0.3)(l_out)           # TODO TODO TODO Using nonlinearity doesn't work! Need to clone it maybe

        l_out = keras.layers.Dense(self.hiddensize, activation=None)(l_out)    # TODO , use_bias=False
        l_out = keras.layers.LeakyReLU(alpha=0.3)(l_out)           # TODO TODO TODO Using nonlinearity doesn't work! Need to clone it maybe

        l_out = keras.layers.Dense(1, activation=None)(l_out)    # TODO , use_bias=False
        l_out = keras.layers.LeakyReLU(alpha=0.3)(l_out)           # TODO TODO TODO Using nonlinearity doesn't work! Need to clone it maybe

        return l_in, l_in_ctx, l_out
