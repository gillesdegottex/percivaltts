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
    def __init__(self, ctxsize, vocoder, cfgarch, layertypes=['FC', 'FC', 'FC'], nameprefix=None):
        bn_axis=-1
        modeltts.ModelTTS.__init__(self, ctxsize, vocoder, cfgarch=cfgarch)

        if nameprefix is None: nameprefix=''

        l_in = keras.layers.Input(shape=(None, ctxsize), name=nameprefix+'input.conditional')
        l_out = l_in

        for layi in xrange(len(layertypes)):
            layerstr = nameprefix+'l'+str(1+layi)+'_{}{}'.format(layertypes[layi], cfgarch.arch_hiddensize)

            if layertypes[layi]=='DO':
                l_out = keras.layers.Dropout(0.2, noise_shape=(5, 1, None))(l_out)  # TODO 5
            if layertypes[layi]=='FC':
                l_out = keras.layers.Dense(cfgarch.arch_hiddensize, use_bias=False, name=layerstr)(l_out) # TODO , kernel_initializer='normal' blows up the values
                l_out = keras.layers.BatchNormalization(axis=bn_axis)(l_out)
                l_out = keras.layers.LeakyReLU(alpha=0.3)(l_out)                  # TODO TODO TODO Using nonlinearity doesn't work! Need to clone it maybe
            if layertypes[layi]=='GRU' or layertypes[layi]=='BGRU':
                l_gru = keras.layers.GRU(cfgarch.arch_hiddensize, activation='tanh', use_bias=True, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, reset_after=False, name=layerstr)
                if layertypes[layi]=='BGRU':    l_out=keras.layers.Bidirectional(l_gru)(l_out)
                else:                           l_out=l_gru(l_out)
            if layertypes[layi]=='LSTM' or layertypes[layi]=='BLSTM':
                # TODO TODO TODO Very very slow!!!
                # TODO TODO TODO Try CuDNNLSTM
                # l_out = keras.layers.SimpleRNN(cfgarch.arch_hiddensize, activation='tanh', return_sequences=True)(l_out)
                l_lstm = keras.layers.LSTM(cfgarch.arch_hiddensize, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=True, name=layerstr)
                if layertypes[layi]=='BLSTM':   l_out=keras.layers.Bidirectional(l_lstm)(l_out)
                else:                           l_out=l_lstm(l_out)
                # TODO TODO TODO Test batch normalisation
                # l_out = keras.layers.LeakyReLU(alpha=0.3)(l_out) # TODO Makes it unstable. tanh works
            elif isinstance(layertypes[layi], list):
                if layertypes[layi][0]=='CNN':
                    nbfilters = layertypes[layi][1]
                    winlen = layertypes[layi][2]
                    l_out = keras.layers.Conv1D(nbfilters*cfgarch.arch_hiddensize, winlen, strides=1, padding='same', dilation_rate=1, activation=None, use_bias=False, kernel_initializer='glorot_uniform', bias_initializer='zeros')(l_out)
                    # TODO TODO TODO Test dilation_rate>1 # TODO TODO TODO nbfilters*cfgarch.arch_hiddensize
                    l_out = keras.layers.BatchNormalization(axis=bn_axis)(l_out)
                    l_out = keras.layers.LeakyReLU(alpha=0.3)(l_out)
                # if layertypes[layi][0]=='GCNN':

        l_out = modeltts.layer_final(l_out, vocoder, mlpg_wins=[])

        self._kerasmodel = keras.Model(inputs=l_in, outputs=l_out)
        self._kerasmodel.summary()

        # from keras.utils import plot_model
        # plot_model(self._kerasmodel, to_file='model.png')
