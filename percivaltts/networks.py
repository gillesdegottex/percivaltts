'''
Functions returning some outpout(s) given input(s), without object

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

from tensorflow import keras
import tensorflow.keras.layers as kl

import vocoders

def DenseBNLReLU03(input, hiddensize, bn_axis=-1):
    out = keras.layers.Dense(hiddensize, use_bias=False, name=layerstr)(input)
    out = keras.layers.BatchNormalization(axis=bn_axis)(out)
    out = keras.layers.LeakyReLU(alpha=0.3)(out)
    return out


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
