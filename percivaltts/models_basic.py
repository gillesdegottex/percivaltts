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

import warnings
from functools import partial

numpy_force_random_seed()

# import theano
# import theano.tensor as T
import lasagne
# lasagne.random.set_rng(np.random)

from backend_theano import *
import model

import vocoders

def layer_final(l_hid, vocoder, mlpg_wins):
    layers_toconcat = []

    if isinstance(vocoder, vocoders.VocoderPML):
        l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+vocoder.spec_size, nonlinearity=None, num_leading_axes=2, name='lo_f0spec')
        l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=vocoder.nm_size, nonlinearity=lasagne.nonlinearities.sigmoid, num_leading_axes=2, name='lo_nm') # sig is best among nonlin_saturatedsigmoid nonlin_tanh_saturated nonlin_tanh_bysigmoid
        layers_toconcat.extend([l_out_f0spec, l_out_nm])
        if len(mlpg_wins)>0:
            l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+vocoder.spec_size, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_d')
            l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=vocoder.nm_size, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2, name='lo_nm_d')
            layers_toconcat.extend([l_out_f0spec, l_out_nm])
            if len(mlpg_wins)>1:
                l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+vocoder.spec_size, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_dd')
                l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=vocoder.nm_size, nonlinearity=partial(nonlin_tanh_saturated, coef=2.0), num_leading_axes=2, name='lo_nm_dd')
                layers_toconcat.extend([l_out_f0spec, l_out_nm])

    elif isinstance(vocoder, vocoders.VocoderWORLD):
        layers_toconcat.append(lasagne.layers.DenseLayer(l_hid, num_units=vocoder.featuressize(), nonlinearity=None, num_leading_axes=2, name='lo_f0specaper'))
        if len(mlpg_wins)>0:
            layers_toconcat.append(lasagne.layers.DenseLayer(l_hid, num_units=vocoder.featuressize(), nonlinearity=None, num_leading_axes=2, name='lo_f0specaper_d'))
            if len(mlpg_wins)>1:
                layers_toconcat.append(lasagne.layers.DenseLayer(l_hid, num_units=vocoder.featuressize(), nonlinearity=None, num_leading_axes=2, name='lo_f0specaper_dd'))

    if len(layers_toconcat)==1: l_out = layers_toconcat[0]
    else:                       l_out = lasagne.layers.ConcatLayer(layers_toconcat, axis=2, name='lo_concatenation')

    return l_out


class ModelFC(model.Model):
    def __init__(self, insize, vocoder, mlpg_wins=[], hiddensize=256, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, nblayers=6, bn_axes=None, dropout_p=-1.0):
        if bn_axes is None: bn_axes=[0,1]
        model.Model.__init__(self, insize, vocoder, hiddensize)

        l_hid = lasagne.layers.InputLayer(shape=(None, None, insize), input_var=self._input_values, name='input_conditional')

        #if 0:
            ## Make a special one to train bottleneck features
            #l_hid = lasagne.layers.DenseLayer(l_hid, num_units=32, nonlinearity=nonlinearity, num_leading_axes=2, name='btln')
            ## This one without batch normalisation, because we want the traditional bias term.

        for layi in xrange(nblayers):
            layerstr = 'l'+str(1+layi)+'_FC{}'.format(hiddensize)

            l_hid = lasagne.layers.DenseLayer(l_hid, num_units=hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr)

            # Add batch normalisation
            if len(bn_axes)>0: l_hid=lasagne.layers.batch_norm(l_hid, axes=bn_axes, name=layerstr+'_FC.bn')

            # Add dropout (after batchnorm)
            if dropout_p>0.0: l_hid=lasagne.layers.dropout(l_hid, p=dropout_p)

        l_out = layer_final(l_hid, vocoder, mlpg_wins)

        self.init_finish(l_out) # Has to be called at the end of the __init__ to print out the architecture, get the trainable params, etc.

    def build_discri(self, discri_input_var, condition_var, outsize, ctxsize, hiddensize=256, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, nblayers=6, bn_axes=None, use_LSweighting=True, LSWGANtransflc=0.5, LSWGANtransc=1.0/8.0, dropout_p=-1.0, use_bn=False):
        if bn_axes is None: bn_axes=[0,1]
        layer_discri = lasagne.layers.InputLayer(shape=(None, None, outsize), input_var=discri_input_var, name='input')

        layer = lasagne.layers.DenseLayer(layer_discri, hiddensize, nonlinearity=nonlinearity, num_leading_axes=2)

        layerstoconcats = []
        layerstoconcats.append(layer)

        # Add the contexts
        layer_ctx_input = lasagne.layers.InputLayer(shape=(None, None, ctxsize), input_var=condition_var, name='ctx_input')
        layer = lasagne.layers.DenseLayer(layer_ctx_input, hiddensize, nonlinearity=nonlinearity, num_leading_axes=2)
        layerstoconcats.append(layer)

        # Concatenate the features analysis with the contexts...
        layer = lasagne.layers.ConcatLayer(layerstoconcats, axis=2, name='ctx_features_concat')

        # ... and finalize with a common FC network
        for layi in xrange(nblayers-1):
            layerstr = 'post_l'+str(1+layi)+'_FC'+str(hiddensize)
            layer = lasagne.layers.DenseLayer(layer, hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr)
            if use_bn: layer=lasagne.layers.batch_norm(layer, axes=_bn_axes)
            # if dropout_p>0.0: layer = lasagne.layers.dropout(layer, p=dropout_p) # Bad for FC

        # output layer (linear)
        layer = lasagne.layers.DenseLayer(layer, 1, nonlinearity=None, num_leading_axes=2, name='projection') # No nonlin for this output
        return [layer, layer_discri, layer_ctx_input]


class ModelBGRU(model.Model):
    def __init__(self, insize, vocoder, mlpg_wins=[], hiddensize=256, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, nblayers=3, bn_axes=None, dropout_p=-1.0, grad_clipping=50):
        if bn_axes is None: bn_axes=[] # Recurrent nets don't like batch norm [ref. needed]
        model.Model.__init__(self, insize, outsize, specsize, nmsize, hiddensize)

        if len(bn_axes)>0: warnings.warn('ModelBGRU: You are using bn_axes={}, but batch normalisation is supposed to make Recurrent Neural Networks (RNNS) unstable [ref. needed]'.format(bn_axes))

        l_hid = lasagne.layers.InputLayer(shape=(None, None, insize), input_var=self._input_values, name='input_conditional')

        for layi in xrange(nblayers):
            layerstr = 'l'+str(1+layi)+'_BGRU{}'.format(hiddensize)

            fwd = lasagne.layers.GRULayer(l_hid, num_units=hiddensize, backwards=False, name=layerstr+'.fwd', grad_clipping=grad_clipping)
            bck = lasagne.layers.GRULayer(l_hid, num_units=hiddensize, backwards=True, name=layerstr+'.bck', grad_clipping=grad_clipping)
            l_hid = lasagne.layers.ConcatLayer((fwd, bck), axis=2)

            # Add batch normalisation
            if len(bn_axes)>0: l_hid=lasagne.layers.batch_norm(l_hid, axes=bn_axes, name=layerstr+'.bn')   # Not be good for recurrent nets!

            # Add dropout (after batchnorm)
            if dropout_p>0.0: l_hid=lasagne.layers.dropout(l_hid, p=dropout_p)

        l_out = layer_final(l_hid, vocoder, mlpg_wins)

        self.init_finish(l_out) # Has to be called at the end of the __init__ to print out the architecture, get the trainable params, etc.


class ModelBLSTM(model.Model):
    def __init__(self, insize, vocoder, mlpg_wins=[], hiddensize=256, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, nblayers=3, bn_axes=None, dropout_p=-1.0, grad_clipping=50):
        if bn_axes is None: bn_axes=[] # Recurrent nets don't like batch norm [ref needed]
        model.Model.__init__(self, insize, vocoder, hiddensize)

        if len(bn_axes)>0: warnings.warn('ModelBLSTM: You are using bn_axes={}, but batch normalisation is supposed to make Recurrent Neural Networks (RNNS) unstable [ref. needed]'.format(bn_axes))

        l_hid = lasagne.layers.InputLayer(shape=(None, None, insize), input_var=self._input_values, name='input_conditional')

        for layi in xrange(nblayers):
            layerstr = 'l'+str(1+layi)+'_BLSTM{}'.format(hiddensize)

            fwd = lasagne.layers.LSTMLayer(l_hid, num_units=hiddensize, backwards=False, name=layerstr+'.fwd', grad_clipping=grad_clipping)
            bck = lasagne.layers.LSTMLayer(l_hid, num_units=hiddensize, backwards=True, name=layerstr+'.bck', grad_clipping=grad_clipping)
            l_hid = lasagne.layers.ConcatLayer((fwd, bck), axis=2)

            # Add batch normalisation
            if len(bn_axes)>0: l_hid=lasagne.layers.batch_norm(l_hid, axes=bn_axes, name=layerstr+'.bn')

            # Add dropout (after batchnorm)
            if dropout_p>0.0: l_hid=lasagne.layers.dropout(l_hid, p=dropout_p)

        l_out = layer_final(l_hid, vocoder, mlpg_wins)

        self.init_finish(l_out) # Has to be called at the end of the __init__ to print out the architecture, get the trainable params, etc.
