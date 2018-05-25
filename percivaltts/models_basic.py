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
import lasagne.layers as ll
# lasagne.random.set_rng(np.random)

from backend_theano import *
import model

import vocoders

def layer_final(l_hid, vocoder, mlpg_wins):
    layers_toconcat = []

    if isinstance(vocoder, vocoders.VocoderPML):
        l_out_f0spec = ll.DenseLayer(l_hid, num_units=1+vocoder.spec_size, nonlinearity=None, num_leading_axes=2, name='lo_f0spec')
        l_out_nm = ll.DenseLayer(l_hid, num_units=vocoder.nm_size, nonlinearity=lasagne.nonlinearities.sigmoid, num_leading_axes=2, name='lo_nm') # sig is best among nonlin_saturatedsigmoid nonlin_tanh_saturated nonlin_tanh_bysigmoid
        layers_toconcat.extend([l_out_f0spec, l_out_nm])
        if len(mlpg_wins)>0:
            l_out_f0spec = ll.DenseLayer(l_hid, num_units=1+vocoder.spec_size, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_d')
            l_out_nm = ll.DenseLayer(l_hid, num_units=vocoder.nm_size, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2, name='lo_nm_d')
            layers_toconcat.extend([l_out_f0spec, l_out_nm])
            if len(mlpg_wins)>1:
                l_out_f0spec = ll.DenseLayer(l_hid, num_units=1+vocoder.spec_size, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_dd')
                l_out_nm = ll.DenseLayer(l_hid, num_units=vocoder.nm_size, nonlinearity=partial(nonlin_tanh_saturated, coef=2.0), num_leading_axes=2, name='lo_nm_dd')
                layers_toconcat.extend([l_out_f0spec, l_out_nm])

    elif isinstance(vocoder, vocoders.VocoderWORLD):
        layers_toconcat.append(ll.DenseLayer(l_hid, num_units=vocoder.featuressize(), nonlinearity=None, num_leading_axes=2, name='lo_f0specaper'))
        if len(mlpg_wins)>0:
            layers_toconcat.append(ll.DenseLayer(l_hid, num_units=vocoder.featuressize(), nonlinearity=None, num_leading_axes=2, name='lo_f0specaper_d'))
            if len(mlpg_wins)>1:
                layers_toconcat.append(ll.DenseLayer(l_hid, num_units=vocoder.featuressize(), nonlinearity=None, num_leading_axes=2, name='lo_f0specaper_dd'))

    if len(layers_toconcat)==1: l_out = layers_toconcat[0]
    else:                       l_out = ll.ConcatLayer(layers_toconcat, axis=2, name='lo_concatenation')

    return l_out


def layer_LSTM(l_hid, hiddensize, nonlinearity, backwards=False, grad_clipping=50, name=""):
    '''
    That's a custom LSTM layer that seems to converge faster.
    '''
    ingate = ll.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0))
    forgetgate = ll.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0))
    outgate = ll.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0))
    cell = ll.Gate(W_cell=None, W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0), nonlinearity=nonlinearity)
    # The final nonline should be TanH otherwise it doesn't converge (why?)
    # by default peepholes=True
    fwd = ll.LSTMLayer(l_hid, num_units=hiddensize, backwards=backwards, ingate=ingate, forgetgate=forgetgate, outgate=outgate, cell=cell, grad_clipping=grad_clipping, nonlinearity=lasagne.nonlinearities.tanh, name=name)

    return fwd

class ModelFC(model.Model):
    def __init__(self, insize, vocoder, mlpg_wins=[], hiddensize=256, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, nblayers=6, bn_axes=None, dropout_p=-1.0):
        if bn_axes is None: bn_axes=[0,1]
        model.Model.__init__(self, insize, vocoder, hiddensize)

        l_hid = ll.InputLayer(shape=(None, None, insize), input_var=self._input_values, name='input_conditional')

        #if 0:
            ## Make a special one to train bottleneck features
            #l_hid = ll.DenseLayer(l_hid, num_units=32, nonlinearity=nonlinearity, num_leading_axes=2, name='btln')
            ## This one without batch normalisation, because we want the traditional bias term.

        for layi in xrange(nblayers):
            layerstr = 'l'+str(1+layi)+'_FC{}'.format(hiddensize)

            l_hid = ll.DenseLayer(l_hid, num_units=hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr)

            # Add batch normalisation
            if len(bn_axes)>0: l_hid=ll.batch_norm(l_hid, axes=bn_axes)

            # Add dropout (after batchnorm)
            if dropout_p>0.0: l_hid=ll.dropout(l_hid, p=dropout_p)

        l_out = layer_final(l_hid, vocoder, mlpg_wins)

        self.init_finish(l_out) # Has to be called at the end of the __init__ to print out the architecture, get the trainable params, etc.


class ModelBGRU(model.Model):
    def __init__(self, insize, vocoder, mlpg_wins=[], hiddensize=256, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, nblayers=3, bn_axes=None, dropout_p=-1.0, grad_clipping=50):
        if bn_axes is None: bn_axes=[] # Recurrent nets don't like batch norm [ref. needed]
        model.Model.__init__(self, insize, vocoder, hiddensize)

        if len(bn_axes)>0: warnings.warn('ModelBGRU: You are using bn_axes={}, but batch normalisation is supposed to make Recurrent Neural Networks (RNNS) unstable [ref. needed]'.format(bn_axes))

        l_hid = ll.InputLayer(shape=(None, None, insize), input_var=self._input_values, name='input_conditional')

        for layi in xrange(nblayers):
            layerstr = 'l'+str(1+layi)+'_BGRU{}'.format(hiddensize)

            fwd = ll.GRULayer(l_hid, num_units=hiddensize, backwards=False, name=layerstr+'.fwd', grad_clipping=grad_clipping)
            bck = ll.GRULayer(l_hid, num_units=hiddensize, backwards=True, name=layerstr+'.bck', grad_clipping=grad_clipping)
            l_hid = ll.ConcatLayer((fwd, bck), axis=2)

            # Add batch normalisation
            if len(bn_axes)>0: l_hid=ll.batch_norm(l_hid, axes=bn_axes)   # Not be good for recurrent nets!

            # Add dropout (after batchnorm)
            if dropout_p>0.0: l_hid=ll.dropout(l_hid, p=dropout_p)

        l_out = layer_final(l_hid, vocoder, mlpg_wins)

        self.init_finish(l_out) # Has to be called at the end of the __init__ to print out the architecture, get the trainable params, etc.


class ModelBLSTM(model.Model):
    def __init__(self, insize, vocoder, mlpg_wins=[], hiddensize=256, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, nblayers=3, bn_axes=None, dropout_p=-1.0, grad_clipping=50):
        if bn_axes is None: bn_axes=[] # Recurrent nets don't like batch norm [ref needed]
        model.Model.__init__(self, insize, vocoder, hiddensize)

        if len(bn_axes)>0: warnings.warn('ModelBLSTM: You are using bn_axes={}, but batch normalisation is supposed to make Recurrent Neural Networks (RNNS) unstable [ref. needed]'.format(bn_axes))

        l_hid = ll.InputLayer(shape=(None, None, insize), input_var=self._input_values, name='input_conditional')

        for layi in xrange(nblayers):
            layerstr = 'l'+str(1+layi)+'_BLSTM{}'.format(hiddensize)

            fwd = layer_LSTM(l_hid, hiddensize, nonlinearity=nonlinearity, backwards=False, grad_clipping=grad_clipping, name=layerstr+'.fwd')
            bck = layer_LSTM(l_hid, hiddensize, nonlinearity=nonlinearity, backwards=True, grad_clipping=grad_clipping, name=layerstr+'.bck')
            l_hid = ll.ConcatLayer((fwd, bck), axis=2)

            # Add batch normalisation
            if len(bn_axes)>0: l_hid=ll.batch_norm(l_hid, axes=bn_axes)

            # Add dropout (after batchnorm)
            if dropout_p>0.0: l_hid=ll.dropout(l_hid, p=dropout_p)

        l_out = layer_final(l_hid, vocoder, mlpg_wins)

        self.init_finish(l_out) # Has to be called at the end of the __init__ to print out the architecture, get the trainable params, etc.
