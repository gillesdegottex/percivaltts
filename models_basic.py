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

from utils import *  # Always include this first to setup a few things

import sys
import os
import warnings
from functools import partial

import numpy as np
# np.random.seed(123) # Comment this line if you want non-deterministic runs

# import theano
# import theano.tensor as T
import lasagne
# lasagne.random.set_rng(np.random)

from utils_theano import *
import model


class ModelFC(model.Model):
    def __init__(self, insize, outsize, specsize, nmsize, hiddensize=512, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, nblayers=6, bn_axes=None, dropout_p=-1.0):
        if bn_axes is None: bn_axes=[0,1]
        model.Model.__init__(self, insize, outsize, specsize, nmsize, hiddensize)

        l_hid = lasagne.layers.InputLayer(shape=(None, None, insize), input_var=self._input_values, name='input_conditional')

        #if 0:
            ## Make a special one to train bottleneck features
            #l_hid = lasagne.layers.DenseLayer(l_hid, num_units=32, nonlinearity=nonlinearity, num_leading_axes=2, name='btln')
            ## This one without batch normalisation, because we want the traditional bias term.

        for layi in xrange(nblayers):
            layerstr = 'l'+str(1+layi)
            l_hid = lasagne.layers.DenseLayer(l_hid, num_units=hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr)

            # Add batch normalisation
            if len(bn_axes)>0: l_hid=lasagne.layers.batch_norm(l_hid, axes=bn_axes, name=layerstr+'_FC.bn')

            # Add dropout (after batchnorm)
            if dropout_p>0.0: l_hid=lasagne.layers.dropout(l_hid, p=dropout_p)

        layers_toconcat = []
        l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec')
        l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=lasagne.nonlinearities.sigmoid, num_leading_axes=2, name='lo_nm') # sig is best among nonlin_saturatedsigmoid nonlin_tanh_saturated nonlin_tanh_bysigmoid
        layers_toconcat.extend([l_out_f0spec, l_out_nm])

        # TODO fn
        if outsize>(1+specsize+nmsize):
            l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_d')
            l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2, name='lo_nm_d')
            layers_toconcat.extend([l_out_f0spec, l_out_nm])
            if outsize>2*(1+specsize+nmsize):
                l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_dd')
                l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=partial(nonlin_tanh_saturated, coef=2.0), num_leading_axes=2, name='lo_nm_dd')
                layers_toconcat.extend([l_out_f0spec, l_out_nm])

        l_out = lasagne.layers.ConcatLayer(layers_toconcat, axis=2, name='lo_concatenation')

        self.init_finish(l_out) # Has to be called at the end of the __init__ to print out the architecture, get the trainable params, etc.


class ModelBGRU(model.Model):
    def __init__(self, insize, outsize, specsize, nmsize, hiddensize=512, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, nblayers=3, bn_axes=None, dropout_p=-1.0, grad_clipping=50):
        if bn_axes is None: bn_axes=[] # Recurrent nets don't like batch norm [ref. needed]
        model.Model.__init__(self, insize, outsize, specsize, nmsize, hiddensize)

        if len(bn_axes)>0: warnings.warn('ModelBGRU: You are using bn_axes={}, but batch normalisation is supposed to make Recurrent Neural Networks (RNNS) unstable [ref. needed]'.format(bn_axes))

        l_hid = lasagne.layers.InputLayer(shape=(None, None, insize), input_var=self._input_values, name='input_conditional')

        for layi in xrange(nblayers):
            layerstr = 'l'+str(1+layi)

            if 0:
                fwd = lasagne.layers.GRULayer(l_hid, num_units=hiddensize, backwards=False, name=layerstr+'.fwd', grad_clipping=grad_clipping)
                bck = lasagne.layers.GRULayer(l_hid, num_units=hiddensize, backwards=True, name=layerstr+'.bck', grad_clipping=grad_clipping)
                l_hid = lasagne.layers.ConcatLayer((fwd, bck), axis=2)

                # Add batch normalisation
                if len(bn_axes)>0: l_hid=lasagne.layers.batch_norm(l_hid, axes=bn_axes, name=layerstr+'.bn')   # Not be good for recurrent nets!

                # Add dropout (after batchnorm)
                if dropout_p>0.0: l_hid=lasagne.layers.dropout(l_hid, p=dropout_p)

            else:
                resetgate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0), W_cell=None)
                updategate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0), W_cell=None)
                hidden_update = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(np.sqrt(2.0/(1.0+(1.0/3.0))**2)), W_hid=lasagne.init.Orthogonal(np.sqrt(2.0/(1+(1.0/3.0))**2)), W_cell=None, nonlinearity=nonlinearity)
                fwd = lasagne.layers.GRULayer(l_hid, num_units=hiddensize, backwards=False, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, grad_clipping=grad_clipping, name=layerstr+'_GRU.fwd')

                resetgate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0), W_cell=None)
                updategate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0), W_cell=None)
                hidden_update = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(np.sqrt(2.0/(1.0+(1.0/3.0))**2)), W_hid=lasagne.init.Orthogonal(np.sqrt(2.0/(1.0+(1.0/3.0))**2)), W_cell=None, nonlinearity=nonlinearity)
                bck = lasagne.layers.GRULayer(l_hid, num_units=hiddensize, backwards=True, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, grad_clipping=grad_clipping, name=layerstr+'_GRU.bck')

                l_hid = lasagne.layers.ConcatLayer((fwd, bck), axis=2, name=layerstr+'_concat')

                # Add batch normalisation
                if len(bn_axes)>0: l_hid=lasagne.layers.batch_norm(l_hid, axes=bn_axes, name=layerstr+'.bn') # Not good for recurrent nets!


        layers_toconcat = []
        l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec')
        l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=lasagne.nonlinearities.sigmoid, num_leading_axes=2, name='lo_nm') # sig is best among nonlin_saturatedsigmoid nonlin_tanh_saturated nonlin_tanh_bysigmoid
        layers_toconcat.extend([l_out_f0spec, l_out_nm])

        # TODO fn
        if outsize>(1+specsize+nmsize):
            l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_d')
            l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2, name='lo_nm_d')
            layers_toconcat.extend([l_out_f0spec, l_out_nm])
            if outsize>2*(1+specsize+nmsize):
                l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_dd')
                l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=partial(nonlin_tanh_saturated, coef=2.0), num_leading_axes=2, name='lo_nm_dd')
                layers_toconcat.extend([l_out_f0spec, l_out_nm])

        l_out = lasagne.layers.ConcatLayer(layers_toconcat, axis=2, name='lo_concatenation')

        self.init_finish(l_out) # Has to be called at the end of the __init__ to print out the architecture, get the trainable params, etc.


class ModelBLSTM(model.Model):
    def __init__(self, insize, outsize, specsize, nmsize, hiddensize=512, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, nblayers=3, bn_axes=None, dropout_p=-1.0, grad_clipping=50):
        if bn_axes is None: bn_axes=[] # Recurrent nets don't like batch norm [ref needed]
        model.Model.__init__(self, insize, outsize, specsize, nmsize, hiddensize)

        if len(bn_axes)>0: warnings.warn('ModelBLSTM: You are using bn_axes={}, but batch normalisation is supposed to make Recurrent Neural Networks (RNNS) unstable [ref. needed]'.format(bn_axes))

        l_hid = lasagne.layers.InputLayer(shape=(None, None, insize), input_var=self._input_values, name='input_conditional')

        for layi in xrange(nblayers):
            layerstr = 'l'+str(1+layi)
            if 0:
                fwd = lasagne.layers.LSTMLayer(l_hid, num_units=hiddensize, backwards=False, name=layerstr+'.fwd', grad_clipping=grad_clipping)
                bck = lasagne.layers.LSTMLayer(l_hid, num_units=hiddensize, backwards=True, name=layerstr+'.bck', grad_clipping=grad_clipping)
                l_hid = lasagne.layers.ConcatLayer((fwd, bck), axis=2)

                # Add batch normalisation
                if len(bn_axes)>0: l_hid=lasagne.layers.batch_norm(l_hid, axes=bn_axes, name=layerstr+'.bn')

                # Add dropout (after batchnorm)
                if dropout_p>0.0: l_hid=lasagne.layers.dropout(l_hid, p=dropout_p)

            else:
                ingate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0))
                forgetgate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0))
                outgate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0))
                cell = lasagne.layers.Gate(W_cell=None, W_in=lasagne.init.Orthogonal(np.sqrt(2.0/(1.0+(1.0/3.0))**2)), W_hid=lasagne.init.Orthogonal(np.sqrt(2.0/(1+(1.0/3.0))**2)), nonlinearity=nonlinearity)
                # The final nonline should be TanH otherwise it doesn't converge (why?)
                # by default peepholes=True
                fwd = lasagne.layers.LSTMLayer(l_hid, num_units=hiddensize, backwards=False, ingate=ingate, forgetgate=forgetgate, outgate=outgate, cell=cell, grad_clipping=grad_clipping, nonlinearity=lasagne.nonlinearities.tanh, name=layerstr+'_LSTM.fwd')


                ingate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0))
                forgetgate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0))
                outgate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0))
                cell = lasagne.layers.Gate(W_cell=None, W_in=lasagne.init.Orthogonal(np.sqrt(2.0/(1.0+(1.0/3.0))**2)), W_hid=lasagne.init.Orthogonal(np.sqrt(2.0/(1+(1.0/3.0))**2)), nonlinearity=nonlinearity)
                # The final nonline should be TanH otherwise it doesn't converge (why?)
                # by default peepholes=True
                bck = lasagne.layers.LSTMLayer(l_hid, num_units=hiddensize, backwards=True, ingate=ingate, forgetgate=forgetgate, outgate=outgate, cell=cell, grad_clipping=grad_clipping, nonlinearity=lasagne.nonlinearities.tanh, name=layerstr+'_LSTM.bck')

                l_hid = lasagne.layers.ConcatLayer((fwd, bck), axis=2, name=layerstr+'_concat')

                # Add batch normalisation
                if len(bn_axes)>0: l_hid=lasagne.layers.batch_norm(l_hid, axes=bn_axes, name=layerstr+'.bn') # Not good for recurrent nets!


        layers_toconcat = []
        l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec')
        l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=lasagne.nonlinearities.sigmoid, num_leading_axes=2, name='lo_nm') # sig is best among nonlin_saturatedsigmoid nonlin_tanh_saturated nonlin_tanh_bysigmoid
        layers_toconcat.extend([l_out_f0spec, l_out_nm])

        # TODO fn
        if outsize>(1+specsize+nmsize):
            l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_d')
            l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2, name='lo_nm_d')
            layers_toconcat.extend([l_out_f0spec, l_out_nm])
            if outsize>2*(1+specsize+nmsize):
                l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_dd')
                l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=partial(nonlin_tanh_saturated, coef=2.0), num_leading_axes=2, name='lo_nm_dd')
                layers_toconcat.extend([l_out_f0spec, l_out_nm])

        l_out = lasagne.layers.ConcatLayer(layers_toconcat, axis=2, name='lo_concatenation')

        self.init_finish(l_out) # Has to be called at the end of the __init__ to print out the architecture, get the trainable params, etc.
