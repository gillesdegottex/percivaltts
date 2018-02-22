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

import sys
import os

from functools import partial
import warnings

import theano
import theano.tensor as T
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/external/Lasagne/')
import lasagne

import numpy as np
rng = np.random.RandomState(123) # TODO Doesn't seems to be working properly bcs each run is different

from utils_theano import *

def LA_NxBGRU(input_values, hiddensize, labsize, featsize, specsize, nmsize, nblayers, nonlinearity=lasagne.nonlinearities.tanh, bn_axes=[], dropout_p=-1.0, grad_clipping=50):

    if len(bn_axes)>0: warnings.warn('LA_NxBGRU: You are using bn_axes={}, but batch normalisation is supposed to make Recurrent Neural Networks (RNNS) unstable'.format(bn_axes))

    updatess = []

    l_hid = lasagne.layers.InputLayer(shape=(None, None, labsize), input_var=input_values, name='input_conditional')

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
            fwd = lasagne.layers.GRULayer(l_hid, num_units=hiddensize, backwards=False, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, grad_clipping=grad_clipping)

            resetgate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0), W_cell=None)
            updategate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0), W_cell=None)
            hidden_update = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(np.sqrt(2.0/(1.0+(1.0/3.0))**2)), W_hid=lasagne.init.Orthogonal(np.sqrt(2.0/(1.0+(1.0/3.0))**2)), W_cell=None, nonlinearity=nonlinearity)
            bck = lasagne.layers.GRULayer(l_hid, num_units=hiddensize, backwards=True, resetgate=resetgate, updategate=updategate, hidden_update=hidden_update, grad_clipping=grad_clipping)

            l_hid = lasagne.layers.ConcatLayer((fwd, bck), axis=2)

            # Add batch normalisation
            if len(bn_axes)>0: l_hid=lasagne.layers.batch_norm(l_hid, axes=bn_axes, name=layerstr+'.bn') # Not good for recurrent nets!


    layers_toconcat = []
    l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec')
    l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=lasagne.nonlinearities.sigmoid, num_leading_axes=2, name='lo_nm') # sig is best among nonlin_saturatedsigmoid nonlin_tanh_saturated nonlin_tanh_bysigmoid
    layers_toconcat.extend([l_out_f0spec, l_out_nm])

    if featsize>=2*(1+specsize+nmsize):
        l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_d')
        l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2, name='lo_nm_d')
        layers_toconcat.extend([l_out_f0spec, l_out_nm])
        if featsize>=3*(1+specsize+nmsize):
            l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_dd')
            l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=partial(nonlin_tanh_saturated, coef=2.0), num_leading_axes=2, name='lo_nm_dd')
            layers_toconcat.extend([l_out_f0spec, l_out_nm])

    l_out = lasagne.layers.ConcatLayer(layers_toconcat, axis=2, name='lo_concatenation')

    return l_out, updatess


def LA_NxBLSTM(input_values, hiddensize, labsize, featsize, specsize, nmsize, nblayers, nonlinearity=lasagne.nonlinearities.tanh, bn_axes=[], dropout_p=-1.0, grad_clipping=50):

    if len(bn_axes)>0: warnings.warn('LA_NxBLSTM: You are using bn_axes={}, but batch normalisation is supposed to make Recurrent Neural Networks (RNNS) unstable'.format(bn_axes))

    updatess = []

    l_hid = lasagne.layers.InputLayer(shape=(None, None, labsize), input_var=input_values, name='input_conditional')

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
            fwd = lasagne.layers.LSTMLayer(l_hid, num_units=hiddensize, backwards=False, ingate=ingate, forgetgate=forgetgate, outgate=outgate, cell=cell, grad_clipping=grad_clipping, nonlinearity=lasagne.nonlinearities.tanh)


            ingate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0))
            forgetgate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0))
            outgate = lasagne.layers.Gate(W_in=lasagne.init.Orthogonal(1.0), W_hid=lasagne.init.Orthogonal(1.0))
            cell = lasagne.layers.Gate(W_cell=None, W_in=lasagne.init.Orthogonal(np.sqrt(2.0/(1.0+(1.0/3.0))**2)), W_hid=lasagne.init.Orthogonal(np.sqrt(2.0/(1+(1.0/3.0))**2)), nonlinearity=nonlinearity)
            # The final nonline should be TanH otherwise it doesn't converge (why?)
            # by default peepholes=True
            bck = lasagne.layers.LSTMLayer(l_hid, num_units=hiddensize, backwards=True, ingate=ingate, forgetgate=forgetgate, outgate=outgate, cell=cell, grad_clipping=grad_clipping, nonlinearity=lasagne.nonlinearities.tanh)

            l_hid = lasagne.layers.ConcatLayer((fwd, bck), axis=2)

            # Add batch normalisation
            if len(bn_axes)>0: l_hid=lasagne.layers.batch_norm(l_hid, axes=bn_axes, name=layerstr+'.bn') # Not good for recurrent nets!


    layers_toconcat = []
    l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec')
    l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=lasagne.nonlinearities.sigmoid, num_leading_axes=2, name='lo_nm') # sig is best among nonlin_saturatedsigmoid nonlin_tanh_saturated nonlin_tanh_bysigmoid
    layers_toconcat.extend([l_out_f0spec, l_out_nm])

    if featsize>=2*(1+specsize+nmsize):
        l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_d')
        l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2, name='lo_nm_d')
        layers_toconcat.extend([l_out_f0spec, l_out_nm])
        if featsize>=3*(1+specsize+nmsize):
            l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_dd')
            l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=partial(nonlin_tanh_saturated, coef=2.0), num_leading_axes=2, name='lo_nm_dd')
            layers_toconcat.extend([l_out_f0spec, l_out_nm])

    l_out = lasagne.layers.ConcatLayer(layers_toconcat, axis=2, name='lo_concatenation')

    return l_out, updatess


def LA_NxFC(input_values, hiddensize, labsize, specsize, nmsize, nblayers, nonlinearity=lasagne.nonlinearities.tanh, bn_axes=[0,1], dropout_p=-1.0):

    updatess = []

    featsize = 1+specsize+nmsize

    l_hid = lasagne.layers.InputLayer(shape=(None, None, labsize), input_var=input_values, name='input_conditional')

    #if 0:
        ## Make a special one to train bottleneck features
        #l_hid = lasagne.layers.DenseLayer(l_hid, num_units=32, nonlinearity=nonlinearity, num_leading_axes=2, name='btln')
        ## This one without batch normalisation, because we want the traditional bias term.

    for layi in xrange(nblayers):
        layerstr = 'l'+str(1+layi)
        l_hid = lasagne.layers.DenseLayer(l_hid, num_units=hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr)

        # Add batch normalisation
        if len(bn_axes)>0: l_hid=lasagne.layers.batch_norm(l_hid, axes=bn_axes, name=layerstr+'.bn')

        # Add dropout (after batchnorm)
        if dropout_p>0.0: l_hid=lasagne.layers.dropout(l_hid, p=dropout_p)


    layers_toconcat = []
    l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec')
    l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=lasagne.nonlinearities.sigmoid, num_leading_axes=2, name='lo_nm') # sig is best among nonlin_saturatedsigmoid nonlin_tanh_saturated nonlin_tanh_bysigmoid
    layers_toconcat.extend([l_out_f0spec, l_out_nm])

    if featsize>=2*(1+specsize+nmsize):
        l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_d')
        l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=lasagne.nonlinearities.tanh, num_leading_axes=2, name='lo_nm_d')
        layers_toconcat.extend([l_out_f0spec, l_out_nm])
        if featsize>=3*(1+specsize+nmsize):
            l_out_f0spec = lasagne.layers.DenseLayer(l_hid, num_units=1+specsize, nonlinearity=None, num_leading_axes=2, name='lo_f0spec_dd')
            l_out_nm = lasagne.layers.DenseLayer(l_hid, num_units=nmsize, nonlinearity=partial(nonlin_tanh_saturated, coef=2.0), num_leading_axes=2, name='lo_nm_dd')
            layers_toconcat.extend([l_out_f0spec, l_out_nm])

    l_out = lasagne.layers.ConcatLayer(layers_toconcat, axis=2, name='lo_concatenation')

    return l_out, updatess
