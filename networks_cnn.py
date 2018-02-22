'''
The CNN-based networks.

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

import inspect

import theano
import theano.tensor as T
import lasagne

import sys
import numpy as np
rng = np.random.RandomState(123) # As in Merlin

from utils_theano import *

# Full architectures -----------------------------------------------------------

class CstMulLayer(lasagne.layers.Layer):
    def __init__(self, incoming, cstW, **kwargs):
        super(CstMulLayer, self).__init__(incoming, **kwargs)
        self.cstW = cstW

    def get_output_for(self, input, **kwargs):
        return input*self.cstW

    # def get_output_shape_for(self, input_shape):
    #     return (input_shape[0], self.num_units)

def layer_GatedConv2DLayer(incoming, num_filters, filter_size, stride=(1, 1), pad=0, nonlinearity=lasagne.nonlinearities.rectify):
    la = lasagne.layers.Conv2DLayer(incoming, num_filters=num_filters, filter_size=filter_size, stride=stride, pad=pad, nonlinearity=nonlinearity)
    lg = lasagne.layers.Conv2DLayer(incoming, num_filters=num_filters, filter_size=filter_size, stride=stride, pad=pad, nonlinearity=theano.tensor.nnet.nnet.sigmoid) # TODO Could use ultra fast sigmoid
    lout = lasagne.layers.ElemwiseMergeLayer([la, lg], T.mul, cropping=None)
    return lout

def LA_3xFC_splitfeats_2xGC2D_C2D(input_var, hiddensize, labsize, specsize, nmsize, nonlinearity=lasagne.nonlinearities.tanh, bn_axes=[0,1], dropout_p=-1.0):

    _nbfilters = 8
    _gen_nblayers = 4
    _spec_freqlen = 13
    _nm_freqlen = 7
    _windur = 0.100 #[s]
    _winlen = int(0.5*_windur/0.005)*2+1

    updatess = []

    layer = lasagne.layers.InputLayer(shape=(None, None, labsize), input_var=input_var)

    layer = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layer, hiddensize, nonlinearity=nonlinearity, num_leading_axes=2), axes=bn_axes)
    layer = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layer, hiddensize, nonlinearity=nonlinearity, num_leading_axes=2), axes=bn_axes)
    layer = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layer, 1+specsize+nmsize, nonlinearity=nonlinearity, num_leading_axes=2), axes=bn_axes)

    # F0 - 1D Gated Conv layers
    layer_f0 = lasagne.layers.SliceLayer(layer, indices=slice(0,1), axis=2)
    layer_f0 = lasagne.layers.dimshuffle(layer_f0, [0, 'x', 1, 2])
    for _ in xrange(_gen_nblayers):
        layer_f0 = lasagne.layers.batch_norm(layer_GatedConv2DLayer(layer_f0, _nbfilters, [_winlen,1], stride=1, pad='same', nonlinearity=nonlinearity))
        if dropout_p>0.0: layer_f0 = lasagne.layers.dropout(layer_f0, p=dropout_p)
    layer_f0 = lasagne.layers.Conv2DLayer(layer_f0, 1, [_winlen,1], stride=1, pad='same', nonlinearity=None)
    layer_f0 = lasagne.layers.dimshuffle(layer_f0, [0, 2, 3, 1])
    layer_f0 = lasagne.layers.flatten(layer_f0, outdim=3)

    # Amplitude spectrum - 2D Gated Conv layers
    layer_spec = lasagne.layers.SliceLayer(layer, indices=slice(1,1+specsize), axis=2)
    layer_spec = lasagne.layers.dimshuffle(layer_spec, [0, 'x', 1, 2])
    for _ in xrange(_gen_nblayers):
        layer_spec = lasagne.layers.batch_norm(layer_GatedConv2DLayer(layer_spec, _nbfilters, [_winlen,_spec_freqlen], stride=1, pad='same', nonlinearity=nonlinearity))
        if dropout_p>0.0: layer_spec = lasagne.layers.dropout(layer_spec, p=dropout_p)
    layer_spec = lasagne.layers.Conv2DLayer(layer_spec, 1, [_winlen,_spec_freqlen], stride=1, pad='same', nonlinearity=None)
    layer_spec = lasagne.layers.dimshuffle(layer_spec, [0, 2, 3, 1])
    layer_spec = lasagne.layers.flatten(layer_spec, outdim=3)

    # Noise mask - 2D Gated Conv layers
    layer_noise = lasagne.layers.SliceLayer(layer, indices=slice(1+specsize,1+specsize+nmsize), axis=2)
    layer_noise = lasagne.layers.dimshuffle(layer_noise, [0, 'x', 1, 2])
    for _ in xrange(_gen_nblayers):
        layer_noise = lasagne.layers.batch_norm(layer_GatedConv2DLayer(layer_noise, _nbfilters, [_winlen,_nm_freqlen], stride=1, pad='same', nonlinearity=nonlinearity))
        if dropout_p>0.0: layer_noise = lasagne.layers.dropout(layer_noise, p=dropout_p)
    # layer_noise = lasagne.layers.Conv2DLayer(layer_noise, 1, [_winlen,_nm_freqlen], stride=1, pad='same', nonlinearity=None)
    layer_noise = lasagne.layers.Conv2DLayer(layer_noise, 1, [_winlen,_nm_freqlen], stride=1, pad='same', nonlinearity=lasagne.nonlinearities.sigmoid) # Force the output to [0,1]
    layer_noise = lasagne.layers.dimshuffle(layer_noise, [0, 2, 3, 1])
    layer_noise = lasagne.layers.flatten(layer_noise, outdim=3)

    layer = lasagne.layers.ConcatLayer((layer_f0, layer_spec, layer_noise), axis=2)

    return layer, updatess

def build_discri_split(discri_input_var, condition_var, specsize, nmsize, ctxsize, hiddensize, dropout_p=-1.0, use_bn=False):
    layer_discri = lasagne.layers.InputLayer(shape=(None, None, 1+specsize+nmsize), input_var=discri_input_var)

    _nonlinearity = lasagne.nonlinearities.very_leaky_rectify
    _bn_axes = [0,1]
    _gen_nblayers = 4
    _nbfilters = 8
    _spec_freqlen = 13
    _nm_freqlen = 7
    _gen_intermfc_nblayers = 4
    _use_LSweighting = True

    _windur = 0.100 #[s]
    _winlen = int(0.5*_windur/0.005)*2+1

    layerstoconcats = []

    # F0
    if 0: # Add f0 in discriminator
        print('f0 winlen={}'.format(_winlen))
        layer = lasagne.layers.SliceLayer(layer_discri, indices=slice(0,1), axis=2)
        layer = lasagne.layers.dimshuffle(layer, [0, 'x', 1, 2])
        layer = lasagne.layers.Conv2DLayer(layer, 1, [_winlen,1], stride=1, pad='same', nonlinearity=None)
        if use_bn: layer=lasagne.layers.batch_norm(layer)
        if dropout_p>0.0: layer=lasagne.layers.dropout(layer, p=dropout_p)
        for _ in xrange(_gen_nblayers):
            layer = layer_GatedConv2DLayer(layer, _nbfilters, [_winlen,1], stride=1, pad='same', nonlinearity=_nonlinearity)
            # layer = layer_GatedResConv2DLayer(layer, _nbfilters, [_winlen,1], stride=1, pad='same', nonlinearity=_nonlinearity)
            if use_bn: layer=lasagne.layers.batch_norm(layer)
            if dropout_p>0.0: layer=lasagne.layers.dropout(layer, p=dropout_p)
        layer = lasagne.layers.dimshuffle(layer, [0, 2, 3, 1])
        layer_f0 = lasagne.layers.flatten(layer, outdim=3)
        layerstoconcats.append(layer_f0)

    # Amplitude spectrum
    print('spec winlen={} freqlen={}'.format(_winlen, _spec_freqlen))
    layer = lasagne.layers.SliceLayer(layer_discri, indices=slice(1,1+specsize), axis=2)

    if _use_LSweighting: # Using weighted WGAN+LS
        print('WGAN Weighted LS - Discri part')
        wganls_spec_weights_ = nonlin_sigmoidparm(np.arange(specsize, dtype=theano.config.floatX),  int(specsize/2), 1.0/8.0)   # TODO
        wganls_weights = theano.shared(value=np.asarray(wganls_spec_weights_), name='wganls_spec_weights_')
        layer = CstMulLayer(layer, cstW=wganls_weights, name='cstdot_wganls_weights')

    layer = lasagne.layers.dimshuffle(layer, [0, 'x', 1, 2])
    stride_init = 1 # TODO
    layer = lasagne.layers.Conv2DLayer(layer, stride_init*_nbfilters, [_winlen,_spec_freqlen], stride=[1,stride_init], pad='same', nonlinearity=None)
    if use_bn: layer=lasagne.layers.batch_norm(layer)
    if dropout_p>0.0: layer=lasagne.layers.dropout(layer, p=dropout_p)
    for _ in xrange(_gen_nblayers):
        layer = layer_GatedConv2DLayer(layer, stride_init*_nbfilters, [_winlen,_spec_freqlen], stride=1, pad='same', nonlinearity=_nonlinearity)
        if use_bn: layer=lasagne.layers.batch_norm(layer)
        # layer = lasagne.layers.batch_norm(layer_GatedResConv2DLayer(layer, _nbfilters, [_winlen,freqlen], stride=1, pad='same', nonlinearity=_nonlinearity))
        if dropout_p>0.0: layer=lasagne.layers.dropout(layer, p=dropout_p)
    layer = lasagne.layers.dimshuffle(layer, [0, 2, 3, 1])
    layer_spec = lasagne.layers.flatten(layer, outdim=3)
    layerstoconcats.append(layer_spec)

    # Noise mask
    if 1: # Add NM in discriminator
        # TODO Should make it pyramidal
        print('bndnm winlen={} freqlen={}'.format(_winlen, _nm_freqlen))
        layer = lasagne.layers.SliceLayer(layer_discri, indices=slice(1+specsize,1+specsize+nmsize), axis=2)

        if _use_LSweighting: # Using weighted WGAN+LS
            print('WGAN Weighted LS - Discri part')
            wganls_spec_weights_ = nonlin_sigmoidparm(np.arange(nmsize, dtype=theano.config.floatX),  int(nmsize/2), 1.0/8.0)   # TODO
            wganls_weights = theano.shared(value=np.asarray(wganls_spec_weights_), name='wganls_spec_weights_')
            layer = CstMulLayer(layer, cstW=wganls_weights, name='cstdot_wganls_weights')

        layer = lasagne.layers.dimshuffle(layer, [0, 'x', 1, 2])
        layer = lasagne.layers.Conv2DLayer(layer, 1, [_winlen,_nm_freqlen], stride=1, pad='same', nonlinearity=None)
        if use_bn: layer=lasagne.layers.batch_norm(layer)
        if dropout_p>0.0: layer=lasagne.layers.dropout(layer, p=dropout_p)
        for _ in xrange(_gen_nblayers):
            layer = layer_GatedConv2DLayer(layer, _nbfilters, [_winlen,_nm_freqlen], stride=1, pad='same', nonlinearity=_nonlinearity)
            if use_bn: layer=lasagne.layers.batch_norm(layer)
            # layer = lasagne.layers.batch_norm(layer_GatedResConv2DLayer(layer, _nbfilters, [_winlen,_nm_freqlen], stride=1, pad='same', nonlinearity=_nonlinearity))
            if dropout_p>0.0: layer=lasagne.layers.dropout(layer, p=dropout_p)
        layer = lasagne.layers.dimshuffle(layer, [0, 2, 3, 1])
        layer_bndnm = lasagne.layers.flatten(layer, outdim=3)
        layerstoconcats.append(layer_bndnm)

    # Add the contexts
    layer_cond = lasagne.layers.InputLayer(shape=(None, None, ctxsize), input_var=condition_var)

    layerstoconcats.append(layer_cond)
    layer = lasagne.layers.ConcatLayer(layerstoconcats, axis=2)

    # finalize with a common FC network
    for _ in xrange(_gen_intermfc_nblayers):
        layer = lasagne.layers.DenseLayer(layer, hiddensize, nonlinearity=_nonlinearity, num_leading_axes=2)
        if use_bn: layer=lasagne.layers.batch_norm(layer, axes=_bn_axes)
        # if dropout_p>0.0: layer = lasagne.layers.dropout(layer, p=dropout_p) # Bad for FC

    # output layer (linear)
    layer = lasagne.layers.DenseLayer(layer, 1, nonlinearity=None, num_leading_axes=2) # No nonlin for this output
    print ("discri output:", layer.output_shape)
    return [layer, layer_discri, layer_cond]
