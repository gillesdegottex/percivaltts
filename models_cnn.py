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

from utils import *  # Always include this first to setup a few things

import sys
import os

import numpy as np
numpy_force_random_seed()

import theano
import theano.tensor as T
import lasagne
# lasagne.random.set_rng(np.random)

from utils_theano import *
import model

# Full architectures -----------------------------------------------------------

class CstMulLayer(lasagne.layers.Layer):
    def __init__(self, incoming, cstW, **kwargs):
        super(CstMulLayer, self).__init__(incoming, **kwargs)
        self.cstW = cstW

    def get_output_for(self, x, **kwargs):
        return x*self.cstW

    # def get_output_shape_for(self, input_shape):
    #     return (input_shape[0], self.num_units)

def layer_GatedConv2DLayer(incoming, num_filters, filter_size, stride=(1, 1), pad=0, nonlinearity=lasagne.nonlinearities.rectify, name=''):
    la = lasagne.layers.Conv2DLayer(incoming, num_filters=num_filters, filter_size=filter_size, stride=stride, pad=pad, nonlinearity=nonlinearity, name=name+'.activation')
    lg = lasagne.layers.Conv2DLayer(incoming, num_filters=num_filters, filter_size=filter_size, stride=stride, pad=pad, nonlinearity=theano.tensor.nnet.nnet.sigmoid, name=name+'.gate') # TODO Could use ultra fast sigmoid
    lout = lasagne.layers.ElemwiseMergeLayer([la, lg], T.mul, cropping=None, name=name+'.mul_merge')
    return lout


class ModelCNN(model.Model):
    def __init__(self, insize, specsize, nmsize, hiddensize=512, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, ctxlayers_nb=1, ctxlayers_type='BLSTM', nbcnnlayers=4, nbfilters=8, spec_freqlen=13, nm_freqlen=7, windur=0.100, bn_axes=None, dropout_p=-1.0):
        if bn_axes is None: bn_axes=[0,1]
        outsize = 1+specsize+nmsize
        model.Model.__init__(self, insize, outsize, specsize, nmsize, hiddensize)

        self._nbcnnlayers = nbcnnlayers
        self._nbfilters = nbfilters
        self._spec_freqlen = spec_freqlen
        self._nm_freqlen = nm_freqlen
        self._windur = windur

        _winlen = int(0.5*windur/0.005)*2+1

        layer_ctx = lasagne.layers.InputLayer(shape=(None, None, insize), input_var=self._input_values, name='ctx_input')

        # Start with a few layers that is supposed to gather the useful information in the context labels
        if ctxlayers_type=='FC':     # TODO Generalize this crap by passing a function in argument
            for layi in xrange(ctxlayers_nb):
                layerstr = 'ctx_l'+str(1+layi)+'_FC{}'.format(hiddensize)
                layer_ctx = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layer_ctx, hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr), axes=bn_axes)
        elif ctxlayers_type=='BLSTM':
            grad_clipping = 50
            for layi in xrange(ctxlayers_nb):
                layerstr = 'ctx_l'+str(1+layi)+'_BLSTM{}'.format(hiddensize)
                fwd = lasagne.layers.LSTMLayer(layer_ctx, num_units=hiddensize, backwards=False, name=layerstr+'.fwd', grad_clipping=grad_clipping)
                bck = lasagne.layers.LSTMLayer(layer_ctx, num_units=hiddensize, backwards=True, name=layerstr+'.bck', grad_clipping=grad_clipping)
                layer_ctx = lasagne.layers.ConcatLayer((fwd, bck), axis=2, name=layerstr+'.concat')

        # layer_ctx = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layer_ctx, 1+specsize+nmsize, nonlinearity=nonlinearity, num_leading_axes=2, name='projection to final output size'), axes=bn_axes)

        # F0 - 1D Gated Conv layers
        # layer_f0 = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layer_ctx, 1, nonlinearity=nonlinearity, num_leading_axes=2, name='f0_projection'), axes=bn_axes)
        layer_f0 = layer_ctx
        # layer_f0 = lasagne.layers.SliceLayer(layer, indices=slice(0,1), axis=2)
        # layer_f0 = layer_ctx
        grad_clipping = 50
        for layi in xrange(2):  # TODO Params hardcoded 2
            layerstr = 'f0_l'+str(1+layi)+'_BLSTM'
            fwd = lasagne.layers.LSTMLayer(layer_f0, num_units=hiddensize, backwards=False, name=layerstr+'.fwd', grad_clipping=grad_clipping)
            bck = lasagne.layers.LSTMLayer(layer_f0, num_units=hiddensize, backwards=True, name=layerstr+'.bck', grad_clipping=grad_clipping)

            layer_f0 = lasagne.layers.ConcatLayer((fwd, bck), axis=2, name=layerstr+'_concat')
        layer_f0 = lasagne.layers.DenseLayer(layer_f0, num_units=1, nonlinearity=None, num_leading_axes=2, name='f0_lout_projection')


        # Amplitude spectrum - 2D Gated Conv layers
        layer_spec = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layer_ctx, specsize, nonlinearity=nonlinearity, num_leading_axes=2, name='spec_projection'), axes=bn_axes)
        # layer_spec = lasagne.layers.SliceLayer(layer_ctx, indices=slice(1,1+specsize), axis=2) # TODO TODO TODO rm ?
        # layer_spec = layer_ctx
        layer_spec = lasagne.layers.dimshuffle(layer_spec, [0, 'x', 1, 2], name='spec_dimshuffle')
        for layi in xrange(nbcnnlayers):
            layerstr = 'spec_l'+str(1+layi)+'_GC{}x{}x{}'.format(nbfilters,_winlen,spec_freqlen)
            layer_spec = lasagne.layers.batch_norm(layer_GatedConv2DLayer(layer_spec, nbfilters, [_winlen,spec_freqlen], stride=1, pad='same', nonlinearity=nonlinearity, name=layerstr))
            if dropout_p>0.0: layer_spec = lasagne.layers.dropout(layer_spec, p=dropout_p)
        layer_spec = lasagne.layers.Conv2DLayer(layer_spec, 1, [_winlen,spec_freqlen], pad='same', nonlinearity=None, name='spec_lout_2DC')
        layer_spec = lasagne.layers.dimshuffle(layer_spec, [0, 2, 3, 1], name='spec_dimshuffle')
        layer_spec = lasagne.layers.flatten(layer_spec, outdim=3, name='spec_flatten')

        # Noise mask - 2D Gated Conv layers
        layer_noise = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layer_ctx, nmsize, nonlinearity=nonlinearity, num_leading_axes=2, name='nm_projection'), axes=bn_axes)
        # layer_noise = lasagne.layers.SliceLayer(layer_ctx, indices=slice(1+specsize,1+specsize+nmsize), axis=2) # TODO TODO TODO rm ?
        # layer_noise = layer_ctx
        layer_noise = lasagne.layers.dimshuffle(layer_noise, [0, 'x', 1, 2], name='nm_dimshuffle')
        for layi in xrange(nbcnnlayers):
            layerstr = 'nm_l'+str(1+layi)+'_GC{}x{}x{}'.format(nbfilters,_winlen,nm_freqlen)
            layer_noise = lasagne.layers.batch_norm(layer_GatedConv2DLayer(layer_noise, nbfilters, [_winlen,nm_freqlen], pad='same', nonlinearity=nonlinearity, name=layerstr))
            if dropout_p>0.0: layer_noise = lasagne.layers.dropout(layer_noise, p=dropout_p)
        # layer_noise = lasagne.layers.Conv2DLayer(layer_noise, 1, [_winlen,nm_freqlen], pad='same', nonlinearity=None)
        layer_noise = lasagne.layers.Conv2DLayer(layer_noise, 1, [_winlen,nm_freqlen], pad='same', nonlinearity=lasagne.nonlinearities.sigmoid, name='nm_lout_2DC') # Force the output to [0,1]
        layer_noise = lasagne.layers.dimshuffle(layer_noise, [0, 2, 3, 1], name='nm_dimshuffle')
        layer_noise = lasagne.layers.flatten(layer_noise, outdim=3, name='nm_flatten')

        layer = lasagne.layers.ConcatLayer((layer_f0, layer_spec, layer_noise), axis=2, name='lout_concat')

        self.init_finish(layer) # Has to be called at the end of the __init__ to print out the architecture, get the trainable params, etc.


def ModelCNN_build_discri(discri_input_var, condition_var, specsize, nmsize, ctxsize, hiddensize=512, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, nbcnnlayers=4, nbfilters=8, spec_freqlen=13, nm_freqlen=7, ctxlayers_type='BLSTM', postlayers_nb=6, windur=0.100, bn_axes=None, use_LSweighting=True, LSWGANtransflc=0.5, LSWGANtransc=1.0/8.0, dropout_p=-1.0, use_bn=False):
    if bn_axes is None: bn_axes=[0,1]
    layer_discri = lasagne.layers.InputLayer(shape=(None, None, 1+specsize+nmsize), input_var=discri_input_var, name='input')

    _winlen = int(0.5*windur/0.005)*2+1

    layerstoconcats = []

    # F0
    if 0: # Add f0 in discriminator. Disabled bcs it makes the f0 curve very noisy
        print('f0 winlen={}'.format(_winlen))
        layer = lasagne.layers.SliceLayer(layer_discri, indices=slice(0,1), axis=2)
        layer = lasagne.layers.dimshuffle(layer, [0, 'x', 1, 2])
        layer = lasagne.layers.Conv2DLayer(layer, 1, [_winlen,1], stride=1, pad='same', nonlinearity=None)
        if use_bn: layer=lasagne.layers.batch_norm(layer)
        if dropout_p>0.0: layer=lasagne.layers.dropout(layer, p=dropout_p)
        for _ in xrange(nbcnnlayers):
            layer = layer_GatedConv2DLayer(layer, nbfilters, [_winlen,1], stride=1, pad='same', nonlinearity=nonlinearity)
            # layer = layer_GatedResConv2DLayer(layer, nbfilters, [_winlen,1], stride=1, pad='same', nonlinearity=nonlinearity)
            if use_bn: layer=lasagne.layers.batch_norm(layer)
            if dropout_p>0.0: layer=lasagne.layers.dropout(layer, p=dropout_p)
        layer = lasagne.layers.dimshuffle(layer, [0, 2, 3, 1])
        layer_f0 = lasagne.layers.flatten(layer, outdim=3)
        layerstoconcats.append(layer_f0)

    # Amplitude spectrum
    layer = lasagne.layers.SliceLayer(layer_discri, indices=slice(1,1+specsize), axis=2, name='spec_slice')

    if use_LSweighting: # Using weighted WGAN+LS
        print('WGAN Weighted LS - Discri - SPEC')
        wganls_spec_weights_ = nonlin_sigmoidparm(np.arange(specsize, dtype=theano.config.floatX),  int(LSWGANtransflc*specsize), LSWGANtransc)
        wganls_weights = theano.shared(value=np.asarray(wganls_spec_weights_), name='wganls_spec_weights_')
        layer = CstMulLayer(layer, cstW=wganls_weights, name='cstdot_wganls_weights')

    layer = lasagne.layers.dimshuffle(layer, [0, 'x', 1, 2], name='spec_dimshuffle')
    for layi in xrange(nbcnnlayers):
        layerstr = 'spec_l'+str(1+layi)+'_GC{}x{}x{}'.format(nbfilters,_winlen,spec_freqlen)
        # strides>1 make the first two Conv layers pyramidal. Increase patches' effects here and there, bad.
        layer = layer_GatedConv2DLayer(layer, nbfilters, [_winlen,spec_freqlen], pad='same', nonlinearity=nonlinearity, name=layerstr)
        if use_bn: layer=lasagne.layers.batch_norm(layer)
        if dropout_p>0.0: layer=lasagne.layers.dropout(layer, p=dropout_p)
    layer = lasagne.layers.dimshuffle(layer, [0, 2, 3, 1], name='spec_dimshuffle')
    layer_spec = lasagne.layers.flatten(layer, outdim=3, name='spec_flatten')
    layerstoconcats.append(layer_spec)

    if 1: # Add Noise mask (NM) in discriminator
        layer = lasagne.layers.SliceLayer(layer_discri, indices=slice(1+specsize,1+specsize+nmsize), axis=2, name='nm_slice')

        if use_LSweighting: # Using weighted WGAN+LS
            print('WGAN Weighted LS - Discri - NM')
            wganls_spec_weights_ = nonlin_sigmoidparm(np.arange(nmsize, dtype=theano.config.floatX),  int(LSWGANtransflc*nmsize), LSWGANtransc)
            wganls_weights = theano.shared(value=np.asarray(wganls_spec_weights_), name='wganls_spec_weights_')
            layer = CstMulLayer(layer, cstW=wganls_weights, name='cstdot_wganls_weights')

        layer = lasagne.layers.dimshuffle(layer, [0, 'x', 1, 2], name='nm_dimshuffle')
        for layi in xrange(nbcnnlayers):
            layerstr = 'nm_l'+str(1+layi)+'_GC{}x{}x{}'.format(nbfilters,_winlen,nm_freqlen)
            layer = layer_GatedConv2DLayer(layer, nbfilters, [_winlen,nm_freqlen], pad='same', nonlinearity=nonlinearity, name=layerstr)
            if use_bn: layer=lasagne.layers.batch_norm(layer)
            if dropout_p>0.0: layer=lasagne.layers.dropout(layer, p=dropout_p)
        layer = lasagne.layers.dimshuffle(layer, [0, 2, 3, 1], name='nm_dimshuffle')
        layer_bndnm = lasagne.layers.flatten(layer, outdim=3, name='nm_flatten')
        layerstoconcats.append(layer_bndnm)

    # Add the contexts
    layer_cond = lasagne.layers.InputLayer(shape=(None, None, ctxsize), input_var=condition_var, name='ctx_input')

    if ctxlayers_type=='BLSTM': # TODO Generalize this crap by passing a function in argument
        layerstr = 'ctx_l1_blstm'
        grad_clipping = 50
        fwd = lasagne.layers.LSTMLayer(layer_cond, num_units=hiddensize, backwards=False, name=layerstr+'.fwd', grad_clipping=grad_clipping)
        bck = lasagne.layers.LSTMLayer(layer_cond, num_units=hiddensize, backwards=True, name=layerstr+'.bck', grad_clipping=grad_clipping)
        # layer_cond = lasagne.layers.ConcatLayer((fwd, bck), axis=2) # It seems concat of concats doesn't work
        layerstoconcats.append(fwd)
        layerstoconcats.append(bck)
    else:
        layerstoconcats.append(layer_cond)
    layer = lasagne.layers.ConcatLayer(layerstoconcats, axis=2, name='ctx_features_concat')

    # finalize with a common FC network
    for layi in xrange(postlayers_nb):
        layerstr = 'post_l'+str(1+layi)+'_FC'+str(hiddensize)
        layer = lasagne.layers.DenseLayer(layer, hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr)
        if use_bn: layer=lasagne.layers.batch_norm(layer, axes=_bn_axes)
        # if dropout_p>0.0: layer = lasagne.layers.dropout(layer, p=dropout_p) # Bad for FC

    # output layer (linear)
    layer = lasagne.layers.DenseLayer(layer, 1, nonlinearity=None, num_leading_axes=2, name='projection') # No nonlin for this output
    print ("discri output:", layer.output_shape)
    return [layer, layer_discri, layer_cond]
