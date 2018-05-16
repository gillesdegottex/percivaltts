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

import models_basic
from models_cnn import CstMulLayer
from models_cnn import layer_GatedConv2DLayer


class ModelGeneric(model.Model):
    def __init__(self, insize, vocoder, mlpg_wins=[], layertypes=['FC', 'FC', 'BLSTM'], hiddensize=256, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, bn_axes=None, dropout_p=-1.0, grad_clipping=50):
        if bn_axes is None: bn_axes=[0,1]
        model.Model.__init__(self, insize, vocoder, hiddensize)

        l_hid = lasagne.layers.InputLayer(shape=(None, None, insize), input_var=self._input_values, name='input_conditional')

        for layi in xrange(len(layertypes)):
            layerstr = 'l'+str(1+layi)+'_{}{}'.format(layertypes[layi], hiddensize)

            if layertypes[layi]=='FC':
                l_hid = lasagne.layers.DenseLayer(l_hid, num_units=hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr)

                if len(bn_axes)>0: l_hid=lasagne.layers.batch_norm(l_hid, axes=bn_axes, name=layerstr+'.bn') # Add batch normalisation

            elif layertypes[layi]=='BLSTM':
                fwd = models_basic.layer_LSTM(l_hid, hiddensize, nonlinearity=nonlinearity, backwards=False, grad_clipping=grad_clipping, name=layerstr+'.fwd')
                bck = models_basic.layer_LSTM(l_hid, hiddensize, nonlinearity=nonlinearity, backwards=True, grad_clipping=grad_clipping, name=layerstr+'.bck')
                l_hid = lasagne.layers.ConcatLayer((fwd, bck), axis=2)

                # Don't add batch norm for RNN-based layers

            elif layertypes[layi]=='CNN':
                # l_hid = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(l_hid, hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name='projection'), axes=bn_axes)
                l_hid = lasagne.layers.dimshuffle(l_hid, [0, 'x', 1, 2], name='dimshuffle_to_2DCNN')
                nbfilters = 3
                winlen = 21
                nbcnnlayers = 1
                for layicnn in xrange(nbcnnlayers):
                    layerstr = 'l'+str(1+layi)+'_'+str(layicnn)+'CNN{}x{}x{}'.format(nbfilters,winlen,1)
                    l_hid = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(l_hid, num_filters=nbfilters, filter_size=[winlen,1], stride=1, pad='same', nonlinearity=nonlinearity, name=layerstr))
                    # l_hid = lasagne.layers.batch_norm(layer_GatedConv2DLayer(l_hid, nbfilters, [_winlen,spec_freqlen], stride=1, pad='same', nonlinearity=nonlinearity, name=layerstr))
                    # if dropout_p>0.0: l_hid = lasagne.layers.dropout(l_hid, p=dropout_p)
                # l_hid = lasagne.layers.Conv2DLayer(l_hid, 1, [_winlen,spec_freqlen], pad='same', nonlinearity=None, name='spec_lout_2DC')
                l_hid = lasagne.layers.dimshuffle(l_hid, [0, 2, 3, 1], name='dimshuffle_back')
                l_hid = lasagne.layers.flatten(l_hid, outdim=3, name='flatten')

            # Add dropout (after batchnorm)
            if dropout_p>0.0: l_hid=lasagne.layers.dropout(l_hid, p=dropout_p)

        l_out = models_basic.layer_final(l_hid, vocoder, mlpg_wins)

        self.init_finish(l_out) # Has to be called at the end of the __init__ to print out the architecture, get the trainable params, etc.


    # Discri arch for WGAN TODO TODO TODO
    _nbcnnlayers = 8
    _nbfilters = 16
    _spec_freqlen = 5
    _noise_freqlen = 5
    _windur = 0.025

    def build_discri(self, discri_input_var, condition_var, vocoder, ctxsize, hiddensize=256, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, nbcnnlayers=8, nbfilters=16, spec_freqlen=5, noise_freqlen=5, ctxlayers_nb=1, postlayers_nb=6, windur=0.025, bn_axes=None, use_LSweighting=True, LSWGANtransflc=0.5, LSWGANtransc=1.0/8.0, dropout_p=-1.0, use_bn=False):
        if bn_axes is None: bn_axes=[0,1]
        layer_discri = lasagne.layers.InputLayer(shape=(None, None, vocoder.featuressize()), input_var=discri_input_var, name='input')

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
        layer = lasagne.layers.SliceLayer(layer_discri, indices=slice(vocoder.f0size(),vocoder.f0size()+vocoder.specsize()), axis=2, name='spec_slice') # Assumed feature order

        if use_LSweighting: # Using weighted WGAN+LS
            print('WGAN Weighted LS - Discri - SPEC')
            wganls_spec_weights_ = nonlin_sigmoidparm(np.arange(vocoder.specsize(), dtype=theano.config.floatX),  int(LSWGANtransflc*vocoder.specsize()), LSWGANtransc)
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

        if vocoder.noisesize()>0: # Add noise in discriminator
            layer = lasagne.layers.SliceLayer(layer_discri, indices=slice(vocoder.f0size()+vocoder.specsize(),vocoder.f0size()+vocoder.specsize()+vocoder.noisesize()), axis=2, name='nm_slice')

            if use_LSweighting: # Using weighted WGAN+LS
                print('WGAN Weighted LS - Discri - NM')
                wganls_spec_weights_ = nonlin_sigmoidparm(np.arange(vocoder.noisesize(), dtype=theano.config.floatX),  int(LSWGANtransflc*vocoder.noisesize()), LSWGANtransc)
                wganls_weights = theano.shared(value=np.asarray(wganls_spec_weights_), name='wganls_spec_weights_')
                layer = CstMulLayer(layer, cstW=wganls_weights, name='cstdot_wganls_weights')

            layer = lasagne.layers.dimshuffle(layer, [0, 'x', 1, 2], name='nm_dimshuffle')
            for layi in xrange(nbcnnlayers):
                layerstr = 'nm_l'+str(1+layi)+'_GC{}x{}x{}'.format(nbfilters,_winlen,noise_freqlen)
                layer = layer_GatedConv2DLayer(layer, nbfilters, [_winlen,noise_freqlen], pad='same', nonlinearity=nonlinearity, name=layerstr)
                if use_bn: layer=lasagne.layers.batch_norm(layer)
                if dropout_p>0.0: layer=lasagne.layers.dropout(layer, p=dropout_p)
            layer = lasagne.layers.dimshuffle(layer, [0, 2, 3, 1], name='nm_dimshuffle')
            layer_bndnm = lasagne.layers.flatten(layer, outdim=3, name='nm_flatten')
            layerstoconcats.append(layer_bndnm)

        # Add the contexts
        layer_ctx_input = lasagne.layers.InputLayer(shape=(None, None, ctxsize), input_var=condition_var, name='ctx_input')
        layer_ctx = layer_ctx_input
        for layi in xrange(ctxlayers_nb):
            layerstr = 'ctx_l'+str(1+layi)+'_FC{}'.format(hiddensize)
            layer_ctx = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(layer_ctx, hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr), axes=bn_axes)
        grad_clipping = 50
        for layi in xrange(ctxlayers_nb):
            layerstr = 'ctx_l'+str(1+layi)+'_BLSTM{}'.format(hiddensize)
            fwd = lasagne.layers.LSTMLayer(layer_ctx, num_units=hiddensize, backwards=False, name=layerstr+'.fwd', grad_clipping=grad_clipping)
            bck = lasagne.layers.LSTMLayer(layer_ctx, num_units=hiddensize, backwards=True, name=layerstr+'.bck', grad_clipping=grad_clipping)
            # layer_ctx = lasagne.layers.ConcatLayer((fwd, bck), axis=2) # It seems concat of concats doesn't work
            if layi==ctxlayers_nb-1:
                layerstoconcats.append(fwd)
                layerstoconcats.append(bck)
            else:
                layer_ctx = lasagne.layers.ConcatLayer((fwd, bck), axis=2)

        # Concatenate the features analysis with the contexts...
        layer = lasagne.layers.ConcatLayer(layerstoconcats, axis=2, name='ctx_features_concat')

        # ... and finalize with a common FC network
        for layi in xrange(postlayers_nb):
            layerstr = 'post_l'+str(1+layi)+'_FC'+str(hiddensize)
            layer = lasagne.layers.DenseLayer(layer, hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr)
            if use_bn: layer=lasagne.layers.batch_norm(layer, axes=_bn_axes)
            # if dropout_p>0.0: layer = lasagne.layers.dropout(layer, p=dropout_p) # Bad for FC

        # output layer (linear)
        layer = lasagne.layers.DenseLayer(layer, 1, nonlinearity=None, num_leading_axes=2, name='projection') # No nonlin for this output
        return [layer, layer_discri, layer_ctx_input]
