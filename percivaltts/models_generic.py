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

import vocoders

import model
import models_basic
from models_cnn import CstMulLayer
from models_cnn import TileLayer
from models_cnn import TimeWeightLayer
from models_cnn import layer_GatedConv2DLayer
from models_cnn import layer_context

class ModelGeneric(model.Model):
    def __init__(self, insize, vocoder, mlpg_wins=[], layertypes=['FC', 'FC', 'BLSTM'], hiddensize=256, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, bn_axes=None, grad_clipping=50, nameprefix=None):
        if bn_axes is None: bn_axes=[0,1]
        model.Model.__init__(self, insize, vocoder, hiddensize)

        if nameprefix is None: nameprefix=''

        l_hid = lasagne.layers.InputLayer(shape=(None, None, insize), input_var=self._input_values, name=nameprefix+'input.conditional')

        for layi in xrange(len(layertypes)):
            layerstr = nameprefix+'l'+str(1+layi)+'_{}{}'.format(layertypes[layi], hiddensize)

            if layertypes[layi]=='FC':
                l_hid = lasagne.layers.DenseLayer(l_hid, num_units=hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr)

                if len(bn_axes)>0: l_hid=lasagne.layers.batch_norm(l_hid, axes=bn_axes) # Add batch normalisation

            elif layertypes[layi]=='BLSTM':
                fwd = models_basic.layer_LSTM(l_hid, hiddensize, nonlinearity=nonlinearity, backwards=False, grad_clipping=grad_clipping, name=layerstr+'.fwd')
                bck = models_basic.layer_LSTM(l_hid, hiddensize, nonlinearity=nonlinearity, backwards=True, grad_clipping=grad_clipping, name=layerstr+'.bck')
                l_hid = lasagne.layers.ConcatLayer((fwd, bck), axis=2)

                # Don't add batch norm for RNN-based layers

            elif isinstance(layertypes[layi], list):
                if layertypes[layi][0]=='CNN':
                    # l_hid = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(l_hid, hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name='projection'), axes=bn_axes)
                    l_hid = lasagne.layers.dimshuffle(l_hid, [0, 'x', 1, 2], name=nameprefix+'dimshuffle')
                    nbfilters = layertypes[layi][1]
                    winlen = layertypes[layi][2]
                    layerstr = nameprefix+'l'+str(1+layi)+'_CNN{}x{}x{}'.format(nbfilters,winlen,1)
                    l_hid = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(l_hid, num_filters=nbfilters, filter_size=[winlen,1], stride=1, pad='same', nonlinearity=nonlinearity, name=layerstr))
                    # l_hid = lasagne.layers.batch_norm(layer_GatedConv2DLayer(l_hid, nbfilters, [winlen,1], stride=1, pad='same', nonlinearity=nonlinearity, name=layerstr))
                    # if dropout_p>0.0: l_hid = lasagne.layers.dropout(l_hid, p=dropout_p)
                    # l_hid = lasagne.layers.Conv2DLayer(l_hid, 1, [_winlen,spec_freqlen], pad='same', nonlinearity=None, name='spec_lout_2DC')
                    l_hid = lasagne.layers.dimshuffle(l_hid, [0, 2, 3, 1], name='dimshuffle_back')
                    l_hid = lasagne.layers.flatten(l_hid, outdim=3, name='flatten')

            # Add dropout (after batchnorm)
            if dropout_p>0.0: l_hid=lasagne.layers.dropout(l_hid, p=dropout_p)

        l_out = models_basic.layer_final(l_hid, vocoder, mlpg_wins)

        self.init_finish(l_out) # Has to be called at the end of the __init__ to print out the architecture, get the trainable params, etc.


    # WGAN: Discriminant arch. parameters
    #       These are usually symmetrical with the model, but it the model can be very different in the case of a generic model, so make D as in CNN model.
    _nbcnnlayers = 8
    _nbfilters = 16
    _spec_freqlen = 5
    _noise_freqlen = 5
    _windur = 0.025

    def build_discri(self, discri_input_var, condition_var, vocoder, ctxsize, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, postlayers_nb=6, bn_axes=None, use_LSweighting=True, LSWGANtransflc=0.5, LSWGANtransc=1.0/8.0, use_WGAN_incnoise=True, use_bn=False):
        if bn_axes is None: bn_axes=[0,1]
        layer_discri = ll.InputLayer(shape=(None, None, vocoder.featuressize()), input_var=discri_input_var, name='input')

        winlen = int(0.5*self._windur/0.005)*2+1

        layerstoconcats = []

        # Amplitude spectrum
        layer = ll.SliceLayer(layer_discri, indices=slice(vocoder.f0size(),vocoder.f0size()+vocoder.specsize()), axis=2, name='spec_slice') # Assumed feature order

        if use_LSweighting: # Using weighted WGAN+LS
            print('WGAN Weighted LS - Discri - SPEC')
            wganls_spec_weights_ = nonlin_sigmoidparm(np.arange(vocoder.specsize(), dtype=theano.config.floatX),  int(LSWGANtransflc*vocoder.specsize()), LSWGANtransc)
            wganls_weights = theano.shared(value=np.asarray(wganls_spec_weights_), name='wganls_spec_weights_')
            layer = CstMulLayer(layer, cstW=wganls_weights, name='cstdot_wganls_weights')

        layer = ll.dimshuffle(layer, [0, 'x', 1, 2], name='spec_dimshuffle')
        for layi in xrange(self._nbcnnlayers):
            layerstr = 'spec_l'+str(1+layi)+'_GC{}x{}x{}'.format(self._nbfilters,winlen,self._spec_freqlen)
            # strides>1 make the first two Conv layers pyramidal. Increase patches' effects here and there, bad.
            layer = layer_GatedConv2DLayer(layer, self._nbfilters, [winlen,self._spec_freqlen], pad='same', nonlinearity=nonlinearity, name=layerstr)
            if use_bn: layer=ll.batch_norm(layer)
        layer = ll.dimshuffle(layer, [0, 2, 3, 1], name='spec_dimshuffle')
        layer_spec = ll.flatten(layer, outdim=3, name='spec_flatten')
        layerstoconcats.append(layer_spec)

        if use_WGAN_incnoise and vocoder.noisesize()>0: # Add noise in discriminator
            layer = ll.SliceLayer(layer_discri, indices=slice(vocoder.f0size()+vocoder.specsize(),vocoder.f0size()+vocoder.specsize()+vocoder.noisesize()), axis=2, name='nm_slice')

            if use_LSweighting: # Using weighted WGAN+LS
                print('WGAN Weighted LS - Discri - NM')
                wganls_spec_weights_ = nonlin_sigmoidparm(np.arange(vocoder.noisesize(), dtype=theano.config.floatX),  int(LSWGANtransflc*vocoder.noisesize()), LSWGANtransc)
                wganls_weights = theano.shared(value=np.asarray(wganls_spec_weights_), name='wganls_spec_weights_')
                layer = CstMulLayer(layer, cstW=wganls_weights, name='cstdot_wganls_weights')

            layer = ll.dimshuffle(layer, [0, 'x', 1, 2], name='nm_dimshuffle')
            for layi in xrange(np.max((1,int(np.ceil(self._nbcnnlayers/2))))):
                layerstr = 'nm_l'+str(1+layi)+'_GC{}x{}x{}'.format(self._nbfilters,winlen,self._noise_freqlen)
                layer = layer_GatedConv2DLayer(layer, self._nbfilters, [winlen,self._noise_freqlen], pad='same', nonlinearity=nonlinearity, name=layerstr)
                if use_bn: layer=ll.batch_norm(layer)
            layer = ll.dimshuffle(layer, [0, 2, 3, 1], name='nm_dimshuffle')
            layer_bndnm = ll.flatten(layer, outdim=3, name='nm_flatten')
            layerstoconcats.append(layer_bndnm)

        # Add the contexts
        layer_ctx_input = ll.InputLayer(shape=(None, None, ctxsize), input_var=condition_var, name='ctx_input')

        layer_ctx = layer_context(layer_ctx_input, ctx_nblayers=self._ctx_nblayers, ctx_nbfilters=self._ctx_nbfilters, ctx_winlen=self._ctx_winlen, hiddensize=self._hiddensize, nonlinearity=nonlinearity, bn_axes=bn_axes)
        layerstoconcats.append(layer_ctx)

        # Concatenate the features analysis with the contexts...
        layer = ll.ConcatLayer(layerstoconcats, axis=2, name='ctx_features_concat')

        # ... and finalize with a common FC network
        for layi in xrange(postlayers_nb):
            layerstr = 'post_l'+str(1+layi)+'_FC'+str(self._hiddensize)
            layer = ll.DenseLayer(layer, self._hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr)
            if use_bn: layer=ll.batch_norm(layer, axes=_bn_axes)

        # output layer (linear)
        layer = ll.DenseLayer(layer, 1, nonlinearity=None, num_leading_axes=2, name='projection') # No nonlin for this output
        return [layer, layer_discri, layer_ctx_input]
