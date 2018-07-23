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

from percivaltts import *  # Always include this first to setup a few things

import numpy as np
numpy_force_random_seed()

import theano
import theano.tensor as T
import lasagne
import lasagne.layers as ll
# lasagne.random.set_rng(np.random)

from external.pulsemodel import sigproc as sp

from backend_theano import *
# TODO Should have this import here or before to ensure repeatability?
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import model
import vocoders

import models_basic

class CstMulLayer(ll.Layer):
    def __init__(self, incoming, cstW, **kwargs):
        super(CstMulLayer, self).__init__(incoming, **kwargs)
        self.cstW = cstW

    def get_output_for(self, x, **kwargs):
        return x*self.cstW

    # def get_output_shape_for(self, input_shape):
    #     return (input_shape[0], self.num_units)

class TileLayer(ll.Layer):
    def __init__(self, incoming, reps, **kwargs):
        super(TileLayer, self).__init__(incoming, **kwargs)
        self.reps = reps

    def get_output_for(self, x, **kwargs):
        x = T.tile(x, self.reps)
        return x

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2]*self.reps[2])


def layer_GatedConv2DLayer(incoming, num_filters, filter_size, stride=(1, 1), pad=0, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, name=''):
    la = ll.Conv2DLayer(incoming, num_filters=num_filters, filter_size=filter_size, stride=stride, pad=pad, nonlinearity=nonlinearity, name=name+'.activation')
    lg = ll.Conv2DLayer(incoming, num_filters=num_filters, filter_size=filter_size, stride=stride, pad=pad, nonlinearity=theano.tensor.nnet.nnet.sigmoid, name=name+'.gate')
    lout = ll.ElemwiseMergeLayer([la, lg], T.mul, cropping=None, name=name+'.mul_merge')
    return lout

def layer_context(layer_ctx, ctx_nblayers, ctx_nbfilters, ctx_winlen, hiddensize, nonlinearity, bn_axes=None, bn_cnn_axes=None, critic=False, useLRN=True):

    layer_ctx = ll.dimshuffle(layer_ctx, [0, 'x', 1, 2], name='ctx.dimshuffle_to_2DCNN')
    for layi in xrange(ctx_nblayers):
        layerstr = 'ctx.l'+str(1+layi)+'_CNN{}x{}x{}'.format(ctx_nbfilters,ctx_winlen,1)
        layer_ctx = ll.Conv2DLayer(layer_ctx, num_filters=ctx_nbfilters, filter_size=[ctx_winlen,1], stride=1, pad='same', nonlinearity=nonlinearity, name=layerstr)
        if not critic and (not bn_cnn_axes is None): layer_ctx=ll.batch_norm(layer_ctx, axes=bn_cnn_axes)
        # layer_ctx = ll.batch_norm(layer_GatedConv2DLayer(layer_ctx, ctx_nbfilters, [ctx_winlen,1], stride=1, pad='same', nonlinearity=nonlinearity, name=layerstr))
        if critic and useLRN: layer_ctx=ll.LocalResponseNormalization2DLayer(layer_ctx)
    layer_ctx = ll.dimshuffle(layer_ctx, [0, 2, 3, 1], name='ctx.dimshuffle_back')
    layer_ctx = ll.flatten(layer_ctx, outdim=3, name='ctx.flatten')

    for layi in xrange(2):
        layerstr = 'ctx.l'+str(1+ctx_nblayers+layi)+'_FC{}'.format(hiddensize)
        layer_ctx = ll.DenseLayer(layer_ctx, hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr)
        if  not critic and (not bn_axes is None): layer_ctx=ll.batch_norm(layer_ctx, axes=bn_axes)

    return layer_ctx


class UniformNoiseLayer(ll.Layer):
    """Uniform noise layer.
    """
    _size = 100
    _low  = 0.0
    _high = 1.0

    def __init__(self, incoming, size, low=0.0, high=1.0, **kwargs):
        super(UniformNoiseLayer, self).__init__(incoming, **kwargs)
        self._srng = RandomStreams(np.random.randint(1, 2147462579))
        self._size = size
        self._low = low
        self._high = high

    def get_output_for(self, input, deterministic=False, **kwargs):
        """
        Parameters
        ----------
        input : tensor
            output from the previous layer
        deterministic : bool
            If true noise is disabled, see notes
        """
        shape = tuple([input.shape[0], input.shape[1], self._size])
        unirnd = self._srng.uniform(shape, low=self._low, high=self._high)
        return unirnd

    def get_output_shape_for(self, input_shape):
        return [input_shape[0], input_shape[1], self._size]


class ModelCNN(model.Model):

    def __init__(self, insize, vocoder, hiddensize=256, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, ctx_nblayers=1, ctx_nbfilters=2, ctx_winlen=21, nbcnnlayers=8, nbfilters=16, spec_freqlen=5, noise_freqlen=5, windur=0.025, bn_axes=None, noisesize=100):
        if bn_axes is None: bn_axes=[0,1]
        model.Model.__init__(self, insize, vocoder, hiddensize)

        self._ctx_nblayers = ctx_nblayers
        self._ctx_nbfilters = ctx_nbfilters
        self._ctx_winlen = ctx_winlen

        self._nbcnnlayers = nbcnnlayers
        self._nbfilters = nbfilters
        self._spec_freqlen = spec_freqlen
        self._noise_freqlen = noise_freqlen
        self._windur = windur

        winlen = int(0.5*self._windur/0.005)*2+1

        layer_ctx_input = ll.InputLayer(shape=(None, None, insize), input_var=self._input_values, name='ctx.input')

        layer_noise_input = UniformNoiseLayer(layer_ctx_input, noisesize, name='noise.input')
        layer_ctx_input = ll.ConcatLayer((layer_ctx_input, layer_noise_input), axis=2, name='concat.input') # TODO Put the noise later on

        self._layer_ctx = layer_context(layer_ctx_input, ctx_nblayers=self._ctx_nblayers, ctx_nbfilters=self._ctx_nbfilters, ctx_winlen=self._ctx_winlen, hiddensize=self._hiddensize, nonlinearity=nonlinearity, bn_axes=[0,1], bn_cnn_axes=[0,2,3])

        layers_toconcat = []

        if vocoder.f0size()>0:
            # F0 - BLSTM layer
            layer_f0 = self._layer_ctx
            grad_clipping = 50
            for layi in xrange(1):
                layerstr = 'f0_l'+str(1+layi)+'_BLSTM{}'.format(self._hiddensize)
                fwd = models_basic.layer_LSTM(layer_f0, self._hiddensize, nonlinearity, backwards=False, grad_clipping=grad_clipping, name=layerstr+'.fwd')
                bck = models_basic.layer_LSTM(layer_f0, self._hiddensize, nonlinearity, backwards=True, grad_clipping=grad_clipping, name=layerstr+'.bck')
                layer_f0 = ll.ConcatLayer((fwd, bck), axis=2, name=layerstr+'.concat')
                # TODO Replace by CNN ?? It didn't work well, maybe didn't work well with WGAN loss, but f0 is not more on WGAN loss
            layer_f0 = ll.DenseLayer(layer_f0, num_units=vocoder.f0size(), nonlinearity=None, num_leading_axes=2, name='f0_lout_projection')
            layers_toconcat.append(layer_f0)

        if vocoder.specsize()>0:
            # Amplitude spectrum - 2D Gated Conv layers
            layer_spec_proj = ll.batch_norm(ll.DenseLayer(self._layer_ctx, vocoder.specsize(), nonlinearity=nonlinearity, num_leading_axes=2, name='spec_projection'), axes=bn_axes)
            # layer_spec_proj = ll.DenseLayer(self._layer_ctx, vocoder.specsize(), nonlinearity=None, num_leading_axes=2, name='spec_projection')
            layer_spec = ll.dimshuffle(layer_spec_proj, [0, 'x', 1, 2], name='spec_dimshuffle')
            for layi in xrange(nbcnnlayers):
                layerstr = 'spec_l'+str(1+layi)+'_GC{}x{}x{}'.format(self._nbfilters,winlen,self._spec_freqlen)
                layer_spec = ll.batch_norm(layer_GatedConv2DLayer(layer_spec, self._nbfilters, [winlen,self._spec_freqlen], stride=1, pad='same', nonlinearity=nonlinearity, name=layerstr))
            layer_spec = ll.Conv2DLayer(layer_spec, 1, [winlen,self._spec_freqlen], pad='same', nonlinearity=None, name='spec_lout_2DC')
            layer_spec = ll.dimshuffle(layer_spec, [0, 2, 3, 1], name='spec_dimshuffle')
            layer_spec = ll.flatten(layer_spec, outdim=3, name='spec_flatten')
            # layer_spec = ll.ElemwiseSumLayer([layer_spec, layer_spec_proj], name='skip')
            layers_toconcat.append(layer_spec)

        if vocoder.noisesize()>0:
            layer_noise = self._layer_ctx
            for layi in xrange(np.max((1,int(np.ceil(nbcnnlayers/2))))):
                layerstr = 'noise_l'+str(1+layi)+'_FC{}'.format(hiddensize)
                layer_noise = ll.DenseLayer(layer_noise, num_units=hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr)
            if isinstance(vocoder, vocoders.VocoderPML):
                layer_noise = ll.DenseLayer(layer_noise, num_units=vocoder.nm_size, nonlinearity=lasagne.nonlinearities.sigmoid, num_leading_axes=2, name='lo_noise') # sig is best among nonlin_saturatedsigmoid nonlin_tanh_saturated nonlin_tanh_bysigmoid
            else:
                layer_noise = ll.DenseLayer(layer_noise, num_units=vocoder.nm_size, nonlinearity=None, num_leading_axes=2, name='lo_noise')
            layers_toconcat.append(layer_noise)

        if vocoder.vuvsize()>0:
            # VUV - BLSTM layer
            layer_vuv = self._layer_ctx
            grad_clipping = 50
            for layi in xrange(1):
                layerstr = 'vuv_l'+str(1+layi)+'_BLSTM{}'.format(self._hiddensize)
                fwd = models_basic.layer_LSTM(layer_vuv, self._hiddensize, nonlinearity, backwards=False, grad_clipping=grad_clipping, name=layerstr+'.fwd')
                bck = models_basic.layer_LSTM(layer_vuv, self._hiddensize, nonlinearity, backwards=True, grad_clipping=grad_clipping, name=layerstr+'.bck')
                layer_vuv = ll.ConcatLayer((fwd, bck), axis=2, name=layerstr+'.concat')
            layer_vuv = ll.DenseLayer(layer_vuv, num_units=vocoder.vuvsize(), nonlinearity=None, num_leading_axes=2, name='vuv_lout_projection')
            layers_toconcat.append(layer_vuv)

        layer = ll.ConcatLayer(layers_toconcat, axis=2, name='lout.concat')

        self.init_finish(layer) # Has to be called at the end of the __init__ to print out the architecture, get the trainable params, etc.


    def build_critic(self, critic_input_var, condition_var, vocoder, ctxsize, nonlinearity=lasagne.nonlinearities.very_leaky_rectify, postlayers_nb=6, use_LSweighting=True, LSWGANtransfreqcutoff=4000, LSWGANtranscoef=1.0/8.0, use_WGAN_incnoisefeature=False):

        useLRN = False # TODO

        layer_critic = ll.InputLayer(shape=(None, None, vocoder.featuressize()), input_var=critic_input_var, name='input')

        winlen = int(0.5*self._windur/0.005)*2+1

        layerstoconcats = []

        # Amplitude spectrum
        layer = ll.SliceLayer(layer_critic, indices=slice(vocoder.f0size(),vocoder.f0size()+vocoder.specsize()), axis=2, name='spec_slice') # Assumed feature order

        if use_LSweighting: # Using weighted WGAN+LS
            print('WGAN Weighted LS - critic - SPEC (trans cutoff {}Hz)'.format(LSWGANtransfreqcutoff))
            # wganls_spec_weights_ = nonlin_sigmoidparm(np.arange(vocoder.specsize(), dtype=theano.config.floatX),  int(LSWGANtransfreqcutoff*vocoder.specsize()), LSWGANtranscoef)
            wganls_spec_weights_ = nonlin_sigmoidparm(np.arange(vocoder.specsize(), dtype=theano.config.floatX), sp.freq2fwspecidx(LSWGANtransfreqcutoff, vocoder.fs, vocoder.specsize()), LSWGANtranscoef)
            wganls_weights = theano.shared(value=np.asarray(wganls_spec_weights_), name='wganls_spec_weights_')
            layer = CstMulLayer(layer, cstW=wganls_weights, name='cstdot_wganls_weights')

        layer = ll.dimshuffle(layer, [0, 'x', 1, 2], name='spec_dimshuffle')
        for layi in xrange(self._nbcnnlayers):
            layerstr = 'spec_l'+str(1+layi)+'_GC{}x{}x{}'.format(self._nbfilters,winlen,self._spec_freqlen)
            # strides>1 make the first two Conv layers pyramidal. Increase patches' effects here and there, bad.
            layer = layer_GatedConv2DLayer(layer, self._nbfilters, [winlen,self._spec_freqlen], pad='same', nonlinearity=nonlinearity, name=layerstr)
            if useLRN: layer = ll.LocalResponseNormalization2DLayer(layer)
        layer = ll.dimshuffle(layer, [0, 2, 3, 1], name='spec_dimshuffle')
        layer_spec = ll.flatten(layer, outdim=3, name='spec_flatten')
        layerstoconcats.append(layer_spec)

        if use_WGAN_incnoisefeature and vocoder.noisesize()>0: # Add noise in critic
            layer = ll.SliceLayer(layer_critic, indices=slice(vocoder.f0size()+vocoder.specsize(),vocoder.f0size()+vocoder.specsize()+vocoder.noisesize()), axis=2, name='nm_slice')

            if use_LSweighting: # Using weighted WGAN+LS
                print('WGAN Weighted LS - critic - NM (trans cutoff {}Hz)'.format(LSWGANtransfreqcutoff))
                # wganls_spec_weights_ = nonlin_sigmoidparm(np.arange(vocoder.noisesize(), dtype=theano.config.floatX),  int(LSWGANtransfreqcutoff*vocoder.noisesize()), LSWGANtranscoef)
                wganls_spec_weights_ = nonlin_sigmoidparm(np.arange(vocoder.noisesize(), dtype=theano.config.floatX),  sp.freq2fwspecidx(LSWGANtransfreqcutoff, vocoder.fs, vocoder.noisesize()), LSWGANtranscoef)
                wganls_weights = theano.shared(value=np.asarray(wganls_spec_weights_), name='wganls_spec_weights_')
                layer = CstMulLayer(layer, cstW=wganls_weights, name='cstdot_wganls_weights')

            layer = ll.dimshuffle(layer, [0, 'x', 1, 2], name='nm_dimshuffle')
            for layi in xrange(np.max((1,int(np.ceil(self._nbcnnlayers/2))))):
                layerstr = 'nm_l'+str(1+layi)+'_GC{}x{}x{}'.format(self._nbfilters,winlen,self._noise_freqlen)
                layer = layer_GatedConv2DLayer(layer, self._nbfilters, [winlen,self._noise_freqlen], pad='same', nonlinearity=nonlinearity, name=layerstr)
                if useLRN: layer = ll.LocalResponseNormalization2DLayer(layer)
            layer = ll.dimshuffle(layer, [0, 2, 3, 1], name='nm_dimshuffle')
            layer_bndnm = ll.flatten(layer, outdim=3, name='nm_flatten')
            layerstoconcats.append(layer_bndnm)

        # Add the contexts
        layer_ctx_input = ll.InputLayer(shape=(None, None, ctxsize), input_var=condition_var, name='ctx_input')
        layer_ctx = layer_context(layer_ctx_input, ctx_nblayers=self._ctx_nblayers, ctx_nbfilters=self._ctx_nbfilters, ctx_winlen=self._ctx_winlen, hiddensize=self._hiddensize, nonlinearity=nonlinearity, bn_axes=None, bn_cnn_axes=None, critic=True, useLRN=useLRN)
        layerstoconcats.append(layer_ctx)

        # Concatenate the features analysis with the contexts...
        layer = ll.ConcatLayer(layerstoconcats, axis=2, name='ctx_features.concat')

        # ... and finalize with a common FC network
        for layi in xrange(postlayers_nb):
            layerstr = 'post.l'+str(1+layi)+'_FC'+str(self._hiddensize)
            layer = ll.DenseLayer(layer, self._hiddensize, nonlinearity=nonlinearity, num_leading_axes=2, name=layerstr)

        # output layer (linear)
        layer = ll.DenseLayer(layer, 1, nonlinearity=None, num_leading_axes=2, name='projection') # No nonlin for this output
        return [layer, layer_critic, layer_ctx_input]
