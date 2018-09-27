'''
The base model class to derive the others from.

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

from functools import partial

# import cPickle
# from collections import defaultdict

from backend_tensorflow import *
from tensorflow import keras
import tensorflow.keras.backend as K

from external.pulsemodel import sigproc as sp

import data

import optimizertts

class RandomWeightedAverage(keras.layers.merge._Merge):
    batchsize = None
    def __init__(self, batchsize):
        keras.layers.merge._Merge.__init__(self)
        self.batchsize = batchsize
    def _merge_function(self, inputs):
        weights = keras.backend.random_uniform((tf.shape(inputs[0])[0], 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def wasserstein_loss(valid_true, valid_pred):
    return K.mean(valid_true * valid_pred)

def specweighted_lse_loss(y_true, y_pred, specweight):

    lsepart = (y_true - y_pred)**2

    lsepart = lsepart*specweight

    return K.mean(lsepart)


class OptimizerTTSWGAN(optimizertts.OptimizerTTS):

    costs_tra_critic_batches = []
    generator_updates = 0

    def __init__(self, cfg, model, errtype='WGAN', critic=None, **kwargs):
        optimizertts.OptimizerTTS.__init__(self, cfg, model, errtype, *kwargs)
        self.critic = critic


    def prepare(self):

        print('    Prepare {} training...'.format(self._errtype))      

        generator = self._model._kerasmodel

        # Construct Computational Graph for Critic

        critic = keras.Model(inputs=[self.critic.input_features, self.critic.input_ctx], outputs=self.critic.output)
        print('    critic architecture:')
        critic.summary()

        # Create a frozen generator for the critic training
        # Use the Network class to avoid irrelevant warning: https://github.com/keras-team/keras/issues/8585
        frozen_generator = keras.engine.network.Network(self._model._kerasmodel.inputs, self._model._kerasmodel.outputs)
        frozen_generator.trainable = False

        # Input for a real sample
        real_sample = keras.layers.Input(shape=(None,self._model.vocoder.featuressize()))
        # Generate an artifical (fake) sample
        fake_sample = frozen_generator(self.critic.input_ctx)

        # Discriminator determines validity of the real and fake images
        fake = critic([fake_sample, self.critic.input_ctx])
        valid = critic([real_sample, self.critic.input_ctx])

        # Construct weighted average between real and fake images
        interpolated_sample = RandomWeightedAverage(self.cfg.train_batch_size)([real_sample, fake_sample])
        # Determine validity of weighted sample
        validity_interpolated = critic([interpolated_sample, self.critic.input_ctx])

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=interpolated_sample)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        print('    compiling critic')
        critic_opti = keras.optimizers.Adam(lr=float(10**self.cfg.train_D_learningrate_log10), beta_1=float(self.cfg.train_D_adam_beta1), beta_2=float(self.cfg.train_D_adam_beta2), epsilon=K.epsilon(), decay=0.0, amsgrad=False)
        print('        optimizer: {}'.format(type(critic_opti).__name__))
        self.critic_model = keras.Model(inputs=[real_sample, self.critic.input_ctx],
                                    outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss],
                                    optimizer=critic_opti, loss_weights=[1, 1, self.cfg.train_pg_lambda])

        self.wgan_valid = -np.ones((self.cfg.train_batch_size, 1, 1))
        self.wgan_fake =  np.ones((self.cfg.train_batch_size, 1, 1))
        self.wgan_dummy = np.zeros((self.cfg.train_batch_size, 1, 1)) # Dummy gt for gradient penalty


        # Construct Computational Graph for Generator

        # Create a frozen critic for the generator training
        frozen_critic = keras.engine.network.Network([self.critic.input_features,self.critic.input_ctx], self.critic.output)
        frozen_critic.trainable = False

        # Sampled noise for input to generator
        ctx_gen = keras.layers.Input(shape=(None, self._model.ctxsize))
        # Generate images based of noise
        pred_sample = generator(ctx_gen)
        # Discriminator determines validity
        valid = frozen_critic([pred_sample,ctx_gen])
        # Defines generator model
        print('    compiling generator')
        gen_opti = keras.optimizers.Adam(lr=float(10**self.cfg.train_G_learningrate_log10), beta_1=float(self.cfg.train_G_adam_beta1), beta_2=float(self.cfg.train_G_adam_beta2), epsilon=K.epsilon(), decay=0.0, amsgrad=False)
        print('        optimizer: {}'.format(type(gen_opti).__name__))

        if self._errtype=='WGAN':
            print('        use WGAN optimization')
            self.generator_model = keras.Model(inputs=ctx_gen, outputs=valid)
            self.generator_model.compile(loss=wasserstein_loss, optimizer=gen_opti)

            def generator_train_validation_fn(x, _valid):
                return self.generator_model.evaluate(x=x, y=_valid, batch_size=1, verbose=0)

        elif self._errtype=='WLSWGAN':
            print('        use WLSWGAN optimization')
            # First pack the outputs, since there is no possibility of doing many(outputs)-to-one(loss) in Keras ... :_(
            self.generator_model = keras.Model(inputs=ctx_gen, outputs=[valid,pred_sample])

            # TODO TODO TODO Clean this crap
            # TODO TODO TODO Clean this crap
            # TODO TODO TODO Clean this crap
            wganls_weights_els = []
            wganls_weights_els.append([0.0]) # For f0
            specvs = np.arange(self._model.vocoder.specsize(), dtype=np.float32)
            if self.cfg.train_LScoef==0.0:
                wganls_weights_els.append(np.ones(self._model.vocoder.specsize()))  # No special weighting for spec
            else:
                wganls_weights_els.append(nonlin_sigmoidparm(specvs,  sp.freq2fwspecidx(self.cfg.train_critic_LSWGANtransfreqcutoff, self._model.vocoder.fs, self._model.vocoder.specsize()), self.cfg.train_critic_LSWGANtranscoef)) # For spec
            if self._model.vocoder.noisesize()>0:
                if self.cfg.train_critic_use_WGAN_incnoisefeature:
                    noisevs = np.arange(self._model.vocoder.noisesize(), dtype=np.float32)
                    wganls_weights_els.append(nonlin_sigmoidparm(noisevs,  sp.freq2fwspecidx(self.cfg.train_critic_LSWGANtransfreqcutoff, self._model.vocoder.fs, self._model.vocoder.noisesize()), self.cfg.train_critic_LSWGANtranscoef)) # For noise
                else:
                    wganls_weights_els.append(np.zeros(self._model.vocoder.noisesize()))
            if self._model.vocoder.vuvsize()>0:
                wganls_weights_els.append([0.0]) # For vuv
            wganls_weights_ = np.hstack(wganls_weights_els)

            wganls_weights_ *= (1.0-self.cfg.train_LScoef)

            wganls_weights_ls = (1.0-wganls_weights_)
            # TODO TODO TODO Clean this crap
            # TODO TODO TODO Clean this crap
            # TODO TODO TODO Clean this crap

            self.generator_model.compile(loss=[wasserstein_loss, partial(specweighted_lse_loss,specweight=wganls_weights_ls)], optimizer=gen_opti, loss_weights=[np.mean(wganls_weights_), 1])


    def train_on_batch(self, batchid, X_trab, Y_trab):

        cost_tra = None

        critic_returns = self.critic_model.train_on_batch([Y_trab, X_trab], [self.wgan_valid, self.wgan_fake, self.wgan_dummy])
        self.costs_tra_critic_batches.append(float(critic_returns[0]))

        # TODO The params below are supposed to ensure the critic is "almost" fully converged
        #      when training the generator. How to evaluate this? Is it the case currently?
        if (self.generator_updates < 25) or (self.generator_updates % 500 == 0):  # TODO Params hardcoded TODO TODO TODO Try to get rid of it
            critic_runs = 10 # TODO Params hardcoded 10
        else:
            critic_runs = 5 # TODO Params hardcoded 5
        # martinarjovsky: "- Loss of the critic should never be negative, since outputing 0 would yeald a better loss so this is a huge red flag."
        # if critic_returns>0 and k%critic_runs==0: # Train only if the estimate of the Wasserstein distance makes sense, and, each N critic iteration TODO Doesn't work well though
        if batchid%critic_runs==0: # Train each N critic iteration
            # Train the generator
            if self._errtype=='WGAN':
                cost_tra = self.generator_model.train_on_batch(X_trab, self.wgan_valid)
            elif self._errtype=='WLSWGAN':
                cost_tra = self.generator_model.train_on_batch(X_trab, [self.wgan_valid, Y_trab])[0]
            self.generator_updates += 1

            if 0: log_plot_samples(Y_vals, Y_preds, nbsamples=nbsamples, fname=os.path.splitext(params_savefile)[0]+'-fig_samples_'+trialstr+'{:07}.png'.format(self.generator_updates), vocoder=self._model.vocoder, title='E{} I{}'.format(epoch,self.generator_updates))

        return cost_tra


    def update_validation_cost(self, costs, X_vals, Y_vals):
        cost_validation_rmse = data.cost_model_prediction_rmse(self._model, [X_vals], Y_vals)
        costs['model_rmse_validation'].append(cost_validation_rmse)

        # TODO The following often breaks when loss functions, etc. Try to find a design which is more prototype-friendly
        if self._errtype=='WGAN':       generator_train_validation_fn_args = [X_vals, Y_vals]
        elif self._errtype=='WLSWGAN':  generator_train_validation_fn_args = [X_vals, [self.wgan_valid for _ in xrange(len(Y_vals))], Y_vals]
        costs['model_validation'].append(data.cost_model_mfn(lambda x, valid, y: self.generator_model.evaluate(x=x, y=[valid,y], batch_size=1, verbose=0)[0], generator_train_validation_fn_args))
        costs['critic_training'].append(np.mean(self.costs_tra_critic_batches))
        critic_train_validation_fn_args = [Y_vals, X_vals]
        costs['critic_validation'].append(data.cost_model_mfn(lambda y,x: self.critic_model.evaluate(x=[y, x], y=[self.wgan_valid[:1,], self.wgan_fake[:1,], self.wgan_dummy[:1,]], batch_size=1, verbose=0)[0], critic_train_validation_fn_args))
        costs['critic_validation_ltm'].append(np.mean(costs['critic_validation'][-self.cfg.train_validation_ltm_winlen:]))
        cost_val = costs['critic_validation_ltm'][-1]

        return cost_val
