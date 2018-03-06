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

import sys
import os
import copy
import time

import random
import cPickle
import functools
from collections import defaultdict

import numpy as np

from utils import *
import data

print('\nLoading Theano')
from utils_theano import *
print_sysinfo_theano()
import theano
import theano.tensor as T
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/external/Lasagne/')
import lasagne

import model

import models_cnn # For GAN discriminator

class Optimizer:

    _model = None # The model whose parameters will be optimised.

    _errtype = 'WGAN' # None for LSE

    _target_values = None
    _params_trainable = None
    _optim_updates = []  # The variables of the optimisation algo, for restoring a training that crashed

    def __init__(self, model, errtype='WGAN'):
        self._model = model

        self._errtype = errtype

        self._target_values = T.ftensor3('target_values')

    def saveTrainingState(self, fstate, cfg=None, extras=dict(), printfn=print):
        # https://github.com/Lasagne/Lasagne/issues/159
        printfn('    saving training state in {} ...'.format(fstate), end='')
        sys.stdout.flush()

        paramsvalues = [(str(p), p.get_value()) for p in self._model.params_all] # The network parameters

        ovs = []
        for ov in self._optim_updates:
            ovs.append([p.get_value() for p in ov.keys()]) # The optim algo state

        DATA = [paramsvalues, ovs, cfg, extras]
        cPickle.dump(DATA, open(fstate, 'wb'))

        print(' done')
        sys.stdout.flush()

    def loadTrainingState(self, fstate, cfg, printfn=print):
        # https://github.com/Lasagne/Lasagne/issues/159
        printfn('    reloading parameters from {} ...'.format(fstate), end='')
        sys.stdout.flush()

        DATA = cPickle.load(open(fstate, 'rb'))
        for p, v in zip(self._model.params_all, DATA[0]): p.set_value(v[1])    # The network parameters

        for ov, da in zip(self._optim_updates, DATA[1]):
            for p, value in zip(ov.keys(), da): p.set_value(value)      # The optim algo state

        print(' done')
        sys.stdout.flush()

        if cfg.__dict__!=DATA[2].__dict__:
            printfn('        configurations are not the same !')
            for attr in cfg.__dict__:
                if attr in DATA[2].__dict__:
                    print('            attribute {}: new state {}, saved state {}'.format(attr, cfg.__dict__[attr], DATA[2].__dict__[attr]))
                else:
                    print('            attribute {}:{} is not in the saved state'.format(attr, cfg.__dict__[attr]))
            for attr in DATA[2].__dict__:
                if attr not in cfg.__dict__:
                    print('            attribute {}:{} is not in the new state'.format(attr, cfg.__dict__[attr]))

        return DATA[2:]

    # Training =================================================================

    def train(self, params, indir, outdir, wdir, fid_lst_tra, fid_lst_val, X_vals, Y_vals, cfg, params_savefile, trialstr='', cont=None):

        print('Model initial status before training')
        worst_val = data.cost_0pred_rmse(Y_vals) # RMSE
        print("    0-pred validation RMSE = {} (100%)".format(worst_val))
        init_pred_std = data.prediction_std(self._model, [X_vals])
        print('    initial std of prediction = {}'.format(init_pred_std))
        init_val = data.cost_model_prediction_rmse(self._model, [X_vals], Y_vals)
        best_val = init_val # Among all trials of hyper-parameters optimisation
        print("    initial validation RMSE = {} ({:.4f}%)".format(init_val, 100.0*init_val/worst_val))

        nbbatches = int(len(fid_lst_tra)/cfg.train_batchsize)
        print('    using {} batches of {} sentences each'.format(nbbatches, cfg.train_batchsize))
        print('    model #parameters={}'.format(self._model.nbParams()))
        print('    #parameters/#frame={:.2f}'.format(float(self._model.nbParams())/(nbbatches*cfg.train_batchsize)))

        if self._errtype=='WGAN':
            print('Preparing discriminator for WGAN...')
            discri_input_var = T.tensor3('discri_input') # Either real data to predict/generate, or, fake data that has been generated
            # TODO Might drop discri_input_var and replace it with self._target_values
            [discri, layer_discri, layer_cond] = models_cnn.ModelCNN_build_discri(discri_input_var, self._model._input_values, self._model.specsize, self._model.nmsize, self._model.insize, hiddensize=self._model._hiddensize, nbcnnlayers=self._model._nbcnnlayers, nbfilters=self._model._nbfilters, spec_freqlen=self._model._spec_freqlen, nm_freqlen=self._model._nm_freqlen, nbpostlayers=self._model._nbprelayers, windur=self._model._windur)

            print('    Discriminator architecture')
            for l in lasagne.layers.get_all_layers(discri):
                print('        {}:{}'.format(l.name, l.output_shape))

            # Create expression for passing real data through the discri
            real_out = lasagne.layers.get_output(discri)
            # Create expression for passing fake data through the discri
            genout = lasagne.layers.get_output(self._model.net_out)
            indict = {layer_discri:lasagne.layers.get_output(self._model.net_out), layer_cond:self._model._input_values}
            fake_out = lasagne.layers.get_output(discri, indict)

            # Create generator's loss expression
            if cfg.train_LScoef>0.0:    # TODO This might be 0 but low freq still needs LS
                if 0: # Use Standard WGAN+LS (no special weighting curve)
                    print('Overall additive LS solution')
                    generator_loss = -(1.0-cfg.train_LScoef)*fake_out.mean() + cfg.train_LScoef*lasagne.objectives.squared_error(genout, self._target_values).mean()
                else:
                    print('WGAN Weighted LS - Generator part')
                    specxs = np.arange(self._model.specsize, dtype=theano.config.floatX)
                    nmxs = np.arange(self._model.nmsize, dtype=theano.config.floatX)
                    wganls_weights_ = np.hstack(([0.0], nonlin_sigmoidparm(specxs,  int(self._model.specsize/2), 1.0/8.0), nonlin_sigmoidparm(nmxs,  int(self._model.nmsize/2), 1.0/8.0)))
                    wganls_weights_ *= (1.0-cfg.train_LScoef)

                    wganls_weights_gan = theano.shared(value=wganls_weights_, name='wganls_weights_gan')

                    lserr = lasagne.objectives.squared_error(genout, self._target_values)
                    wganls_weights_ls = theano.shared(value=(1.0-wganls_weights_), name='wganls_weights_ls')

                    generator_loss = -(fake_out*wganls_weights_gan).mean() + (lserr*wganls_weights_ls).mean() # TODO A term in [-oo,oo] and one in [0,oo] .. ?

            else:
                # Standard WGAN, no special mixing with LSE
                generator_loss = -fake_out.mean()

            discri_loss = fake_out.mean() - real_out.mean()

            # Improved training for Wasserstein GAN
            epsi = T.TensorType(dtype=theano.config.floatX,broadcastable=(False, True, True))()
            mixed_X = (epsi * genout) + (1-epsi) * discri_input_var
            indict = {layer_discri:mixed_X, layer_cond:self._model._input_values}
            output_D_mixed = lasagne.layers.get_output(discri, inputs=indict)
            grad_mixed = T.grad(T.sum(output_D_mixed), mixed_X)
            norm_grad_mixed = T.sqrt(T.sum(T.square(grad_mixed),axis=[1,2]))
            grad_penalty = T.mean(T.square(norm_grad_mixed -1))
            discri_loss = discri_loss + cfg.train_pg_lambda*grad_penalty

            # Create update expressions for training
            generator_params = lasagne.layers.get_all_params(self._model.net_out, trainable=True)
            discri_params = lasagne.layers.get_all_params(discri, trainable=True)
            generator_updates = lasagne.updates.adam(generator_loss, generator_params, learning_rate=cfg.train_G_learningrate, beta1=cfg.train_G_adam_beta1, beta2=cfg.train_G_adam_beta2)
            discri_updates = lasagne.updates.adam(discri_loss, discri_params, learning_rate=cfg.train_D_learningrate, beta1=cfg.train_D_adam_beta1, beta2=cfg.train_D_adam_beta2)
            self._optim_updates.extend([generator_updates, discri_updates])

            # Compile functions performing a training step on a mini-batch (according
            # to the updates dictionary) and returning the corresponding score:
            print('Compiling generator training function...')
            generator_train_fn_ins = [self._model._input_values]
            if cfg.train_LScoef>0.0: generator_train_fn_ins.append(self._target_values)
            train_fn = theano.function(generator_train_fn_ins, generator_loss, updates=generator_updates)
            train_validation_fn = theano.function(generator_train_fn_ins, generator_loss, no_default_updates=True)
            print('Compiling discriminator training function...')
            discri_train_fn_ins = [self._model._input_values]
            discri_train_fn_ins.extend([discri_input_var, epsi])
            discri_train_fn = theano.function(discri_train_fn_ins, discri_loss, updates=discri_updates)
            discri_train_validation_fn = theano.function(discri_train_fn_ins, discri_loss, no_default_updates=True)

        else:
            print('    LSE Training')
            predicttrain_values = lasagne.layers.get_output(self._model.net_out, deterministic=False)
            costout = (predicttrain_values - self._target_values)**2

            self.cost = T.mean(costout) # self.cost = T.mean(T.sum(costout, axis=-1)) ?

            print("    creating parameters updates ...")
            updates = lasagne.updates.adam(self.cost, params, learning_rate=10**cfg.train_learningrate_log10, beta1=cfg.train_adam_beta1, beta2=cfg.train_adam_beta2, epsilon=10**cfg.train_adam_epsilon_log10)
            self._optim_updates.append(updates)
            print("    compiling training function ...")
            train_fn = theano.function(self._model.inputs+[self._target_values], self.cost, updates=updates)

        costs = defaultdict(list)
        epochs_modelssaved = []
        epochs_durs = []
        nbnodecepochs = 0
        generator_updates = 0
        epochstart = 1
        if cont:
            print('    reloading previous training state ...')
            _, extras = self.loadTrainingState(params_savefile+'.trainingstate.last', cfg)
            cost_val = extras['cost_val']
            best_val = extras['best_val']
            # Restoring some local variables
            costs = extras['costs']
            epochs_modelssaved = extras['epochs_modelssaved']
            epochs_durs = extras['epochs_durs']
            nbnodecepochs = extras['nbnodecepochs']
            generator_updates = extras['generator_updates']
            epochstart = extras['epoch']+1

        print_log("    start training ...")
        rndidx = np.arange(len(fid_lst_tra))
        for epoch in range(epochstart,1+cfg.train_max_nbepochs):
            timeepochstart = time.time()
            random.shuffle(rndidx)
            rndidxb = np.split(rndidx, nbbatches)
            costs_tra_batches = []
            costs_tra_discri_batches = []
            load_times = []
            train_times = []
            for k in xrange(nbbatches):

                timeloadstart = time.time()
                print_tty('\r    Training batch {}/{}'.format(1+k, nbbatches))

                # Load training data online, because data is often too heavy to hold in memory
                fid_lst_trab = [fid_lst_tra[bidx] for bidx in rndidxb[k]]
                X_trab, MX_trab, Y_trab, MY_trab = data.load_inoutset(indir, outdir, wdir, fid_lst_trab, length=cfg.train_batch_length, lengthmax=cfg.train_batch_lengthmax, maskpadtype=cfg.train_batch_padtype)

                if 0: # Plot batch
                    import matplotlib.pyplot as plt
                    plt.ion()
                    plt.imshow(Y_trab[0,].T, origin='lower', aspect='auto', interpolation='none', cmap='jet')
                    from IPython.core.debugger import  Pdb; Pdb().set_trace()

                load_times.append(time.time()-timeloadstart)
                print_tty('({:.6f}s); training '.format(load_times[-1]))

                timetrainstart = time.time()
                if self._errtype=='WGAN':
                    # TODO Tune this
                    if (generator_updates < 25) or (generator_updates % 500 == 0):
                        discri_runs = 10
                    else:
                        discri_runs = 5

                    random_epsilon = np.random.uniform(size=(cfg.train_batchsize, 1,1)).astype('float32')

                    discri_returns = discri_train_fn(X_trab, Y_trab, random_epsilon)        # Train the discrimnator
                    costs_tra_discri_batches.append(float(discri_returns))

                    if k%discri_runs==0:
                        # Train the generator
                        trainargs = [X_trab]
                        if cfg.train_LScoef>0.0: trainargs.append(Y_trab)
                        cost_tra = train_fn(*trainargs)
                        cost_tra = float(cost_tra)
                        generator_updates += 1

                else:
                    # LSE
                    train_returns = train_fn(X_trab, Y_trab)
                    cost_tra = np.sqrt(float(train_returns))

                train_times.append(time.time()-timetrainstart)

                print_tty('err={:.4f} ({:.4f}s)                  '.format(cost_tra,train_times[-1]))
                if np.isnan(cost_tra):                      # pragma: no cover
                    print_log('    previous costs: {}'.format(costs_tra_batches))
                    print_log('    epoch {} Batch {}/{} train cost = {}'.format(epoch, 1+k, nbbatches, cost_tra))
                    raise ValueError('ERROR: Training cost is nan!')
                costs_tra_batches.append(cost_tra)
            print_tty('\r                                                           \r')
            costs['model_training'].append(np.mean(costs_tra_batches))

            # Eval validation cost
            cost_validation_rmse = data.cost_model_prediction_rmse(self._model, [X_vals], Y_vals)
            costs['model_rmse_validation'].append(cost_validation_rmse)

            if self._errtype=='WGAN':
                train_validation_fn_args = [X_vals]
                if cfg.train_LScoef>0.0: train_validation_fn_args.append(Y_vals)
                costs['model_validation'].append(data.cost_model(train_validation_fn, train_validation_fn_args))
                costs['discri_training'].append(np.mean(costs_tra_discri_batches))
                random_epsilon = [np.random.uniform(size=(1,1)).astype('float32')]*len(X_vals)
                discri_train_validation_fn_args = [X_vals]
                discri_train_validation_fn_args.extend([Y_vals, random_epsilon])
                costs['discri_validation'].append(data.cost_model(discri_train_validation_fn, discri_train_validation_fn_args))

            cost_val = costs['model_rmse_validation'][-1]

            print_log("    epoch {} {}  cost_tra={:.6f} (load:{}s train:{}s)  cost_val={:.6f} ({:.4f}%)  {} MiB GPU {} MiB RAM".format(epoch, trialstr, costs['model_training'][-1], time2str(np.sum(load_times)), time2str(np.sum(train_times)), cost_val, 100*cost_val/worst_val, nvidia_smi_gpu_memused(), proc_memresident()))
            sys.stdout.flush()

            if np.isnan(cost_val): raise ValueError('ERROR: Validation cost is nan!')
            if cost_val>=cfg.train_cancel_validthresh*worst_val: raise ValueError('ERROR: Validation cost blew up! It is higher than {} times the worst possible values'.format(cfg.train_cancel_validthresh))

            self._model.saveAllParams(params_savefile+'.last', cfg=cfg, printfn=print_log, extras={'cost_val':cost_val})

            # Save model parameters
            if cost_val<best_val: # Among all trials of hyper-parameter optimisation
                self._model.saveAllParams(params_savefile, cfg=cfg, printfn=print_log, extras={'cost_val':cost_val})
                epochs_modelssaved.append(epoch)
                best_val = cost_val

            if cfg.train_log_plot:
                print_log('    saving plots')
                log_plot_costs(costs, worst_val, fname=params_savefile+'.fig_costs_'+trialstr+'.svg', epochs_modelssaved=epochs_modelssaved)

                nbsamples = 2
                nbsamples = min(nbsamples, len(X_vals))
                Y_preds = []
                for sampli in xrange(nbsamples): Y_preds.append(self._model.predict(np.reshape(X_vals[sampli],[1]+[s for s in X_vals[sampli].shape]))[0,])

                plotsuffix = ''
                if len(epochs_modelssaved)>0 and epochs_modelssaved[-1]==epoch: plotsuffix='_best'
                else:                                                           plotsuffix='_last'
                log_plot_samples(Y_vals, Y_preds, nbsamples=nbsamples, shift=0.005, fname=params_savefile+'.fig_samples_'+trialstr+plotsuffix+'.png', title='epoch={}'.format(epoch), specsize=self._model.specsize)

            if len(costs['model_rmse_validation'])<2 or costs['model_rmse_validation'][-1]<min(costs['model_rmse_validation'][:-1]):
                nbnodecepochs = 0
            elif epoch>cfg.train_cancel_nodecepochs:        # pragma: no cover
                nbnodecepochs += 1
                if nbnodecepochs>=cfg.train_cancel_nodecepochs:
                    print_log('WARNING: validation error did not decrease for {} epochs. Early stop!'.format(cfg.train_cancel_nodecepochs))
                    break

            epochs_durs.append(time.time()-timeepochstart)
            print_log('    epoch time: {}   max tot train ~time: {}s   train ~time left {}'.format(time2str(epochs_durs[-1]), time2str(np.median(epochs_durs)*cfg.train_max_nbepochs), time2str(np.median(epochs_durs)*(cfg.train_max_nbepochs-epoch))))

            self.saveTrainingState(params_savefile+'.trainingstate.last', cfg=cfg, printfn=print_log, extras={'cost_val':cost_val, 'best_val':best_val, 'costs':costs, 'epochs_modelssaved':epochs_modelssaved, 'epochs_durs':epochs_durs, 'nbnodecepochs':nbnodecepochs, 'generator_updates':generator_updates, 'epoch':epoch})


        return {'epoch_stopped':epoch, 'worst_val':worst_val, 'best_epoch':epochs_modelssaved[-1] if len(epochs_modelssaved)>0 else -1, 'best_val':best_val, 'best_val_percent':100*best_val/worst_val}


    def randomize_hyper(self, cfg):
        cfg = copy.copy(cfg) # Create a new one instead of updating the object passed as argument

        # Randomized the hyper parameters
        if len(cfg.train_hypers)<1: return cfg, ''

        hyperstr = ''
        for hyper in cfg.train_hypers:
            if type(hyper[1]) is int and type(hyper[2]) is int:
                setattr(cfg, hyper[0], np.random.randint(hyper[1],hyper[2]))
            else:
                setattr(cfg, hyper[0], np.random.uniform(hyper[1],hyper[2]))
            hyperstr += hyper[0]+'='+str(getattr(cfg, hyper[0]))+','
        hyperstr = hyperstr[:-1]

        return cfg, hyperstr

    def train_multipletrials(self, indir, outdir, wdir, fid_lst_tra, fid_lst_val, params, params_savefile, cfgtomerge=None, cont=None, **kwargs):
        # Hyp: always uses batches

        # All kwargs arguments are specific configuration values
        # First, fill a struct with the default configuration values ...
        cfg = configuration() # Init structure

        # LSE
        cfg.train_learningrate_log10 = -3.39794       # (10**-3.39794=0.0004 confirmed on 2xBGRU256_bn20) [potential hyper-parameter] Merlin:0.001 (or 0.002, or 0.004)
        cfg.train_adam_beta1 = 0.98           # [potential hyper-parameter]
        cfg.train_adam_beta2 = 0.999          # [potential hyper-parameter]
        cfg.train_adam_epsilon_log10 = -8     # [potential hyper-parameter]
        # WGAN
        cfg.train_D_learningrate = 0.0001     # [potential hyper-parameter]
        cfg.train_D_adam_beta1 = 0.0          # [potential hyper-parameter]
        cfg.train_D_adam_beta2 = 0.9          # [potential hyper-parameter]
        cfg.train_G_learningrate = 0.001      # [potential hyper-parameter]
        cfg.train_G_adam_beta1 = 0.0          # [potential hyper-parameter]
        cfg.train_G_adam_beta2 = 0.9          # [potential hyper-parameter]
        cfg.train_pg_lambda = 10              # [potential hyper-parameter]
        cfg.train_LScoef = 0.25               # [potential hyper-parameter]

        cfg.train_max_nbepochs = 100
        cfg.train_batchsize = 5               # [potential hyper-parameter]
        cfg.train_batch_padtype = 'randshift' # See load_inoutset(..., maskpadtype)
        cfg.train_batch_length = None # Duration [frames] of each batch (def. None, i.e. the shortest duration of the batch if using maskpadtype = 'randshift')
        cfg.train_batch_lengthmax = None # Maximum duration [frames] of each batch
        cfg.train_cancel_validthresh = 10.0 # Cancel train if valid err is more than N times higher than the 0-pred valid err
        cfg.train_cancel_nodecepochs = 50
        cfg.train_nbtrials = 1
        cfg.train_hypers=[]
        #cfg.hypers = [('learningrate_log10', -6.0, -2.0), ('adam_beta1', 0.8, 1.0)] # For ADAM
        ##cfg.train_hyper = [('train_learningrate', 0.0001, 0.1), ('train_adam_beta1', 0.8, 1.0), ('train_adam_beta2', 0.995, 1.0), ('train_adam_epsilon_log10', -10.0, -6.0), ('train_batchsize', 1, 200)] # For ADAM
        cfg.train_log_plot=True
        # ... add/overwrite configuration from cfgtomerge ...
        if not cfgtomerge is None: cfg.merge(cfgtomerge)
        # ... and add/overwrite specific configuration from the generic arguments
        for kwarg in kwargs.keys(): setattr(cfg, kwarg, kwargs[kwarg])

        print('Training configuration')
        cfg.print_content()

        print('Loading all validation data at once ...')
        # from IPython.core.debugger import  Pdb; Pdb().set_trace()
        # X_val, Y_val = data.load_inoutset(indir, outdir, wdir, fid_lst_val, verbose=1)
        X_vals = data.load(indir, fid_lst_val, verbose=1, label='Context labels: ')
        Y_vals = data.load(outdir, fid_lst_val, verbose=1, label='Output features: ')
        X_vals, Y_vals = data.cropsize([X_vals, Y_vals])

        if cfg.train_nbtrials>1:
            self._model.saveAllParams(params_savefile+'.init', cfg=cfg, printfn=print_log)
            # self.saveTrainingState(params_savefile+'.trainingstate-init', cfg=cfg, printfn=print_log)

        try:
            trials = []
            for triali in xrange(1,1+cfg.train_nbtrials):  # Run multiple trials with different hyper-parameters
                print('\nStart trial {} ...'.format(triali))

                try:
                    trialstr = 'trial'+str(triali)
                    if len(cfg.train_hypers)>0:
                        cfg, hyperstr = self.randomize_hyper(cfg)
                        trialstr += ','+hyperstr
                        print('    randomized hyper-parameters: '+trialstr)
                    if cfg.train_nbtrials>1:
                        self._model.loadAllParams(params_savefile+'.init')
                        # self.loadTrainingState(params_savefile+'.trainingstate-init')

                    timewholetrainstart = time.time()
                    train_rets = self.train(params, indir, outdir, wdir, fid_lst_tra, fid_lst_val, X_vals, Y_vals, cfg, params_savefile, trialstr=trialstr, cont=cont)
                    cont = None
                    print_log('Total trial run time: {}s'.format(time2str(time.time()-timewholetrainstart)))

                except KeyboardInterrupt:                   # pragma: no cover
                    raise KeyboardInterrupt
                except:                                     # pragma: no cover
                    if len(cfg.train_hypers)>0: print_log('WARNING: Training crashed!')
                    else:                       print_log('ERROR: Training crashed!')
                    import traceback
                    traceback.print_exc()
                    pass

                if cfg.train_nbtrials>1:
                    trials.append([triali]+[getattr(cfg, field[0]) for field in cfg.train_hypers]+[train_rets[key] for key in sorted(train_rets.keys())])
                    # Save results of each trial
                    np.savetxt(params_savefile+'.trials', np.vstack(trials), header=('trials '+' '.join([field[0] for field in cfg.train_hypers]+sorted(train_rets.keys()))))

        except KeyboardInterrupt:                           # pragma: no cover
            print_log('WARNING: Training interrupted by user!')
            pass

        print_log('Finished')
