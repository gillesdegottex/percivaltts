'''
An optimizer class that print and plot information during training that is
mainly dedicated to TTS.

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

import sys
import os
import copy
import time
import glob
from functools import partial

import cPickle
from collections import defaultdict

import numpy as np
numpy_force_random_seed()

from backend_tensorflow import *
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

from external.pulsemodel import sigproc as sp

import data

import networktts

# if tf_cuda_available():
#     from pygpu.gpuarray import GpuArrayException   # pragma: no cover
# else:
#     class GpuArrayException(Exception): pass       # declare a dummy one if pygpu is not loaded


def lse_loss(y_true, y_pred):   # i.e. mse_loss
    return K.mean((y_true - y_pred)**2)


class OptimizerTTS:

    _model = None # The model whose parameters will be optimised.
    _errtype = 'LSE' # or 'LSE'

    def __init__(self, cfgtomerge, model, errtype='LSE', **kwargs):
        self._model = model
        self._errtype = errtype

        # All kwargs arguments are specific configuration values
        # First, fill a struct with the default configuration values ...
        cfg = configuration() # Init structure

        # All the default parameters for the optimizer
        # Common options to any optimisation schemes
        cfg.train_min_nbepochs = 200
        cfg.train_max_nbepochs = 300
        cfg.train_nbepochs_scalewdata = True
        cfg.train_cancel_nodecepochs = 50
        cfg.train_cancel_validthresh = 10.0     # Cancel train if valid err is more than N times higher than the initial worst valid err
        cfg.train_batch_size = 5                # [potential hyper-parameter]
        cfg.train_batch_padtype = 'randshift'   # See load_inoutset(..., maskpadtype)
        cfg.train_batch_cropmode = 'begendbigger'     # 'begend', 'begendbigger', 'all'
        cfg.train_batch_length = None           # Duration [frames] of each batch (def. None, i.e. the shortest duration of the batch if using maskpadtype = 'randshift') # TODO Remove for lengthmax
        cfg.train_batch_lengthmax = None        # Maximum duration [frames] of each batch
        cfg.train_nbtrials = 1                  # Just run one training only
        cfg.train_hypers=[]

        cfg = self.default_options(cfg)

        #cfg.train_hypers = [('train_lse_learningrate_log10', -6.0, -2.0), ('adam_beta1', 0.8, 1.0)] # For ADAM
        #cfg.train_hyper = [('train_wgan_critic_learningrate', 0.0001, 0.1), ('train_wgan_critic_adam_beta1', 0.8, 1.0), ('train_wgan_critic_adam_beta2', 0.995, 1.0), ('train_batch_size', 1, 200)] # For ADAM
        cfg.train_log_plot=True
        # ... add/overwrite configuration from cfgtomerge ...
        if not cfgtomerge is None: cfg.merge(cfgtomerge)
        # ... and add/overwrite specific configuration from the generic arguments
        for kwarg in kwargs.keys(): setattr(cfg, kwarg, kwargs[kwarg])

        self.cfg = cfg

        print('Training configuration')
        self.cfg.print_content()

    def saveTrainingState(self, fstate, extras=None, printfn=print):
        # TODO TODO TODO FIX
        if extras is None: extras=dict()
        printfn('    saving training state in {} ...'.format(fstate), end='')
        sys.stdout.flush()

        self.saveTrainingStateLossSpecific(fstate)

        # Save the extra data
        DATA = [self.cfg, extras, np.random.get_state()]
        cPickle.dump(DATA, open(fstate+'.model.cfgextras.pkl', 'wb'))

        print(' done')
        sys.stdout.flush()

    def loadTrainingState(self, fstate, printfn=print):
        # TODO TODO TODO FIX
        printfn('    reloading parameters from {} ...'.format(fstate), end='')
        sys.stdout.flush()

        self.loadTrainingStateLossSpecific(fstate)

        # Load the extra data
        DATA = cPickle.load(open(fstate+'.model.cfgextras.pkl', 'rb'))

        print(' done')
        sys.stdout.flush()

        cfg_restored = DATA[0]

        if self.cfg.__dict__!=cfg_restored.__dict__:
            printfn('        configurations are not the same!')
            for attr in self.cfg.__dict__:
                if attr in cfg_restored.__dict__:
                    if self.cfg.__dict__[attr]!=cfg_restored.__dict__[attr]:
                        print('            attribute {}: new state {}, saved state {}'.format(attr, self.cfg.__dict__[attr], cfg_restored.__dict__[attr]))
                else:
                    print('            attribute {}: is not in the saved configuration state'.format(attr))
            for attr in cfg_restored.__dict__:
                if attr not in self.cfg.__dict__:
                    print('            attribute {}: is not in the new configuration state'.format(attr))

        return DATA

    # Training =================================================================

    def train_oneparamset(self, indir, outdir, wdir, fid_lst_tra, fid_lst_val, params_savefile, trialstr='', cont=None):

        print('Loading all validation data at once ...')
        # X_val, Y_val = data.load_inoutset(indir, outdir, wdir, fid_lst_val, verbose=1)
        X_vals = data.load(indir, fid_lst_val, verbose=1, label='Context labels: ')
        Y_vals = data.load(outdir, fid_lst_val, verbose=1, label='Output features: ')
        X_vals, Y_vals = data.croplen([X_vals, Y_vals])
        print('    {} validation files'.format(len(fid_lst_val)))
        print('    number of validation files / train files: {:.2f}%'.format(100.0*float(len(fid_lst_val))/len(fid_lst_tra)))

        print('Model initial status before training')
        worst_val = data.cost_0pred_rmse(Y_vals)
        print("    0-pred validation RMSE = {} (100%)".format(worst_val))
        init_pred_rms = data.prediction_rms(self._model, [X_vals])
        print('    initial RMS of prediction = {}'.format(init_pred_rms))
        init_val = data.cost_model_prediction_rmse(self._model, [X_vals], Y_vals)
        best_val = None
        print("    initial validation RMSE = {} ({:.4f}%)".format(init_val, 100.0*init_val/worst_val))

        nbbatches = int(len(fid_lst_tra)/self.cfg.train_batch_size)
        print('    using {} batches of {} sentences each'.format(nbbatches, self.cfg.train_batch_size))
        print('    model #parameters={}'.format(self._model.count_params()))

        nbtrainframes = 0
        for fid in fid_lst_tra:
            X = data.loadfile(outdir, fid)
            nbtrainframes += X.shape[0]
        print('    Training set: {} sentences, #frames={} ({})'.format(len(fid_lst_tra), nbtrainframes, time.strftime('%H:%M:%S', time.gmtime((nbtrainframes*self._model.vocoder.shift)))))
        print('    #parameters/#frames={:.2f}'.format(float(self._model.count_params())/nbtrainframes))
        if self.cfg.train_nbepochs_scalewdata and not self.cfg.train_batch_lengthmax is None:
            # During an epoch, the whole data is _not_ seen by the training since cfg.train_batch_lengthmax is limited and smaller to the sentence size.
            # To compensate for it and make the config below less depedent on the data, the min ans max nbepochs are scaled according to the missing number of frames seen.
            # TODO Should consider only non-silent frames, many recordings have a lot of pre and post silences
            epochcoef = nbtrainframes/float((self.cfg.train_batch_lengthmax*len(fid_lst_tra)))
            print('    scale number of epochs wrt number of frames')
            self.cfg.train_min_nbepochs = int(self.cfg.train_min_nbepochs*epochcoef)
            self.cfg.train_max_nbepochs = int(self.cfg.train_max_nbepochs*epochcoef)
            print('        train_min_nbepochs={}'.format(self.cfg.train_min_nbepochs))
            print('        train_max_nbepochs={}'.format(self.cfg.train_max_nbepochs))

        self.prepare()  # This has to be overwritten by sub-classes

        costs = defaultdict(list)
        epochs_modelssaved = []
        epochs_durs = []
        nbnodecepochs = 0
        generator_updates = 0
        epochstart = 1
        if cont and len(glob.glob(os.path.splitext(params_savefile)[0]+'-trainingstate-last.h5*'))>0:
            print('    reloading previous training state ...')
            savedcfg, extras, rngstate = self.loadTrainingState(os.path.splitext(params_savefile)[0]+'-trainingstate-last.h5')
            np.random.set_state(rngstate)
            cost_val = extras['cost_val']
            # Restoring some local variables
            costs = extras['costs']
            epochs_modelssaved = extras['epochs_modelssaved']
            epochs_durs = extras['epochs_durs']
            generator_updates = extras['generator_updates']
            epochstart = extras['epoch']+1
            # Restore the saving criteria if only none of those 3 cfg values changed:
            if (savedcfg.train_min_nbepochs==self.cfg.train_min_nbepochs) and (savedcfg.train_max_nbepochs==self.cfg.train_max_nbepochs) and (savedcfg.train_cancel_nodecepochs==self.cfg.train_cancel_nodecepochs):
                best_val = extras['best_val']
                nbnodecepochs = extras['nbnodecepochs']

        print_log("    start training ...")
        for epoch in range(epochstart,1+self.cfg.train_max_nbepochs):
            timeepochstart = time.time()
            rndidx = np.arange(int(nbbatches*self.cfg.train_batch_size))    # Need to restart from ordered state to make the shuffling repeatable after reloading training state, the shuffling will be different anyway
            np.random.shuffle(rndidx)
            rndidxb = np.split(rndidx, nbbatches)
            cost_tra = None
            costs_tra_batches = []
            costs_tra_gen_wgan_lse_ratios = []
            load_times = []
            train_times = []
            for batchid in xrange(nbbatches):

                timeloadstart = time.time()
                print_tty('\r    Training batch {}/{}'.format(1+batchid, nbbatches))

                # Load training data online, because data is often too heavy to hold in memory
                fid_lst_trab = [fid_lst_tra[bidx] for bidx in rndidxb[batchid]]
                X_trab, Y_trab, W_trab = data.load_inoutset(indir, outdir, wdir, fid_lst_trab, length=self.cfg.train_batch_length, lengthmax=self.cfg.train_batch_lengthmax, maskpadtype=self.cfg.train_batch_padtype, cropmode=self.cfg.train_batch_cropmode)

                if 0: # Plot batch
                    import matplotlib.pyplot as plt
                    plt.ion()
                    plt.imshow(Y_trab[0,].T, origin='lower', aspect='auto', interpolation='none', cmap='jet')
                    from IPython.core.debugger import  Pdb; Pdb().set_trace()

                load_times.append(time.time()-timeloadstart)
                print_tty(' (iter load: {:.6f}s); training '.format(load_times[-1]))

                timetrainstart = time.time()

                cost_tra = self.train_on_batch(batchid, X_trab, Y_trab)  # This has to be overwritten by sub-classes

                train_times.append(time.time()-timetrainstart)

                if not cost_tra is None:
                    print_tty('err={:.4f} (iter train: {:.4f}s)                  '.format(cost_tra,train_times[-1]))
                    if np.isnan(cost_tra):                      # pragma: no cover
                        print_log('    previous costs: {}'.format(costs_tra_batches))
                        print_log('    E{} Batch {}/{} train cost = {}'.format(epoch, 1+batchid, nbbatches, cost_tra))
                        raise ValueError('ERROR: Training cost is nan!')
                    costs_tra_batches.append(cost_tra)
            print_tty('\r                                                           \r')
            costs['model_training'].append(np.mean(costs_tra_batches))

            cost_val = self.update_validation_cost(costs, X_vals, Y_vals)  # This has to be overwritten by sub-classes

            print_log("    E{}/{} {}  cost_tra={:.6f} (load:{}s train:{}s)  cost_val={:.6f} ({:.4f}% RMSE)  {} MiB GPU {} MiB RAM".format(epoch, self.cfg.train_max_nbepochs, trialstr, costs['model_training'][-1], time2str(np.sum(load_times)), time2str(np.sum(train_times)), cost_val, 100*costs['model_rmse_validation'][-1]/worst_val, tf_gpu_memused(), proc_memresident()))
            sys.stdout.flush()

            if np.isnan(cost_val): raise ValueError('ERROR: Validation cost is nan!')
            # if (self._errtype=='LSE') and (cost_val>=self.cfg.train_cancel_validthresh*worst_val): raise ValueError('ERROR: Validation cost blew up! It is higher than {} times the worst possible values'.format(self.cfg.train_cancel_validthresh)) # TODO

            self._model.save(os.path.splitext(params_savefile)[0]+'-last.h5', printfn=print_log, extras={'cost_val':cost_val})

            # Save model parameters
            if epoch>=self.cfg.train_min_nbepochs: # Assume no model is good enough before self.cfg.train_min_nbepochs
                if ((best_val is None) or (cost_val<best_val)): # Among all trials of hyper-parameter optimisation
                    best_val = cost_val
                    self._model.save(params_savefile, printfn=print_log, extras={'cost_val':cost_val}, infostr='(E{} C{:.4f})'.format(epoch, best_val))
                    epochs_modelssaved.append(epoch)
                    nbnodecepochs = 0
                else:
                    nbnodecepochs += 1

            if self.cfg.train_log_plot:
                print_log('    saving plots')
                log_plot_costs(costs, worst_val, fname=os.path.splitext(params_savefile)[0]+'-fig_costs_'+trialstr+'.svg', epochs_modelssaved=epochs_modelssaved)

                nbsamples = 2
                nbsamples = min(nbsamples, len(X_vals))
                Y_preds = []
                for sampli in xrange(nbsamples): Y_preds.append(self._model.predict(np.reshape(X_vals[sampli],[1]+[s for s in X_vals[sampli].shape]))[0,])

                plotsuffix = ''
                if len(epochs_modelssaved)>0 and epochs_modelssaved[-1]==epoch: plotsuffix='_best'
                else:                                                           plotsuffix='_last'
                log_plot_samples(Y_vals, Y_preds, nbsamples=nbsamples, fname=os.path.splitext(params_savefile)[0]+'-fig_samples_'+trialstr+plotsuffix+'.png', vocoder=self._model.vocoder, title='E{}'.format(epoch))

            epochs_durs.append(time.time()-timeepochstart)
            print_log('    ET: {}   max TT: {}s   train ~time left: {}'.format(time2str(epochs_durs[-1]), time2str(np.median(epochs_durs[-10:])*self.cfg.train_max_nbepochs), time2str(np.median(epochs_durs[-10:])*(self.cfg.train_max_nbepochs-epoch))))

            self.saveTrainingState(os.path.splitext(params_savefile)[0]+'-trainingstate-last.h5', printfn=print_log, extras={'cost_val':cost_val, 'best_val':best_val, 'costs':costs, 'epochs_modelssaved':epochs_modelssaved, 'epochs_durs':epochs_durs, 'nbnodecepochs':nbnodecepochs, 'generator_updates':generator_updates, 'epoch':epoch})

            if nbnodecepochs>=self.cfg.train_cancel_nodecepochs: # pragma: no cover
                print_log('WARNING: validation error did not decrease for {} epochs. Early stop!'.format(self.cfg.train_cancel_nodecepochs))
                break

        if best_val is None: raise ValueError('No model has been saved during training!')
        return {'epoch_stopped':epoch, 'worst_val':worst_val, 'best_epoch':epochs_modelssaved[-1] if len(epochs_modelssaved)>0 else -1, 'best_val':best_val}


    @classmethod
    def randomize_hyper(cls, cfg):
        cfg = copy.copy(cfg) # Create a new one instead of updating the object passed as argument

        # Randomized the hyper parameters
        if len(cfg.train_hypers)<1: return cfg, ''

        hyperstr = ''
        for hyper in cfg.train_hypers:
            if isinstance(hyper[1], int) and isinstance(hyper[2], int):
                setattr(cfg, hyper[0], np.random.randint(hyper[1],hyper[2]))
            else:
                setattr(cfg, hyper[0], np.random.uniform(hyper[1],hyper[2]))
            hyperstr += hyper[0]+'='+str(getattr(cfg, hyper[0]))+','
        hyperstr = hyperstr[:-1]

        return cfg, hyperstr

    def train(self, indir, outdir, wdir, fid_lst_tra, fid_lst_val, params_savefile, cont=None):

        if self.cfg.train_nbtrials>1:
            self._model.save(os.path.splitext(params_savefile)[0]+'-init.h5', printfn=print_log)

        try:
            trials = []
            for triali in xrange(1,1+self.cfg.train_nbtrials):  # Run multiple trials with different hyper-parameters
                print('\nStart trial {} ...'.format(triali))

                try:
                    train_rets = None
                    trialstr = 'trial'+str(triali)
                    if len(self.cfg.train_hypers)>0:
                        cfg, hyperstr = self.randomize_hyper(cfg)
                        trialstr += ','+hyperstr
                        print('    randomized hyper-parameters: '+trialstr)
                    if self.cfg.train_nbtrials>1:
                        self._model.load(os.path.splitext(params_savefile)[0]+'-init.h5')

                    timewholetrainstart = time.time()
                    train_rets = self.train_oneparamset(indir, outdir, wdir, fid_lst_tra, fid_lst_val, params_savefile, trialstr=trialstr, cont=cont)
                    cont = None
                    print_log('Total trial run time: {}s'.format(time2str(time.time()-timewholetrainstart)))

                except KeyboardInterrupt:                   # pragma: no cover
                    raise KeyboardInterrupt
                except (ValueError):     # TODO This needs , GpuArrayException ?  # pragma: no cover
                    if len(self.cfg.train_hypers)>0:
                        print_log('WARNING: Training crashed!')
                        import traceback
                        traceback.print_exc()
                    else:
                        print_log('ERROR: Training crashed!')
                        raise   # Crash the whole training if there is only one trial

                if self.cfg.train_nbtrials>1:
                    # Save the results of each trial, but only the non-crashed trials
                    if not train_rets is None:
                        ntrialline = [triali]+[getattr(cfg, field[0]) for field in self.cfg.train_hypers]
                        ntrialline = ntrialline+[train_rets[key] for key in sorted(train_rets.keys())]
                        header='trials '+' '.join([field[0] for field in self.cfg.train_hypers])+' '+' '.join(sorted(train_rets.keys()))
                        trials.append(ntrialline)
                        np.savetxt(os.path.splitext(params_savefile)[0]+'-trials.txt', np.vstack(trials), header=header)

        except KeyboardInterrupt:                           # pragma: no cover
            print_log('WARNING: Training interrupted by user!')

        print_log('Finished')


    # The functions below should be overwritten by any sub-class of OptimizerTTS
    # Default is LSE(MSE) training

    def default_options(self, cfg):
        # Options for LSE optimisation scheme
        cfg.train_lse_learningrate_log10 = -3.39794 # [potential hyper-parameter] (10**-3.39794=0.0004)
        cfg.train_lse_adam_beta1 = 0.9              # [potential hyper-parameter]
        cfg.train_lse_adam_beta2 = 0.999            # [potential hyper-parameter]
        cfg.train_lse_adam_epsilon_log10 = -8       # [potential hyper-parameter]
        return cfg

    def prepare(self):
        print('    Prepare LSE training')

        # opti = tf.train.RMSPropOptimizer(float(10**cfg.train_lse_learningrate_log10))  # Saving training states doesn't work with these ones
        # opti = keras.optimizers.RMSprop(lr=float(10**cfg.train_lse_learningrate_log10)) #
        opti = keras.optimizers.Adam(lr=float(10**self.cfg.train_lse_learningrate_log10), beta_1=float(self.cfg.train_lse_adam_beta1), beta_2=float(self.cfg.train_lse_adam_beta2), epsilon=float(10**self.cfg.train_lse_adam_epsilon_log10), decay=0.0, amsgrad=False)
        print('    optimizer: {}'.format(type(opti).__name__))

        print("    compiling training function ...")
        self._model.kerasmodel.compile(loss=lse_loss, optimizer=opti) # Use the explicit lse_loss instead of the built-in 'mse' for comparison purpose with WLSWGAN

    def train_on_batch(self, batchid, X_trab, Y_trab):

        train_returns = self._model.kerasmodel.train_on_batch(X_trab, Y_trab)
        cost_tra = np.sqrt(float(train_returns))

        return cost_tra # It has to return a cost/error/loss related to the generator/predictor's error, no matter the type of error (e.g. MSE, discri/critic error)

    def update_validation_cost(self, costs, X_vals, Y_vals):
        cost_validation_rmse = data.cost_model_prediction_rmse(self._model, [X_vals], Y_vals)
        costs['model_rmse_validation'].append(cost_validation_rmse)

        cost_val = costs['model_rmse_validation'][-1]

        return cost_val # It should return a cost value that is used for validation purpose. This cost_val will be used for saving the model if smaller than previous cost_val.

    def saveTrainingStateLossSpecific(self, fstate):
        # # Apparently the tf.keras.models.save_model saves the optimizer setup, but doesn't
        # # save its current parameter values. So save them in a seperate file.
        # # Or only necessary when using TF optimizers?
        # # https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
        # symbolic_weights = getattr(self._model.kerasmodel.optimizer, 'weights')
        # weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
        # with open(fstate+'.optimizer.pkl', 'wb') as f:
        #     cPickle.dump(weight_values, f)

        self._model.kerasmodel.save(fstate+'.model', include_optimizer=True)
        # self._model.kerasmodel.save_weights(fstate+'.model.weights.h5')

    def loadTrainingStateLossSpecific(self, fstate):
        # # Apparently the tf.keras.models.save_model saves the optimizer setup, but doesn't
        # # save its current parameter values. So load them from a seperate file.
        # # Or only necessary when using TF optimizers?
        # # https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state
        # self._model.kerasmodel._make_train_function()
        # with open(fstate+'.optimizer.pkl', 'rb') as f:
        #     weight_values = cPickle.load(f)
        # self._model.kerasmodel.optimizer.set_weights(weight_values)

        self._model.kerasmodel = load_model(fstate+'.model', custom_objects={'GaussianNoiseInput':networktts.GaussianNoiseInput, 'lse_loss':lse_loss}, compile=True)
        # self._model.kerasmodel.load_weights(fstate+'.model.weights.h5')
