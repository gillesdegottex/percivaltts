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

import time
import random
import cPickle
import functools

import numpy as np

from utils import *
import data

print('\nLoading Theano')
import theano
import theano.tensor as T
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/external/Lasagne/')
import lasagne
print_sysinfo_theano()

from utils_theano import *

class Model:

    def nbParams(self):
        return paramss_count(self.params_all)

    def saveAllParams(self, fmodel, cfg=None, extras=dict(), printfn=print):
        # https://github.com/Lasagne/Lasagne/issues/159
        printfn('    saving parameters in {} ...'.format(fmodel), end='')
        sys.stdout.flush()
        paramsvalues = [(str(p), p.get_value()) for p in self.params_all]
        DATA = [paramsvalues, cfg, extras]
        cPickle.dump(DATA, open(fmodel, 'wb'))
        print(' done')
        sys.stdout.flush()

    def loadAllParams(self, fmodel, printfn=print):
        # https://github.com/Lasagne/Lasagne/issues/159
        printfn('    reloading parameters from {} ...'.format(fmodel), end='')
        sys.stdout.flush()
        DATA = cPickle.load(open(fmodel, 'rb'))
        for p, v in zip(self.params_all, DATA[0]): p.set_value(v[1])
        print(' done')
        sys.stdout.flush()

    def saveTrainingState(self, fstate, cfg=None, extras=dict(), printfn=print):
        # https://github.com/Lasagne/Lasagne/issues/159
        printfn('    saving training state in {} ...'.format(fstate), end='')
        sys.stdout.flush()

        paramsvalues = [(str(p), p.get_value()) for p in self.params_all] # The network parameters

        ovs = []
        for ov in self.optim_updates:
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
        for p, v in zip(self.params_all, DATA[0]): p.set_value(v[1])    # The network parameters

        for ov, da in zip(self.optim_updates, DATA[1]):
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


        return DATA[3]

    # Training =================================================================

    def train(self, params, indir, outdir, outwdir, fid_lst_tra, fid_lst_val, X_vals, Y_vals, cfg, params_savefile, trialstr='', cont=None):
        raise ValueError('You need to implement train(.)')

    def randomize_hyper(self, cfg):
        # Randomized the hyper parameters
        if len(cfg.hypers)<1: return ''

        hyperstr = ''
        for hyper in cfg.hypers:
            if type(hyper[1]) is int and type(hyper[2]) is int:
                setattr(cfg, hyper[0], np.random.randint(hyper[1],hyper[2]))
            else:
                setattr(cfg, hyper[0], np.random.uniform(hyper[1],hyper[2]))
            hyperstr += hyper[0]+'='+str(getattr(cfg, hyper[0]))+','
        hyperstr = hyperstr[:-1]

        return cfg, hyperstr

    def train_multipletrials(self, indir, outdir, outwdir, fid_lst_tra, fid_lst_val, params, params_savefile, cfgtomerge=None, cont=None, **kwargs):
        # Hyp: always uses batches

        # All kwargs arguments are specific configuration values
        # First, fill a struct with the default configuration values ...
        cfg = configuration() # Init structure
        cfg.train_batchsize = 5         # [potential hyper-parameter]

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
        # X_val, Y_val = data.load_inoutset(indir, outdir, outwdir, fid_lst_val, verbose=1)
        X_vals = data.load(indir, fid_lst_val, verbose=1, label='Context labels: ')
        Y_vals = data.load(outdir, fid_lst_val, verbose=1, label='Output features: ')
        X_vals, Y_vals = data.cropsize([X_vals, Y_vals])

        if cfg.train_nbtrials>1:
            self.saveAllParams(params_savefile+'.init', cfg=cfg, printfn=print_log)
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
                        self.loadAllParams(params_savefile+'.init')
                        # self.loadTrainingState(params_savefile+'.trainingstate-init')

                    timewholetrainstart = time.time()
                    train_rets = self.train(params, indir, outdir, outwdir, fid_lst_tra, fid_lst_val, X_vals, Y_vals, cfg, params_savefile, trialstr=trialstr, cont=cont)
                    cont = None
                    print_log('Total trial run time: {}s'.format(time2str(time.time()-timewholetrainstart)))

                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    if len(cfg.train_hypers)>0: print_log('WARNING: Training crashed!')
                    else:                       print_log('ERROR: Training crashed!')
                    import traceback
                    traceback.print_exc()
                    pass

                if cfg.train_nbtrials>1:
                    trials.append([triali]+[getattr(cfg, field[0]) for field in cfg.train_hypers]+[train_rets[key] for key in sorted(train_rets.keys())])
                    # Save results of each trial
                    np.savetxt(params_savefile+'.trials', np.vstack(trials), header=('trials '+' '.join([field[0] for field in cfg.train_hypers]+sorted(train_rets.keys()))))

        except KeyboardInterrupt:
            print_log('WARNING: Training interrupted by user!')
            pass

        print_log('Finished')


    def generate(self, params_savefile, outsuffix, cfg, do_objmeas=True, do_resynth=True, indicestosynth=None
            , spec_comp='fwspec'
            , spec_size=129
            , nm_size=33
            , do_mlpg=False
            , pp_mcep=False
            , pp_spec_pf_coef=-1 # Common value is 1.2
            , pp_spec_extrapfreq=-1
            ):

        # Options dependent on features composition
        outsize_wodeltas = 1+spec_size+nm_size    # 1+: logf0

        # Options for synthesis only
        dftlen = 4096
        f0clipmin = 50
        f0clipmax = 700

        # ----------------------------------------------------------------------

        fid_lst = data.loadids(cfg.fileids)
        fid_lst_gen = fid_lst[cfg.id_valid_start+cfg.id_valid_nb:cfg.id_valid_start+cfg.id_valid_nb+cfg.id_test_nb]

        print('Reloading output stats')
        Ymean = np.fromfile(os.path.dirname(cfg.outdir)+'/mean4norm.dat', dtype='float32')
        Ystd = np.fromfile(os.path.dirname(cfg.outdir)+'/std4norm.dat', dtype='float32')

        print('\nLoading generation data at once ...')
        if do_objmeas:
            X_test = data.load(cfg.indir, fid_lst_gen, verbose=1)
            y_test = data.load(cfg.outdir, fid_lst_gen, verbose=1)
            X_test, y_test = data.cropsize((X_test, y_test))
            #cost_test = data.cost_model_merlin(mod, X_test, y_test, model_outsize=cfg.model_outsize)
            #print("    test cost = {:.6f} ({:.4f}%)".format(cost_test, 100*np.sqrt(cost_test)/np.sqrt(worst_val)))
        else:
            X_test = data.load(cfg.indir, fid_lst_gen, verbose=1)
            #(X_test), M_test = data.maskify([X_test])

        # import model_generative
        # mod = model_generative.ModelGAN(X_test[0].shape[1], len(Ymean))    # Build the model
        self.loadAllParams(params_savefile)              # Load the model's parameters

        import generate_pp

        def decomposition(CMP, outsize_wodeltas, do_mlpg=False, pp_mcep=True, f0clipmin=-1, f0clipmax=-1):

            # Denormalise
            CMP = CMP*np.tile(Ystd, (CMP.shape[0], 1)) + np.tile(Ymean, (CMP.shape[0], 1))

            if do_mlpg:
                from mlpg_fast import MLParameterGenerationFast as MLParameterGeneration
                mlpg_algo = MLParameterGeneration(delta_win=[-0.5, 0.0, 0.5], acc_win=[1.0, -2.0, 1.0])
                var = np.tile(Ystd**2,(CMP.shape[0],1)) # Simplification!
                CMP = mlpg_algo.generation(CMP, var, len(Ymean)/3)
            else:
                CMP = CMP[:,:outsize_wodeltas]

            f0sgen = CMP[:,0].copy()
            f0sgen[f0sgen>0] = np.exp(f0sgen[f0sgen>0])
            if f0clipmin>0: f0sgen=np.clip(f0sgen, f0clipmin, f0clipmax)
            ts = (cfg.shift)*np.arange(len(f0sgen))
            f0sgen = np.vstack((ts, f0sgen)).T

            mcep = CMP[:,1:1+spec_size]
            if spec_comp=='fwspec':
                SPEC = np.exp(sp.fwbnd2linbnd(mcep, cfg.fs, dftlen, smooth=True))

            elif spec_comp=='mcep':
                if pp_mcep: mcep=generate_pp.mcep_postproc_sptk(mcep, cfg.fs, dftlen=dftlen) # Apply Merlin's post-proc on spec env
                SPEC = sp.mcep2spec(mcep, sp.bark_alpha(cfg.fs), dftlen=dftlen)

            if 0:
                import matplotlib.pyplot as plt
                plt.ion()
                plt.imshow(sp.mag2db(SPEC).T, origin='lower', aspect='auto', interpolation='none', cmap='jet', extent=[0.0, ts[-1], 0.0, cfg.fs/2])
                #plt.plot(ts, 0.5*cfg.fs*LSF/np.pi, 'k')
                from IPython.core.debugger import  Pdb; Pdb().set_trace()

            bndnm = CMP[:,1+spec_size:]
            nm = sp.fwbnd2linbnd(bndnm, cfg.fs, dftlen)

            if pp_spec_extrapfreq>0:
                idxlim = int(dftlen*pp_spec_extrapfreq/cfg.fs)
                for n in xrange(SPEC.shape[0]):
                    SPEC[n,idxlim:] = SPEC[n,idxlim]

            if pp_spec_pf_coef>0:
                for n in xrange(SPEC.shape[0]):
                    #if n*0.005<1.085: continue
                    # Post-processing similar to Merlin's
                    # But really NOT equivalent
                    # This one creates way more low-pass effect with same coef (1.4)
                    cc = np.fft.irfft(np.log(abs(SPEC[n,:])))
                    cc = cc[:dftlen/2+1]
                    cc[1:] = 2.0*cc[1:]
                    cc[2:] = pp_spec_pf_coef*cc[2:]
                    spec_pp = abs(np.exp(np.fft.rfft(cc, dftlen)))
                    if 0:
                        import matplotlib.pyplot as plt
                        plt.ion()
                        plt.clf()
                        FF = cfg.fs*np.arange(dftlen/2+1)/dftlen
                        plt.plot(FF, sp.mag2db(SPEC[n,:]), 'k')
                        plt.plot(FF, sp.mag2db(spec_pp), 'b')
                        from IPython.core.debugger import  Pdb; Pdb().set_trace()
                    SPEC[n,:] = spec_pp

            return f0sgen, SPEC, nm, mcep, bndnm

        import pulsemodel
        import pulsemodel.sigproc as sp
        syndir = os.path.splitext(params_savefile)[0] + outsuffix
        if not os.path.isdir(syndir): os.makedirs(syndir)
        features_err = dict()

        if indicestosynth is None: indicestosynth=range(0,len(X_test))
        for vi in indicestosynth:
            print('Generating {}/{} ...'.format(1+vi, len(X_test)))
            print('    Predict ...')

            CMP = self.predict(np.reshape(X_test[vi],[1]+[s for s in X_test[vi].shape]))

            # (X_test_masked,), _ = data.maskify([[X_test[vi]]])
            # CMP = self.predict(X_test_masked)
            CMP = CMP[0,:,:]

            print('    Decompose ...')
            f0sgen, specgen, nmgen, mcepgen, bndnmgen = decomposition(CMP, outsize_wodeltas=outsize_wodeltas, do_mlpg=do_mlpg, pp_mcep=pp_mcep, f0clipmin=f0clipmin, f0clipmax=f0clipmax)

            if do_objmeas:
                # Objective measurements
                f0strg, spectrg, nmtrg, mceptrg, bndnmtrg = decomposition( y_test[vi], outsize_wodeltas=outsize_wodeltas, do_mlpg=False, pp_mcep=False)
                features_err.setdefault('F0', []).append(np.sqrt(np.mean((f0strg[:,1]-f0sgen[:,1])**2)))
                features_err.setdefault('MCEP', []).append(sp.log2db(np.sqrt(np.mean((mceptrg-mcepgen)**2, 0))))
                features_err.setdefault('BNDNM', []).append(np.sqrt(np.mean((bndnmtrg-bndnmgen)**2, 0)))

                if do_resynth:
                    # resyn = pulsemodel.synthesis.synthesize(cfg.fs, f0strg, spectrg, NM=nmtrg, nm_forcebinary=True) # Prev version
                    resyn = pulsemodel.synthesis.synthesize(cfg.fs, f0strg, spectrg, NM=nmtrg, nm_cont=False)
                    sp.wavwrite(syndir+'/'+fid_lst_gen[vi]+'-resynth.wav', resyn, cfg.fs, norm_abs=True, force_norm_abs=True, verbose=1)

            # syn = pulsemodel.synthesis.synthesize(cfg.fs, f0sgen, specgen, NM=nmgen, nm_forcebinary=True)
            syn = pulsemodel.synthesis.synthesize(cfg.fs, f0sgen, specgen, NM=nmgen, nm_cont=False)

            sp.wavwrite(syndir+'/'+fid_lst_gen[vi]+'.wav', syn, cfg.fs, norm_abs=True, force_norm_abs=True, verbose=1)

        for key in features_err:
            # np.mean(np.vstack(features_err[key]), 0) TODO Per dimension
            print('{} err={}'.format(key, np.mean(np.vstack(features_err[key]))))

        print_log('Generation finished')
