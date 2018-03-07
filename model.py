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

from utils import *  # Always include this first to setup a few things

import sys
import os
import cPickle

import numpy as np
numpy_force_random_seed()

print('\nLoading Theano')
from utils_theano import *
print_sysinfo_theano()
import theano
import theano.tensor as T
import lasagne
# lasagne.random.set_rng(np.random)

import data

class Model:

    # lasagne.nonlinearities.rectify, lasagne.nonlinearities.leaky_rectify, lasagne.nonlinearities.very_leaky_rectify, lasagne.nonlinearities.elu, lasagne.nonlinearities.softplus, lasagne.nonlinearities.tanh, networks.nonlin_softsign
    _nonlinearity = lasagne.nonlinearities.very_leaky_rectify

    # Network and Prediction variables

    insize = -1
    _input_values = None # Input contextual values (e.g. text, labels)
    inputs = None   # All the inputs of prediction function

    params_all = None # All possible parameters, including non trainable, running averages of batch normalisation, etc.
    params_trainable = None # Trainable parameters
    updates = []

    _hiddensize = 512

    outsize = -1
    net_out = None  # Network output
    outputs = None  # Outputs of prediction function

    predict = None  # Prection function

    def __init__(self, insize, outsize, specsize, nmsize, hiddensize=512):
        # Force additional random inputs is using anyform of GAN
        print("Building the model")

        self.insize = insize

        self.specsize = specsize
        self.nmsize = nmsize
        self._hiddensize = hiddensize
        self.outsize = outsize

        self._input_values = T.ftensor3('input_values')


    def init_finish(self, net_out):

        self.net_out = net_out

        print('    architecture:')
        for l in lasagne.layers.get_all_layers(self.net_out):
            print('        {}:{}'.format(l.name, l.output_shape))

        self.params_all = lasagne.layers.get_all_params(self.net_out)
        self.params_trainable = lasagne.layers.get_all_params(self.net_out, trainable=True)

        print('    params={}'.format(self.params_all))

        print('    compiling prediction function ...')
        self.inputs = [self._input_values]

        predicted_values = lasagne.layers.get_output(self.net_out, deterministic=True)
        self.outputs = predicted_values
        self.predict = theano.function(self.inputs, self.outputs, updates=self.updates)

        print('')


    def nbParams(self):
        return paramss_count(self.params_all)

    def saveAllParams(self, fmodel, cfg=None, extras=None, printfn=print):
        if extras is None: extras=dict()
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
        return DATA[1:]


    def generate(self, params_savefile, outsuffix, cfg, do_objmeas=True, do_resynth=True, indicestosynth=None
            , spec_comp='fwspec'
            , spec_size=129
            , nm_size=33
            , do_mlpg=False
            , pp_mcep=False
            , pp_spec_pf_coef=-1 # Common value is 1.2
            , pp_spec_extrapfreq=-1
            ):
        '''
            TODO Make fn to generate one arbitrary sample
        '''

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
        X_test = data.load(cfg.indir, fid_lst_gen, verbose=1)
        if do_objmeas:
            y_test = data.load(cfg.outdir, fid_lst_gen, verbose=1)
            X_test, y_test = data.cropsize((X_test, y_test))
            #cost_test = data.cost_model_merlin(mod, X_test, y_test, model_outsize=cfg.model_outsize)
            #print("    test cost = {:.6f} ({:.4f}%)".format(cost_test, 100*np.sqrt(cost_test)/np.sqrt(worst_val)))

        self.loadAllParams(params_savefile)              # Load the model's parameters

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

            elif spec_comp=='mcep': # pragma: no cover
                                    # nothing is guaranteed, this one even less.
                # SPTK necessary here, but it doesn't bring better quality
                # anyway, so no need to submodule SPTK nor test these lines.
                import generate_pp
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
                    cc = cc[:int(dftlen/2)+1]
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
            if vi>=len(X_test): continue

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
