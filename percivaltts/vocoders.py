'''
Vocoder classes (backend independent)

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

from percivaltts import *  # Always include this first to setup a few things

import numpy as np

from external.pulsemodel import sigproc as sp

from external import pulsemodel

class Vocoder:
    _name = None

    shift = None
    fs = None

    def __init__(self, _name, _fs, _shift):
        self._name = _name
        self.fs = _fs
        self.shift = _shift
        pass

        # if pp_spec_extrapfreq>0:
        #     idxlim = int(dftlen*pp_spec_extrapfreq/cfg.fs)
        #     for n in xrange(SPEC.shape[0]):
        #         SPEC[n,idxlim:] = SPEC[n,idxlim]
        #
        # if pp_spec_pf_coef>0:
        #     # A fast version of formant enhancer
        #     for n in xrange(SPEC.shape[0]):
        #         #if n*0.005<1.085: continue
        #         # Post-processing similar to Merlin's
        #         # But really NOT equivalent
        #         # This one creates way more low-pass effect with same coef (1.4)
        #         cc = np.fft.irfft(np.log(abs(SPEC[n,:])))
        #         cc = cc[:int(dftlen/2)+1]
        #         cc[1:] = 2.0*cc[1:]
        #         cc[2:] = pp_spec_pf_coef*cc[2:]
        #         spec_pp = abs(np.exp(np.fft.rfft(cc, dftlen)))
        #         if 0:
        #             import matplotlib.pyplot as plt
        #             plt.ion()
        #             plt.clf()
        #             FF = cfg.fs*np.arange(dftlen/2+1)/dftlen
        #             plt.plot(FF, sp.mag2db(SPEC[n,:]), 'k')
        #             plt.plot(FF, sp.mag2db(spec_pp), 'b')
        #             from IPython.core.debugger import  Pdb; Pdb().set_trace()
        #         SPEC[n,:] = spec_pp

    def __str__(self):
         return '{} (fs={}, shift={})'.format(self.name(), self.fs, self.shift)

    def name(self): return self._name

    def featuressize(self):
        raise ValueError('This member function needs to be re-implemented in the sub-classes')


class VocoderPML(Vocoder):
    spec_size = None
    nm_size = None

    def __init__(self, fs, shift, _spec_size, _nm_size, _dftlen=4096):
        Vocoder.__init__(self, 'PML', fs, shift)
        self.spec_size = _spec_size
        self.nm_size = _nm_size
        self.dftlen = _dftlen
        pass

    def featuressize(self):
        return 1+self.spec_size+self.nm_size

    def analysisf(self, cfg, fwav, ff0, fspec, fnm):              # pragma: no cover  coverage not detected
        print('Extracting PML features from: '+fwav)
        pulsemodel.analysisf(fwav, shift=self.shift, f0estimator='REAPER', f0_min=cfg.f0_min, f0_max=cfg.f0_max, ff0=ff0, f0_log=True, fspec=fspec, spec_nbfwbnds=self.spec_size, fnm=fnm, nm_nbfwbnds=self.nm_size, verbose=1)

        # pulsemodel.analysisf(wav_path.replace('*',fid), f0_min=cfg.f0_min, f0_max=cfg.f0_max, ff0=f0_path.replace('*',fid), f0_log=True, fspec=cp+wav_dir+'_mcep60/*.mcep'.replace('*',fid), spec_mceporder=59, verbose=1)

        # return [f0, SPEC, APER] # TODO

    def analysisfid(self, cfg, fid, wav_path, outputpathdicts):   # pragma: no cover  coverage not detected
        return self.analysisf(cfg, wav_path.replace('*',fid), outputpathdicts['f0'].replace('*',fid), outputpathdicts['spec'].replace('*',fid), outputpathdicts['noise'].replace('*',fid))

    def synthesis(self, fs, features):

        f0s = features[0]
        SPEC = features[1]
        NM = features[2]

        syn = pulsemodel.synthesis.synthesize(fs, f0s, SPEC, NM=NM, nm_cont=False, pp_atten1stharminsilences=-25)

        return syn


class VocoderWORLD(Vocoder):
    spec_size = None
    aper_size = None

    def __init__(self, fs, shift, _spec_size, _aper_size, _dftlen=4096):
        Vocoder.__init__(self, 'WORLD', fs, shift)
        self.spec_size = _spec_size
        self.spec_type = 'fwbnd' # 'fwbnd' 'mcep'
        self.aper_size = _aper_size
        self.dftlen = _dftlen
        pass

    def featuressize(self):
        return 1+self.spec_size+self.aper_size+1

    def analysisf(self, cfg, fwav, ff0, fspec, faper, fvuv):          # pragma: no cover  coverage not detected
        print('Extracting WORLD features from: '+fwav)

        wav, fs, _ = sp.wavread(fwav)

        import pyworld as pw

        if 0:
            # Check direct copy re-synthesis without compression/encoding
            print(pw.__file__)
            # _f0, ts = pw.dio(wav, fs, f0_floor=cfg.f0_min, f0_ceil=cfg.f0_max, channels_in_octave=2, frame_period=self.shift*1000.0)
            _f0, ts = pw.dio(wav, fs, f0_floor=cfg.f0_min, f0_ceil=cfg.f0_max, channels_in_octave=2, frame_period=self.shift*1000.0)
            # _f0, ts = pw.harvest(wav, fs)
            f0 = pw.stonemask(wav, _f0, ts, fs)
            SPEC = pw.cheaptrick(wav, f0, ts, fs, fft_size=self.dftlen)
            APER = pw.d4c(wav, f0, ts, fs, fft_size=self.dftlen)
            resyn = pw.synthesize(f0.astype('float64'), SPEC.astype('float64'), APER.astype('float64'), fs, self.shift*1000.0)
            sp.wavwrite('resynth.wav', resyn, fs, norm_abs=True, force_norm_abs=True, verbose=1)
            from IPython.core.debugger import  Pdb; Pdb().set_trace()

        _f0, ts = pw.dio(wav, fs, f0_floor=cfg.f0_min, f0_ceil=cfg.f0_max, channels_in_octave=2, frame_period=self.shift*1000.0)
        f0 = pw.stonemask(wav, _f0, ts, fs)
        SPEC = pw.cheaptrick(wav, f0, ts, fs, fft_size=self.dftlen)
        # SPEC = 10.0*np.sqrt(SPEC) # TODO Best gain correction I could find. Hard to find the good one between PML and WORLD different syntheses
        APER = pw.d4c(wav, f0, ts, fs, fft_size=self.dftlen)

        unvoiced = np.where(f0<20)[0]
        f0 = np.interp(ts, ts[f0>0], f0[f0>0])
        f0 = np.log(f0)
        makedirs(os.path.dirname(ff0))
        f0.astype('float32').tofile(ff0)

        vuv = np.ones(len(f0))
        vuv[unvoiced] = 0
        makedirs(os.path.dirname(fvuv))
        vuv.astype('float32').tofile(fvuv)

        if self.spec_type=='fwbnd':
            SPEC = sp.linbnd2fwbnd(np.log(abs(SPEC)), fs, self.dftlen, self.spec_size)
        elif self.spec_type=='mcep':
            # TODO test
            SPEC = sp.spec2mcep(SPEC*fs, sp.bark_alpha(fs), self.spec_size-1)

        makedirs(os.path.dirname(fspec))
        SPEC.astype('float32').tofile(fspec)

        APER = sp.linbnd2fwbnd(APER, fs, self.dftlen, self.aper_size)
        APER = sp.mag2db(APER)
        makedirs(os.path.dirname(faper))
        APER.astype('float32').tofile(faper)

        CMP = np.concatenate((f0.reshape((-1,1)), SPEC, APER, vuv.reshape((-1,1))), axis=1)

        if 0:
            import matplotlib.pyplot as plt
            plt.ion()
            resyn = self.synthesis(fs, CMP)
            sp.wavwrite('resynth.wav', resyn, fs, norm_abs=True, force_norm_abs=True, verbose=1)
            from IPython.core.debugger import  Pdb; Pdb().set_trace()

        return CMP

    def analysisfid(self, cfg, fid, wav_path, outputpathdicts):              # pragma: no cover  coverage not detected
        return self.analysisf(cfg, wav_path.replace('*',fid), outputpathdicts['f0'].replace('*',fid), outputpathdicts['spec'].replace('*',fid), outputpathdicts['noise'].replace('*',fid), outputpathdicts['vuv'].replace('*',fid))

    def synthesis(self, fs, CMP, pp_mcep=False):
        import pyworld as pw

        f0 = CMP[:,0]
        f0 = np.exp(f0)
        vuv = CMP[:,-1]
        f0[vuv<0.5] = 0

        SPEC = CMP[:,1:1+self.spec_size]
        if self.spec_type=='fwbnd':
            SPEC = np.exp(sp.fwbnd2linbnd(SPEC, fs, self.dftlen, smooth=True))
            if pp_mcep:
                print('        Merlin/SPTK Post-proc on MCEP')
                import external.merlin.generate_pp
                mcep = sp.spec2mcep(SPEC*fs, sp.bark_alpha(fs), 256)    # Arbitrary high order
                mcep_pp = external.merlin.generate_pp.mcep_postproc_sptk(mcep, fs, dftlen=self.dftlen) # Apply Merlin's post-proc on spec env
                SPEC = sp.mcep2spec(mcep_pp, sp.bark_alpha(fs), dftlen=self.dftlen)/fs
        elif self.spec_type=='mcep':
            # TODO test
            SPEC = sp.mcep2spec(SPEC, sp.bark_alpha(fs), dftlen=self.dftlen)
            if pp_mcep:
                print('        Merlin/SPTK Post-proc on MCEP')
                import external.merlin.generate_pp
                mcep_pp = external.merlin.generate_pp.mcep_postproc_sptk(SPEC, fs, dftlen=self.dftlen) # Apply Merlin's post-proc on spec env
                SPEC = sp.mcep2spec(mcep_pp, sp.bark_alpha(fs), dftlen=self.dftlen)
            SPEC /= float(fs)

        APER = CMP[:,1+self.spec_size:1+self.spec_size+self.aper_size]
        APER = sp.db2mag(APER)
        APER = sp.fwbnd2linbnd(APER, fs, self.dftlen)

        if 0:
            import matplotlib.pyplot as plt
            plt.ion()
            plt.subplot(311)
            plt.plot(f0)
            plt.xlim(0, len(f0))
            plt.subplot(312)
            plt.imshow(sp.mag2db(SPEC).T, origin='lower', aspect='auto', interpolation='none', cmap='jet', vmin=-140, vmax=5)
            plt.subplot(313)
            plt.imshow(APER.T, origin='lower', aspect='auto', interpolation='none', cmap='jet', vmin=0, vmax=1)
            from IPython.core.debugger import  Pdb; Pdb().set_trace()

        syn = pw.synthesize(f0.astype('float64'), SPEC.astype('float64'), APER.astype('float64'), fs, self.shift*1000.0)

        return syn

    # Objective measures for WORLD vocoder

    features_err = dict()

    def objmeasures_clear(self): self.features_err=dict()

    def objmeasures_add(self, CMP, REF):
        # Objective measurements
        f0trg = np.exp(REF[:,0])
        f0gen = np.exp(CMP[:,0])
        self.features_err.setdefault('F0[Hz]', []).append(np.sqrt(np.mean((f0trg-f0gen)**2)))
        spectrg = sp.log2db(REF[:,1:1+self.spec_size])
        specgen = sp.log2db(CMP[:,1:1+self.spec_size])
        self.features_err.setdefault('SPEC[dB]', []).append(np.sqrt(np.mean((spectrg-specgen)**2, 0)))
        apertrg = REF[:,1+self.spec_size:1+self.spec_size+self.aper_size]
        apergen = CMP[:,1+self.spec_size:1+self.spec_size+self.aper_size]
        self.features_err.setdefault('APER[dB]', []).append(np.sqrt(np.mean((apertrg-apergen)**2, 0)))

    def objmeasures_stats(self):
        for key in self.features_err:
            print('{} RMS={}'.format(key, np.mean(np.vstack(self.features_err[key]))))
