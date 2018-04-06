'''
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

import sys
import os
import datetime
import re

import numpy as np
numpy_force_random_seed()

import data

def normalise_minmax(filepath, fids, featurepaths=None, outfilepath=None, nrange=None, keepidx=None, verbose=1):
    '''

    '''
    if nrange is None: nrange=[-1,1]
    if outfilepath is None: outfilepath=filepath
    print('Normalise data using min and max values (in={}, out={})'.format(filepath,outfilepath))

    mins = np.fromfile(os.path.dirname(filepath)+'/min.dat', dtype='float32')
    maxs = np.fromfile(os.path.dirname(filepath)+'/max.dat', dtype='float32')
    orisize = len(maxs)

    if keepidx is None: keepidx=np.arange(len(mins))

    mins = mins[keepidx]
    maxs = maxs[keepidx]

    if verbose>1:                                           # pragma: no cover
        print('    mins={}'.format(mins))
        print('    maxs={}'.format(maxs))

    # Write the statistics that are used for the normalisation
    if not os.path.isdir(os.path.dirname(outfilepath)): os.mkdir(os.path.dirname(outfilepath))
    mins.astype('float32').tofile(os.path.dirname(outfilepath)+'/min4norm.dat')
    maxs.astype('float32').tofile(os.path.dirname(outfilepath)+'/max4norm.dat')

    maxmindiff = (maxs-mins)

    maxmindiff[maxmindiff==0.0] = 1.0   # Avoid division by zero in dead dimensions

    for nf, fid in enumerate(fids):
        finpath = filepath.replace('*',fid)
        Y = np.fromfile(finpath, dtype='float32')
        Y = Y.reshape((-1,orisize))

        Y = Y[:,keepidx]

        Y = (Y - mins)/maxmindiff

        Y -= 0.5  # ... then center it ...
        Y *= 2.0  # ... and scale it to put it in [-1, 1]. Now DWTFYW
        Y *= (nrange[1]-nrange[0])/2.0    # 2.0 is the current range
        Y += 0.5*(nrange[0]+nrange[1])

        print_tty('\r    Write normed data file {}: {}                '.format(nf, fid))

        foutpath = outfilepath.replace('*',fid)
        Y.astype('float32').tofile(foutpath)
    print_tty('\r                                                           \r')

def normalise_meanstd(filepath, fids, featurepaths=None, outfilepath=None, keepidx=None, verbose=1):

    if outfilepath is None: outfilepath=filepath
    print('Normalise data using mean and standard-deviation (in={}, out={})'.format(filepath,outfilepath))

    means = np.fromfile(os.path.dirname(filepath)+'/mean.dat', dtype='float32')
    stds = np.fromfile(os.path.dirname(filepath)+'/std.dat', dtype='float32')

    if keepidx is None: keepidx=np.arange(len(means))

    if verbose>1:                                           # pragma: no cover
        print('    means4norm={}'.format(means))
        print('    stds4norm={}'.format(stds))

    # Write the statistics that are used for the normalisation
    if not os.path.isdir(os.path.dirname(outfilepath)): os.mkdir(os.path.dirname(outfilepath))
    means.astype('float32').tofile(os.path.dirname(outfilepath)+'/mean4norm.dat')
    stds.astype('float32').tofile(os.path.dirname(outfilepath)+'/std4norm.dat')

    stds[stds==0.0] = 1.0 # Force std to 1 for constant values to avoid division by zero
                          # This modification is not saved in std4norm.
                          # Though, during denormalisation, the data variance will be crushed to zero variance, and not one, which is the correct behavior.
    for nf, fid in enumerate(fids):
        finpath = filepath.replace('*',fid)
        Y = np.fromfile(finpath, dtype='float32')
        Y = Y.reshape((-1,len(means)))
        Y = (Y - means)/stds
        print_tty('\r    Write normed data file {}: {}                '.format(nf, fid))
        foutpath = outfilepath.replace('*',fid)
        Y.astype('float32').tofile(foutpath)
    print_tty('\r                                                           \r')

def normalise_meanstd_bndminmaxm11(filepath, fids, featurepaths=None, outfilepath=None, keepidx=None, verbose=1):

    if outfilepath is None: outfilepath=filepath
    print('Normalise data using mean and standard-deviation (in={}, out={})'.format(filepath,outfilepath))

    means = np.fromfile(os.path.dirname(filepath)+'/mean.dat', dtype='float32')
    stds = np.fromfile(os.path.dirname(filepath)+'/std.dat', dtype='float32')

    if keepidx is None: keepidx=np.arange(len(means))

    if 1:
        # TODO This is specific to BNDNM usual setup !!!
        # Recover sizes
        dummy, f0size = data.getpathandshape(featurepaths[0])
        if f0size is None: f0size=1
        dummy, specsize = data.getpathandshape(featurepaths[1])
        specsize = specsize[1]
        dummy, bndnmsize = data.getpathandshape(featurepaths[2])
        bndnmsize = bndnmsize[1]
        outsizeori = f0size+specsize+bndnmsize
        print('sizes f0:{} spec:{} noise:{}'.format(f0size, specsize, bndnmsize))

        # Hack for bndnm
        if 0: # Do not scale it at all
            means[f0size+specsize:f0size+specsize+bndnmsize] = 0.0
            stds[f0size+specsize:f0size+specsize+bndnmsize] = 1.0

            means[outsizeori+f0size+specsize:outsizeori+f0size+specsize+bndnmsize] = 0.0
            stds[outsizeori+f0size+specsize:outsizeori+f0size+specsize+bndnmsize] = 1.0

            means[2*outsizeori+f0size+specsize:2*outsizeori+f0size+specsize+bndnmsize] = 0.0
            stds[2*outsizeori+f0size+specsize:2*outsizeori+f0size+specsize+bndnmsize] = 1.0

        elif 1: # Scale Noise Mask values to [-1,+1], the so called "bndminmaxm11"
            #self.output = T.set_subtensor(self.output[:,starti:starti+act[0]/3], T.nnet.sigmoid(lin_output[:,starti:starti+act[0]/3]))
            #self.output = T.set_subtensor(self.output[:,starti+act[0]/3:starti+2*act[0]/3], 0.5*T.tanh(lin_output[:,starti+act[0]/3:starti+2*act[0]/3]))
            #self.output = T.set_subtensor(self.output[:,starti+2*act[0]/3:starti+act[0]], 2*T.tanh(lin_output[:,starti+2*act[0]/3:starti+act[0]]))

            #stds[:] /= 0.288675135    # Put the variance, not to 1, but to the devstd of uniform [-0.5,+0.5]

            # The following will normalise bnd output to zero centered and unit variance
            means[f0size+specsize:f0size+specsize+bndnmsize] = 0.5
            #stds[f0size+specsize:f0size+specsize+bndnmsize] = 1.0 # Already in [-0.5,+0.5] # *2 in order to have std 0.288675135 of uniform distrib
            stds[f0size+specsize:f0size+specsize+bndnmsize] = 0.5 # Put in [-1,+1]

            if len(means)>outsizeori:
                #print('hack delta')
                means[outsizeori+f0size+specsize:outsizeori+f0size+specsize+bndnmsize] = 0.0
                #stds[outsizeori+f0size+specsize:outsizeori+f0size+specsize+bndnmsize] = 1.0 # Already in [-0.5,+0.5]
                stds[outsizeori+f0size+specsize:outsizeori+f0size+specsize+bndnmsize] = 0.5 # Put in [-1,+1]

                if len(means)>2*outsizeori:
                    #print('hack delta2')
                    means[2*outsizeori+f0size+specsize:2*outsizeori+f0size+specsize+bndnmsize] = 0.0
                    #stds[2*outsizeori+f0size+specsize:2*outsizeori+f0size+specsize+bndnmsize] = 2.0*2.0 # in [-2,+2] even though mostly in [-1,+1]
                    stds[2*outsizeori+f0size+specsize:2*outsizeori+f0size+specsize+bndnmsize] = 1.0 # Mostly in [-1,+1]

    if verbose>1:                                           # pragma: no cover
        print('    means4norm={}'.format(means))
        print('    stds4norm={}'.format(stds))

    # Write the statistics that are used for the normalisation
    if not os.path.isdir(os.path.dirname(outfilepath)): os.mkdir(os.path.dirname(outfilepath))
    means.astype('float32').tofile(os.path.dirname(outfilepath)+'/mean4norm.dat')
    stds.astype('float32').tofile(os.path.dirname(outfilepath)+'/std4norm.dat')

    stds[stds==0.0] = 1.0 # Force std to 1 for constant values to avoid division by zero
    for nf, fid in enumerate(fids):
        finpath = filepath.replace('*',fid)
        Y = np.fromfile(finpath, dtype='float32')
        Y = Y.reshape((-1,len(means)))
        Y = (Y - means)/stds
        print_tty('\r    Write normed data file {}: {}                '.format(nf, fid))
        sys.stdout.flush()
        foutpath = outfilepath.replace('*',fid)
        Y.astype('float32').tofile(foutpath)
    print_tty('\r                                                           \r')

def normalise_meanstd_bndnmnoscale(filepath, fids, featurepaths=None, outfilepath=None, keepidx=None, verbose=1):

    if outfilepath is None: outfilepath=filepath
    print('Normalise data using mean and standard-deviation (in={}, out={})'.format(filepath,outfilepath))

    means = np.fromfile(os.path.dirname(filepath)+'/mean.dat', dtype='float32')
    stds = np.fromfile(os.path.dirname(filepath)+'/std.dat', dtype='float32')

    if keepidx is None: keepidx=np.arange(len(means))

    if 1:
        # Recover sizes of each feature
        f0size = data.getlastdim(featurepaths[0])       # ATTENTION: This is specific to BNDNM usual ordering of usual PML features !!!
        specsize = data.getlastdim(featurepaths[1])     # ATTENTION: This is specific to BNDNM usual ordering of usual PML features !!!
        bndnmsize = data.getlastdim(featurepaths[2])    # ATTENTION: This is specific to BNDNM usual ordering of usual PML features !!!
        outsizeori = f0size+specsize+bndnmsize
        print('sizes f0:{} spec:{} noise:{}'.format(f0size, specsize, bndnmsize))

        # Hack the moments for bndnm to avoid any normalisation
        means[f0size+specsize:f0size+specsize+bndnmsize] = 0.0
        stds[f0size+specsize:f0size+specsize+bndnmsize] = 1.0

        if len(means)>outsizeori:
            means[outsizeori+f0size+specsize:outsizeori+f0size+specsize+bndnmsize] = 0.0
            stds[outsizeori+f0size+specsize:outsizeori+f0size+specsize+bndnmsize] = 1.0

            if len(means)>2*outsizeori:
                means[2*outsizeori+f0size+specsize:2*outsizeori+f0size+specsize+bndnmsize] = 0.0
                stds[2*outsizeori+f0size+specsize:2*outsizeori+f0size+specsize+bndnmsize] = 1.0

    if verbose>1:                                           # pragma: no cover
        print('    means4norm={}'.format(means))
        print('    stds4norm={}'.format(stds))

    # Write the statistics that are used for the normalisation in seperate files
    if not os.path.isdir(os.path.dirname(outfilepath)): os.mkdir(os.path.dirname(outfilepath))
    means.astype('float32').tofile(os.path.dirname(outfilepath)+'/mean4norm.dat')
    stds.astype('float32').tofile(os.path.dirname(outfilepath)+'/std4norm.dat')

    stds[stds==0.0] = 1.0 # Force std to 1 for constant values to avoid division by zero
    for nf, fid in enumerate(fids):
        finpath = filepath.replace('*',fid)
        Y = np.fromfile(finpath, dtype='float32')
        Y = Y.reshape((-1,len(means)))
        Y = (Y - means)/stds
        print_tty('\r    Write normed data file {}: {}                '.format(nf, fid))
        foutpath = outfilepath.replace('*',fid)
        Y.astype('float32').tofile(foutpath)
    print_tty('\r                                                           \r')


def compose(featurepaths, fileidspath, outfilepath, wins=None, id_valid_start=-1, normfn=None, do_finalcheck=False, dropzerovardims=False, verbose=1):
    '''
    wins: default values are wins=[[-0.5, 0.0, 0.5], [1.0, -2.0, 1.0]] (as in Merlin)
    '''
    print('Compose data (id_valid_start={})'.format(id_valid_start))

    if wins is None: wins=[]

    outfilepath = re.sub(r':[^:]+$', "", outfilepath)   # ignore any shape suffix in the output path
    if not os.path.isdir(os.path.dirname(outfilepath)): os.mkdir(os.path.dirname(outfilepath))

    if len(wins)>0: import scipy.signal
    size = None
    mins = None
    maxs = None
    means = None
    nbframes = 0
    with open(fileidspath) as f:
        lines = f.readlines()
        lines = [x for x in map(str.strip, lines) if x]
        fids = filter(None, lines)

        for nf, fid in enumerate(fids):
            print_tty('\r    Composing file {}/{} {}               '.format(1+nf, len(fids), fid))

            features = []
            minlen = None
            for featurepath in featurepaths:
                infilepath, shape = data.getpathandshape(featurepath)
                if shape is None: shape=(-1,1)
                infilepath = infilepath.replace('*',fid)
                feature = np.fromfile(infilepath, dtype='float32')
                feature=feature.reshape(shape)
                features.append(feature)
                if minlen is None:  minlen=feature.shape[0]
                else:               minlen=np.min((minlen,feature.shape[0]))

            # Crop features to same length
            for feati in xrange(len(features)):
                features[feati] = features[feati][:minlen,]

            Y = np.hstack(features)

            if len(wins)>0:
                YWs = [Y] # Always add first the static values
                for win in wins:
                    # Then concatenate the windowed values
                    YW = np.ones(Y.shape)
                    win_p = (len(win)+1)/2
                    for d in xrange(Y.shape[1]):
                        YW[win_p-1:-(win_p-1),d] = -scipy.signal.convolve(Y[:,d], win)[win_p:-win_p] # The fastest
                        YW[:win_p-1,d] = YW[win_p-1,d]
                        YW[-(win_p-1):,d] = YW[-(win_p-1)-1,d]
                    YWs.append(YW)
                Y = np.hstack(YWs)

                #if 0:
                    #from merlin.mlpg_fast import MLParameterGenerationFast as MLParameterGeneration
                    #mlpg_algo = MLParameterGeneration()
                    #var = np.tile(np.ones(CMP.shape[1]),(CMP.shape[0],1)) # Simplification!
                    #YGEN = mlpg_algo.generation(CMP, var, 1)

                    #plt.plot(Y, 'k')
                    #plt.plot(YGEN, 'b')
                    #from IPython.core.debugger import  Pdb; Pdb().set_trace()

            size = Y.shape[1]

            if nf<id_valid_start:
                if mins is None:  mins=Y.min(axis=0)
                else:             mins=np.minimum(mins, Y.min(axis=0))
                if maxs is None:  maxs=Y.max(axis=0)
                else:             maxs=np.maximum(maxs, Y.max(axis=0))
                if means is None: means =Y.sum(axis=0).astype('float64')
                else:             means+=Y.sum(axis=0).astype('float64')
                nbframes += Y.shape[0]

            #print('\r    Write data file {}: {}                '.format(nf, fid)),
            Y.astype('float32').tofile(outfilepath.replace('*',fid))
        print_tty('\r                                                           \r')

        means /= nbframes
        zerovaridx = np.where((maxs-mins)==0.0)[0]  # Indices of dimensions having zero-variance

        mins.astype('float32').tofile(os.path.dirname(outfilepath)+'/min.dat')
        if verbose>1: print('    mins={}'.format(mins))     # pragma: no cover
        maxs.astype('float32').tofile(os.path.dirname(outfilepath)+'/max.dat')
        if verbose>1: print('    maxs={}'.format(maxs))     # pragma: no cover
        means.astype('float32').tofile(os.path.dirname(outfilepath)+'/mean.dat')
        if verbose>1: print('    means={}'.format(means))   # pragma: no cover

        # Now that we have the mean, we can do the std
        stds = None
        for nf, fid in enumerate(fids):
            Y = np.fromfile(outfilepath.replace('*',fid), dtype='float32')
            Y = Y.reshape((-1,size))
            if nf<id_valid_start:
                if stds is None: stds =((Y-means)**2).sum(axis=0).astype('float64')
                else:            stds+=((Y-means)**2).sum(axis=0).astype('float64')
        stds /= nbframes-1  # unbiased variance estimator
        stds = np.sqrt(stds)

        stds.astype('float32').tofile(os.path.dirname(outfilepath)+'/std.dat')
        if verbose>1: print('    stds={}'.format(stds))

        keepidx = np.arange(len(means))
        if dropzerovardims:
            keepidx = np.setdiff1d(np.arange(len(means)), zerovaridx)
            size = len(keepidx)
            keepidx.astype('int32').tofile(os.path.dirname(outfilepath)+'/keepidx.dat')
            print('Dropped dimensions with zero variance. Remains {} dims'.format(size))

        if normfn is not None: # TODO This shouldn't be called within compose, it should come afterwards
            normfn(outfilepath, fids, featurepaths=featurepaths, keepidx=keepidx, verbose=verbose)

        print('{} files'.format(len(fids)))
        print('{} frames ({}s assuming 5ms time shift)'.format(nbframes, datetime.timedelta(seconds=nbframes*0.005)))
        strsize = ''
        for fpath in featurepaths:
            dummy, shape = data.getpathandshape(fpath)
            if shape is None:   strsize+='1+'
            else:               strsize+=str(shape[1])+'+'
        strsize = strsize[:-1]
        if dropzerovardims:
            strsize+='-'+str(len(zerovaridx))
        print('nb dimensions={} (features: ({})x{})'.format(size, strsize, 1+len(wins)))
        print('{} dimensions with zero-variance ({}){}'.format(len(zerovaridx), zerovaridx, ', which have been dropped' if dropzerovardims else ', which have been kept'))
        if normfn is not None:
            print('normalisation done using: {}'.format(normfn.__name__))
        else:
            print('no normalisation called')
        print('output path: {}'.format(outfilepath))

        if do_finalcheck:
            print('Check data final statistics')
            verif_means = None
            verif_stds = None
            verif_mins = None
            verif_maxs = None
            verif_nbframes = 0
            for nf, fid in enumerate(fids):
                if nf>=id_valid_start: continue
                fpath = outfilepath.replace('*',fid)
                Y = np.fromfile(fpath, dtype='float32')
                Y = Y.reshape((-1,size))
                if verif_means is None: verif_means =Y.sum(axis=0).astype('float64')
                else:                   verif_means+=Y.sum(axis=0).astype('float64')
                if verif_mins is None:  verif_mins=Y.min(axis=0)
                else:                   verif_mins=np.minimum(verif_mins, Y.min(axis=0))
                if verif_maxs is None:  verif_maxs=Y.max(axis=0)
                else:                   verif_maxs=np.maximum(verif_maxs, Y.max(axis=0))
                verif_nbframes += Y.shape[0]
            verif_means /= verif_nbframes
            for nf, fid in enumerate(fids):
                if nf>=id_valid_start: continue
                fpath = outfilepath.replace('*',fid)
                Y = np.fromfile(fpath, dtype='float32')
                Y = Y.reshape((-1,size))
                if verif_stds is None: verif_stds =((Y-verif_means)**2).sum(axis=0).astype('float64')
                else:                  verif_stds+=((Y-verif_means)**2).sum(axis=0).astype('float64')
            verif_stds /= verif_nbframes-1
            if verbose>1: # TODO                            # pragma: no cover
                print('verif_min={}'.format(verif_mins))
                print('verif_max={}'.format(verif_maxs))
                print('verif_means={}'.format(verif_means))
                print('verif_stds={}'.format(verif_stds))


def create_weights(specfeaturepath, fileidspath, outfilepath, thresh=-32, dftlen=4096, spec_type='fwlspec'):
    '''
    This function creates a one-column vector with one weight value per frame.
    E.g. The weight is here below computed as a silence coefficient. During
    training, silent segments at the very begining or very end of the sample
    will be dropped (i.e. dropped if weight<0.5).

    thresh : [dB] The weight of the frames whose energy < threshold are set
             weight = 0, and 1 otherwise.
    '''

    def mag2db(a): return 20.0*np.log10(np.abs(a))

    outfilepath = re.sub(r':[^:]+$', "", outfilepath)   # ignore any shape suffix in the output path
    if not os.path.isdir(os.path.dirname(outfilepath)): os.mkdir(os.path.dirname(outfilepath))

    with open(fileidspath) as f:
        lines = f.readlines()
        lines = [x for x in map(str.strip, lines) if x]
        fids = filter(None, lines)

        for nf, fid in enumerate(fids):
            print_tty('\r    Processing feature files {} for {}                '.format(nf, fid))

            infilepath, shape = data.getpathandshape(specfeaturepath)
            if shape is None: shape=(-1,1)
            infilepath = infilepath.replace('*',fid)

            if spec_type=='fwlspec':
                Yspec = np.fromfile(infilepath, dtype='float32')
                Yspec = Yspec.reshape(shape)
                ener = mag2db(np.exp(np.mean(Yspec, axis=1)))
            elif spec_type=='mcep':
                Ymcep = np.fromfile(infilepath, dtype='float32')
                Ymcep = Ymcep.reshape(shape)
                ener = mag2db(np.exp(Ymcep[:,0]))    # Just extract the first coef
            elif spec_type=='fwcep':
                Ymcep = np.fromfile(infilepath, dtype='float32')
                Ymcep = Ymcep.reshape(shape)
                ener = mag2db(np.exp(Ymcep[:,0]))    # Just extract the first coef

            ener -= np.max(ener)

            weight = ener.copy()
            weight[ener>=thresh] = 1.0
            weight[ener<thresh] = 0.0

            weight.astype('float32').tofile(outfilepath.replace('*',fid))

            if 0:
                import matplotlib.pyplot as plt
                plt.plot(ener, 'k')
                plt.plot(np.log10(weight), 'b')
                plt.plot([0, len(ener)], thresh*np.array([1, 1]), 'k')
                from IPython.core.debugger import  Pdb; Pdb().set_trace()

        print_tty('\r                                                           \r')
