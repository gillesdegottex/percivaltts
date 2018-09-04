'''
Load, crop, stack the data into 3D matrices for training for building a data batch during training.
(independent of the ML backend)

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

import os
import copy
import time
import re

import numpy as np
numpy_force_random_seed()

def getpath(path):
    """
    Split a path into a standard UNIX-style path, discarding any optional 'shape extension' that defines the shape of the data in each file.
    """
    matches = re.findall(r'(.*):\((.*)\)', path)
    if len(matches)>0:
        path = matches[0][0]

    return path

def getpathandshape(path, shape=None):
    """
    Split a path into a standard UNIX-style path and an optional 'shape extension' that defines the shape of the data in each file.
    E.g. /data/supervoice/spectra/*.spec:(-1,60)
    """
    matches = re.findall(r'(.*):\((.*)\)', path)
    if len(matches)>0:
        path = matches[0][0]
        shapesstrs = matches[0][1].split(',')
        shapeselector = ()
        for shapesstr in shapesstrs:
            if is_int(shapesstr):
                # The shape selector is a simple integer
                shapeselector = shapeselector + (int(shapesstr),)
            else:
                # The shape selector seems to be a file name in the same directory
                # so take the size from the first dimension of the file's dimension
                X = np.fromfile(os.path.join(os.path.dirname(matches[0][0]), shapesstr), dtype=np.float32)
                shapeselector = shapeselector + (X.shape[0],)
        if not shape is None:
            print('WARNING: shape has been set both as argument ({}) of the getpathandshape(.) function and in the file selector ({}) (at the end of path); The one set as argument will be used: {}'.format(shape, shapeselector, shape))
        else:
            shape = shapeselector

    return path, shape

def getlastdim(path):
    """
    Return the last dimension of the optional shape extension of a given path.
    """
    _, size = getpathandshape(path)
    if size is None: return 1
    else:            return size[-1]

def loadfile(fpath, fbase=None, shape=None):
    if not fbase is None:
        fpath = fpath.replace('*',fbase)

    fpath, shape = getpathandshape(fpath, shape)

    if not os.path.isfile(fpath):
        raise ValueError('{} does not exists'.format(fpath))# pragma: no cover

    X = np.fromfile(fpath, dtype='float32')
    if not shape is None:
        X = X.reshape(shape)

    if np.isnan(X).any(): ValueError('ERROR: There are nan in {}'.format(fpath))
    if np.isinf(X).any(): ValueError('ERROR: There are inf in {}'.format(fpath))

    return X

def load(dirpath, fbases, shape=None, frameshift=0.005, verbose=0, label=''):
    """
    Load data into a list of matrices.
    """
    Xs = []

    totlen = 0
    memsize = 0

    dirpath, shape = getpathandshape(dirpath, shape)
    for n, fbase in enumerate(fbases):

        if verbose>0:
            print_tty('\r    {}Loading file {}/{} {}: ({:.2f}% done)        '.format(label, 1+n, len(fbases), fbase, 100*float(n)/len(fbases)))

        fX = dirpath.replace('*',fbase)
        if not os.path.isfile(fX):
            raise ValueError('{} does not exists'.format(fX))# pragma: no cover

        X = np.fromfile(fX, dtype='float32')
        if not shape is None:
            X = X.reshape(shape)

        if np.isnan(X).any(): ValueError('ERROR: There are nan in {}'.format(fX))
        if np.isinf(X).any(): ValueError('ERROR: There are inf in {}'.format(fX))

        Xs.append(X)

        totlen += X.shape[0]
        memsize += (np.prod(X.shape))*4/(1024**2) # 4 implies float32

    if verbose>0:
        print_tty('\r                                                                 \r')
        print('    {}{} sentences, frames={} ({}), {} MB                     '.format(label, len(fbases), totlen,time.strftime('%H:%M:%S', time.gmtime((totlen*frameshift))), memsize))

    # Xs = np.array(Xs) # Leads to very weird assignements sometimes. What was it usefull for?

    return Xs

def gettotallen(Xs, axis=0):
    """Return the sum of the length matrices in a given list of matrices, on a given axis."""
    # for batched data, use axis=1
    l = 0
    for i in xrange(len(Xs)):
        l += Xs[i].shape[axis]
    return l

def croplen(xs, axis=0):
    """
    Crop each matrices of a list of matrices to the same length among other matching list of matrices (Attention: Argument xs modified!)
    E.g.
    A = [zeros(134, 60), zeros(542, 60)]
    B = [zeros(135, 12), zeros(538, 12)]
    [A, B] = croplen([A, B])
    Ensures:
    A: [zeros(134, 60), zeros(538, 60)]
    B: [zeros(134, 12), zeros(538, 12)]
    """

    if axis>2:
        raise ValueError('Do not manage axis values bigger than 2') # pragma: no cover
    if len(set([len(x) for x in xs]))>1:
        raise ValueError('the size of the data sets are not identical ({})'.format([len(x) for x in xs])) # pragma: no cover

    # ys = [[] for i in range(len(xs))]
    for ki in xrange(len(xs[0])):   # For each sample of the data set
        # print('croplen: {}'.format([x[ki].shape[axis] for x in xs]))
        siz = np.min([x[ki].shape[axis] for x in xs])
        for x in xs:
            # print('croplen: {} {}'.format(x[ki].shape, siz))
            if axis==0:   x[ki] = x[ki][:siz,]
            elif axis==1: x[ki] = x[ki][:,:siz,]            # pragma: no cover
            elif axis==2: x[ki] = x[ki][:,:,:siz,]          # pragma: no cover
            # etc. TODO How to generalize this?

    return xs

def croplen_weight(xs, w, thresh=0.5, cropmode='begend', cropsize=int(0.750/0.005)):
    """
    Similar to croplen(xs), but crop according to some weight w and a threshold on this weight (only at beginning and end of file).
    """

    if len(set([len(w)]+[len(x) for x in xs]))>1:
        raise ValueError('the size of the data sets are not identical ({})'.format([len(x) for x in xs])) # pragma: no cover

    for ki in xrange(len(w)):   # For each sample of the data set

        if cropmode=='begend':
            if len(w[ki].shape)>1:      speechidx = np.where(w[ki][:,0]>thresh)[0]
            else:                       speechidx = np.where(w[ki]>thresh)[0]

            starti = min(speechidx)
            endi = max(speechidx)

            # Crop each feature given at the beginning and end
            for x in xs:
                # print('cropsilences: {} {}'.format(starti, endi))
                x[ki] = x[ki][starti:endi,]     # TODO This is changing the reference!

            # Crop the weight
            w[ki] = w[ki][starti:endi,]

        elif cropmode=='begendbigger':
            # Start as usual...
            if len(w[ki].shape)>1:      keep=w[ki][:,0]>thresh
            else:                       keep=w[ki]>thresh

            speechidx = np.where(keep)[0]
            # ... and replace the False where the distance is small
            speechidxd = np.diff(speechidx)
            spidxd1 = np.where(speechidxd>1)[0]
            for spd1 in spidxd1:
                if speechidxd[spd1]<int(cropsize):
                    keep[speechidx[spd1]:speechidx[spd1+1]] = True
            speechidx = np.where(keep)[0]

            for x in xs:
                # print('cropsilences: {} {}'.format(starti, endi))
                x[ki] = x[ki][speechidx,]       # TODO This is changing the reference!

            # Crop the weight too
            w[ki] = w[ki][speechidx,]

        elif cropmode=='all':

            if len(w[ki].shape)>1:      speechidx = np.where(w[ki][:,0]>thresh)[0]
            else:                       speechidx = np.where(w[ki]>thresh)[0]

            for x in xs:
                # print('cropsilences: {} {}'.format(starti, endi))
                x[ki] = x[ki][speechidx,]       # TODO This is changing the reference!

            # Crop the weight
            w[ki] = w[ki][speechidx,]

    return xs, w


def maskify(xs, length=None, lengthmax=None, padtype='randshift'):
    """
    Create a batched composition of multiple matrices in xs (resulting of 3D matrices for each sentence).
    Various pading types are supported, the most common being 'padright', which add zeros at the end of matrices that are too short.

    Returns
    -------
    xbs : list of the batched composition of the elements of xs.
    MB : A mask with 1 at meaningfull values in elements of xbs and 0 where the matrice was too short.
    """

    if len(set([len(x) for x in xs]))>1:
        raise ValueError('the size of the data sets are not identical ({})'.format([len(x) for x in xs])) # pragma: no cover

    if length is None:
        # Consider only the first var
        maxlength = xs[0][0].shape[0]
        minlength = xs[0][0].shape[0]
        for b in xrange(1,len(xs[0])):
            maxlength = np.max((maxlength, xs[0][b].shape[0]))
            minlength = np.min((minlength, xs[0][b].shape[0]))

        if padtype=='padright': length = maxlength
        else:                   length = minlength

    if not lengthmax is None:
        if length>lengthmax: length=lengthmax

    xbs = [None]*len(xs)
    for xi in xrange(len(xs)):
        featsize = 1 if len(xs[xi][0].shape)==1 else xs[xi][0].shape[1]
        xbs[xi] = np.zeros((len(xs[xi]), length, featsize), dtype='float32')
    MB = np.zeros((len(xs[0]), length), dtype='float32')

    shift = 0
    for b in xrange(len(xs[0])):
        #samplelen = np.min([x[b].shape[0] for x in xs]) # Use smallest size among all features (does cropping at the same time)
        samplelen = xs[0][b].shape[0] # The length of the sample b (assuming samples have been cropped to same length among features already)
        minlen = np.min([samplelen, length])

        if padtype=='randshift':
            shift = np.random.randint(0,(samplelen-length)+1)   # Assume this sample length is always >= minlen

        for xi in xrange(len(xs)):
            xbs[xi][b,:minlen,:] = xs[xi][b][shift:shift+minlen,:]
            xbs[xi][b,:minlen,:] = xs[xi][b][shift:shift+minlen,:]
        MB[b,:minlen] = 1

    return xbs, MB

def addstop(X, value=1.0):
    """Add a stop symbol to inputs"""
    X = copy.deepcopy(X)
    framestop = np.zeros(X[0].shape[1]+1)
    framestop[-1] = value
    for xi in xrange(len(X)):
        X[xi] = np.concatenate((X[xi],np.zeros((X[xi].shape[0],1))), axis=1)
        X[xi] = np.vstack((X[xi],framestop))

    return X

def load_inoutset(indir, outdir, outwdir, fid_lst, inouttimesync=True, length=None, lengthmax=None, maskpadtype='padright', cropmode='begend', verbose=0):
    """Directly load batches of input and corresponding outputs (crop the lengths)."""

    X_val = load(indir, fid_lst, verbose=verbose, label='Context labels: ')
    Y_val = load(outdir, fid_lst, verbose=verbose, label='Output features: ')
    W_val = load(outwdir, fid_lst, verbose=verbose, label='Time weights: ')

    # Crop time sequences according to model type
    if inouttimesync:
        X_val, Y_val, W_val = croplen([X_val, Y_val, W_val])
        [X_val, Y_val], W_val = croplen_weight([X_val, Y_val], W_val, cropmode=cropmode)
    else:
        X_val = addstop(X_val)
        Y_val, W_val = croplen([Y_val, W_val])
        [Y_val], W_val = croplen_weight([Y_val], W_val, cropmode=cropmode)
        Y_val = addstop(Y_val)

    # Maskify the validation data according to the batchsize
    if inouttimesync:
        [X_val, Y_val, W_val], MX_val = maskify([X_val, Y_val, W_val], length=length, lengthmax=lengthmax, padtype=maskpadtype)
        MY_val = MX_val
    else:     # TODO rm
        [X_val], MX_val = maskify([X_val], length=length, lengthmax=lengthmax, padtype=maskpadtype)
        [Y_val], MY_val = maskify([Y_val], length=length, lengthmax=lengthmax, padtype=maskpadtype)

    return X_val, MX_val, Y_val, MY_val, W_val


# Evaluation functions ---------------------------------------------------------

def cost_0pred_rmse(Y_val):
    """
    Compute the Root Mean Square Error (RMSE), assuming the prediction is always zero (i.e. worst predictor RMSE).
    This is the true RMSE of the data in Y_val (not the mean of sub-RMSEs).
    """
    if isinstance(Y_val, list):
        worst_val = 0.0
        nbel = 0
        for k in xrange(len(Y_val)):
            worst_val += np.sum(Y_val[k]**2)
            nbel += Y_val[k].size
        worst_val /= nbel               # This is not variance, so no nbel-1
        worst_val = np.sqrt(worst_val)
    else:
        worst_val = np.sqrt(np.mean(Y_val**2))
    return worst_val

def cost_model_mfn(fn, Xs):
    """Run a function on the argument Xs and average the returned values."""
    cost = 0.0
    if isinstance(Xs[0], list):
        for xi in xrange(len(Xs[0])): # Make them one by one to avoid blowing up the memory TODO still even a single one might be too big

            ins = []
            for inp in Xs:
                ins.append(np.reshape(inp[xi],[1]+[s for s in inp[xi].shape]))

            cost += fn(*ins) # TODO Put [0] in an anonymous fn  # TODO without a square errors could compensate on bi-directionlal errors (as in GAN)

        cost /= len(Xs[0])

    return cost

def cost_model_prediction_rmse(mod, Xs, Y_val, inouttimesync=True):
    """Compute the RMSE between prediction from Xs and ground truth values Y_val."""
    cost = 0.0
    if isinstance(Xs[0], list):
        nbel = 0
        for xi in xrange(len(Xs[0])): # Make them one by one to avoid blowing up the memory
            ins = []
            for inp in Xs:
                ins.append(np.reshape(inp[xi],[1]+[s for s in inp[xi].shape]))
            ypred = mod.predict(*ins)
            cost += np.sum((Y_val[xi]-ypred[0,])**2)
            nbel += ypred[0,].size
        cost /= nbel                    # This is not variance, so no nbel-1
        cost = np.sqrt(cost)

    return cost

def prediction_mstd(mod, Xs):
    """Mean of standard-deviation of each sample"""
    init_pred_std = 0.0
    if isinstance(Xs[0], list):
        for xi in xrange(len(Xs[0])): # Make them one by one to avoid blowing up the memory
            ins = []
            for inp in Xs:
                ins.append(np.reshape(inp[xi],[1]+[s for s in inp[xi].shape]))
            ypred = mod.predict(*ins)
            init_pred_std += np.std(ypred[0,])
        init_pred_std /= len(Xs[0]) # Average of std!

    return init_pred_std

def prediction_rms(mod, Xs):
    """Return RMS of the predicted values (used for verification purposes)"""
    init_pred_rms = 0.0
    if isinstance(Xs[0], list):
        nbel = 0
        for xi in xrange(len(Xs[0])): # Make them one by one to avoid blowing up the memory
            ins = []
            for inp in Xs:
                ins.append(np.reshape(inp[xi],[1]+[s for s in inp[xi].shape]))
            ypred = mod.predict(*ins)
            init_pred_rms += np.sum((ypred[0,])**2)
            nbel += ypred[0,].size
        init_pred_rms /= nbel              # This is not variance, so no nbel-1
        init_pred_rms = np.sqrt(init_pred_rms)

    return init_pred_rms
