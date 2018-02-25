'''
Load, crop, stack the data into 3D matrices for training for building a data batch during training.

This file is meant to be library-independent (independent of theano, lasagne, tensorflow, etc.)

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
import glob
import time
import re
import cPickle

import numpy as np
import random

from utils import *

def loadids(fileids):
    with open(fileids, 'r') as f:
    #with open(cp+'/file_id_list.scp', 'r') as f:
        lines = f.readlines()
        lines = [x for x in map(str.strip, lines) if x]
        lines = filter(None, lines)
    return lines

def getpathandshape(path, shape=None):
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
                # The shape selector seems to be a file name
                # so take the size from the first dimension of the file's dimension
                X = np.fromfile(os.path.join(os.path.dirname(matches[0][0]), shapesstr), dtype=np.float32)
                shapeselector = shapeselector + (X.shape[0],)
        if not shape is None:
            print('WARNING: shape has been set both as argument ({}) of the getpathandshape(.) function and in the file selector ({}) (at the end of path); The one set as argument will be used: {}'.format(shape, shapeselector, shape))
        else:
            shape = shapeselector

    return path, shape

def getlastdim(path):
    _, size = getpathandshape(path)
    if size is None: return 1
    else:            return size[-1]

def load(dirpath, fbases, shape=None, frameshift=0.005, verbose=0, label=''):
    Xs = []

    totlen = 0
    memsize = 0

    dirpath, shape = getpathandshape(dirpath, shape)
    for n, fbase in enumerate(fbases):

        if verbose>0:
            print_tty('\r    {}Loading file {}/{} {}: ({:.2f}% done)        '.format(label, 1+n, len(fbases), fbase, 100*float(n)/len(fbases)))

        fX = dirpath.replace('*',fbase)
        if not os.path.isfile(fX):
            print('fX={} does not exists'.format(fX))
            raise

        X = np.fromfile(fX, dtype='float32')
        if not shape is None:
            X = X.reshape(shape)

        if np.isnan(X).any(): print('ERROR: There are nan in {}'.format(fX)); raise
        if np.isinf(X).any(): print('ERROR: There are inf in {}'.format(fX)); raise

        Xs.append(X)

        totlen += X.shape[0]
        memsize += (np.prod(X.shape))*4/(1024**2) # 4 implies float32

    if verbose>0:
        print_tty('\r                                                                 \r')
        print('    {}{} sentences, frames={} ({}), {} MB                     '.format(label, len(fbases), totlen,time.strftime('%H:%M:%S', time.gmtime((totlen*frameshift))), memsize))

    # Xs = np.array(Xs) # Leads to very weird assignements sometimes. What was it usefull for?

    return Xs

def gettotallen(Xs, axis=0):
    # for batched data, use axis=1
    l = 0
    for i in xrange(len(Xs)):
        l += Xs[i].shape[axis]
    return l

def cropsize(xs, axis=0):
    # Attention! It modifies the argument

    if axis>2: raise ValueError('Do not manage axis values bigger than 2')
    if len(set([len(x) for x in xs]))>1:
        raise ValueError('the size of the data sets are not identical ({})'.format([len(x) for x in xs]))

    # ys = [[] for i in range(len(xs))]
    for ki in xrange(len(xs[0])):   # For each sample of the data set
        # print('cropsize: {}'.format([x[ki].shape[axis] for x in xs]))
        siz = np.min([x[ki].shape[axis] for x in xs])
        for x in xs:
            # print('cropsize: {} {}'.format(x[ki].shape, siz))
            if axis==0:   x[ki] = x[ki][:siz,]
            elif axis==1: x[ki] = x[ki][:,:siz,]
            elif axis==2: x[ki] = x[ki][:,:,:siz,]
            # etc. TODO How to generalize?

    return xs

def cropsilences(xs, w, thresh=0.5):

    if len(set([len(w)]+[len(x) for x in xs]))>1:
        raise ValueError('the size of the data sets are not identical ({})'.format([len(x) for x in xs]))

    for ki in xrange(len(w)):   # For each sample of the data set

        speechidx = np.where(w[ki]>thresh)[0]
        # if len(speechidx)==0: from IPython.core.debugger import  Pdb; Pdb().set_trace()
        starti = min(speechidx)
        endi = max(speechidx)

        # Crop each feature given
        for x in xs:
            # print('cropsilences: {} {}'.format(starti, endi))
            x[ki] = x[ki][starti:endi,]

        # Crop the weight
        w[ki] = w[ki][starti:endi,]

    return xs, w


def vstack_masked(X, M):
    UX = []
    for b in xrange(X.shape[0]):
        seqlen = np.max(np.where(M[b,:]==1)[0])
        UX.append(X[b,:seqlen+1,:])
    return np.vstack(UX)

def maskify(xs, length=None, lengthmax=None, padtype='padright'):

    if len(set([len(x) for x in xs]))>1:
        raise ValueError('the size of the data sets are not identical ({})'.format([len(x) for x in xs]))

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
    for xi in xrange(len(xs)): xbs[xi] = np.zeros((len(xs[xi]), length, xs[xi][0].shape[1]), dtype='float32')
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
    framestop = np.zeros(X[0].shape[1]+1)
    framestop[-1] = value
    for xi in xrange(len(X)):
        X[xi] = np.concatenate((X[xi],np.zeros((X[xi].shape[0],1))), axis=1)
        X[xi] = np.vstack((X[xi],framestop))

    return X

def load_inoutset(indir, outdir, outwdir, fid_lst, inouttimesync=True, length=None, lengthmax=None, maskpadtype='padright', verbose=0):
    X_val = load(indir, fid_lst, verbose=verbose, label='Context labels: ')
    Y_val = load(outdir, fid_lst, verbose=verbose, label='Output features: ')
    W_val = load(outwdir, fid_lst, verbose=verbose, label='Time weights: ')

    # Crop time sequences according to model type
    if inouttimesync:
        X_val, Y_val, W_val = cropsize([X_val, Y_val, W_val])
        [X_val, Y_val], W_val = cropsilences([X_val, Y_val], W_val)
    else:
        X_val = addstop(X_val)
        Y_val, W_val = cropsize([Y_val, W_val])
        [Y_val], W_val = cropsilences([Y_val], W_val)
        Y_val = addstop(Y_val)

    # Maskify the validation data according to the batchsize
    if inouttimesync:
        [X_val, Y_val], MX_val = maskify([X_val, Y_val], length=length, lengthmax=lengthmax, padtype=maskpadtype)
        MY_val = MX_val
    else:
        [X_val], MX_val = maskify([X_val], length=length, lengthmax=lengthmax, padtype=maskpadtype)
        [Y_val], MY_val = maskify([Y_val], length=length, lengthmax=lengthmax, padtype=maskpadtype)

    return X_val, MX_val, Y_val, MY_val


# Evaluation functions ---------------------------------------------------------

def cost_0pred_rmse(Y_val):
    if isinstance(Y_val, list):
        worst_val = 0.0
        nbel = 0
        for k in xrange(len(Y_val)):
            worst_val += np.sum(Y_val[k]**2)
            nbel += Y_val[k].size
        worst_val /= nbel
        worst_val = np.sqrt(worst_val)
    else:
        worst_val = np.sqrt(np.mean(Y_val[k,]**2))
    return worst_val

def cost_model(fn, Xs):
    cost = 0.0
    if isinstance(Xs[0], list):
        nbel = 0
        for xi in xrange(len(Xs[0])): # Make them one by one to avoid blowing up the memory bcs the batch size would be too big

            ins = []
            for inp in Xs:
                ins.append(np.reshape(inp[xi],[1]+[s for s in inp[xi].shape]))

            cost += fn(*ins) # TODO Put [0] in an anonymous fn

        cost /= len(Xs[0])
    # else: # TODO

    return cost

def cost_model_prediction_rmse(mod, Xs, Y_val, inouttimesync=True):
    cost = 0.0
    if isinstance(Xs[0], list):
        nbel = 0
        for xi in xrange(len(Xs[0])): # Make them one by one to avoid blowing up the memory bcs the batch size would be too big
            ins = []
            for inp in Xs:
                ins.append(np.reshape(inp[xi],[1]+[s for s in inp[xi].shape]))
            ypred = mod.predict(*ins)

            cost += np.sum((Y_val[xi]-ypred[0,])**2)
            nbel += ypred.size
        cost /= nbel
        cost = np.sqrt(cost)
    # else: # TODO

    return cost

def prediction_std(mod, Xs):
    init_pred_std = 0.0
    if isinstance(Xs[0], list):
        for xi in xrange(len(Xs[0])): # Make them one by one to avoid blowing up the memory bcs the batch size would be too big
            ins = []
            for inp in Xs:
                ins.append(np.reshape(inp[xi],[1]+[s for s in inp[xi].shape]))
            ypred = mod.predict(*ins)
            init_pred_std += np.std(ypred)
        init_pred_std /= len(Xs[0]) # Average of std!
    # else: # TODO
            # ins.append(inp[xi:xi+1,])

    return init_pred_std
