'''
Useful functions related to Theano.

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

import percivaltts  # Always include this first to setup a few things

import os

import numpy as np
percivaltts.numpy_force_random_seed()

import theano
import theano.tensor as T

def print_sysinfo_theano():
    """Print some information about Theano installation"""
    print('    Theano: {} {}'.format(theano.__version__, theano.__file__))
    print('    THEANO_FLAGS: '+str(os.getenv('THEANO_FLAGS')))
    print('    floatX={}'.format(theano.config.floatX))
    print('    base_compiledir={}'.format(theano.config.base_compiledir))
    print('    device={}'.format(theano.config.device))
    print('    CUDA_ROOT={}'.format(theano.config.cuda.root))
    try:    print('    cuDNN={}'.format(theano.gpuarray.dnn.version()))
    except RuntimeError: print('    cuDNN=Unavailable')
    print('    GID={}'.format(percivaltts.nvidia_smi_current_gpu()))
    print('')

def th_cuda_available():
    """Returns True if CUDA is available"""
    return theano.config.cuda.root!=''

def th_print(msg, op):
    """Print the content of a theano variable with a message"""
    print_shape = theano.printing.Print(msg, attrs = [ 'shape' ])
    print_val = theano.printing.Print(msg)
    op = print_val(print_shape(op))
    return op

def th_print_shape(msg, op):
    """Print the content of a theano variable with a message"""
    print_shape = theano.printing.Print(msg, attrs = [ 'shape' ])
    op = print_shape(op)
    return op

def nonlin_tanh_saturated(x, coef=1.01):
    """Hyperbolic tangent which spans slightly below and above -1 and +1, in order to avoid unreachable -1 and +1 values."""
    return coef*T.tanh(x)

def nonlin_saturatedsigmoid(x, coef=1.01):
    """Sigmoid which spans slightly above +1, in order to avoid unreachable +1 values."""
    return coef*theano.tensor.nnet.sigmoid(x)

# def nonlin_tanh_byultrafastsigmoid(x):
    # return (T.nnet.ultra_fast_sigmoid(x)-0.5)*(1.0049698233144269*2.0)
def nonlin_tanh_bysigmoid(x):
    """Use sigmoid to implement hyperbolic tangent, in order to speed up the hyperbolic tangents using Theano's ultra_fast_sigmoid."""
    return (T.nnet.sigmoid(x)-0.5)*2.0

def nonlin_tanhcm11(x):
    """Hyperbolic tangent with maxima of 2nd derivative at -1 and +1"""
    return T.tanh((2.0/3.0)*x)

def nonlin_softsign(x):
    """Softsign activation function."""
    return x / (1.0+abs(x))

def nonlin_sigmoidparm(x, c=0.0, f=1.0):
    """Parametrized sigmoid in order to chose its center and smoothness."""
    return 1.0 / (1.0 + np.exp(-(x-c)*f))

def params_count(paramss):
    """
    Returns the number of parameters in the set of parameters.

    (counting verified on a FC512 only)
    """
    nbparams = 0
    for p in paramss:
        shap = p.get_value().shape
        if len(shap)==1: nbparams += shap[0]
        else:            nbparams += np.prod(shap)
    return nbparams

def print_network(net, params=None):
    for li, l in enumerate(lasagne.layers.get_all_layers(net)):
        # print('        {}: {}({})'.format(li, l.name, l.output_shape))
        print('        {}: {}({})'.format(li, l.name, l.output_shape))
        for p in l.get_params():
            X = p.get_value()
            istrainedstr = ''
            if (params!=None) and (p in params): istrainedstr=colored('TRAINING', 'green')
            print('            {}({}) [{}] {}'.format(p.name, X.shape, hash(str(X)), istrainedstr))

def params_keeponly(params, regexp, reverse=False):
    paramstotrain = []
    for p in params:
        if reverse:
            if re.search(regexp, str(p))==None:
                paramstotrain.append(p)
        else:
            if re.search(regexp, str(p))!=None:
                paramstotrain.append(p)

    return paramstotrain

def params_keeponlynewstr(params, paramsnew):
    paramsstr = [str(p) for p in params]

    paramstotrain = []
    for p in paramsnew:
        if not(str(p) in paramsstr):
            paramstotrain.append(p)

    return paramstotrain

def params_transfer(paramsfrom, paramsto):
    '''
    regexp : has to match parameter's name.
    '''

    paramstostr = [str(pto) for pto in paramsto]

    for pfrom in paramsfrom:
        if str(pfrom) in paramstostr:
            pmatch = [pto for pto in paramsto if str(pto)==str(pfrom)]
            if len(pmatch)>1:
                print('        {}: {} destination parameters match the same name'.format(str(pfrom),len(pmatch)))
            else:
                print(colored('        {}: TRANSFERRED'.format(str(pfrom)), 'green'))
                pmatch[0].set_value(pfrom.get_value())
        else:
            print(colored('        {}: Cannot be found in destination parameters! NOT transferred'.format(str(pfrom)), 'red'))

def layers_transfer(netfrom, netto, regexp, strictnames=True):
    '''
    regexp : has to match both layer and parameter's name.
    '''
    layersfrom = lasagne.layers.get_all_layers(netfrom)
    layers = lasagne.layers.get_all_layers(netto)
    for li in xrange(len(layersfrom)):
        lfrom = layersfrom[li]
        lto = layers[li]
        if re.search(regexp, lfrom.name)!=None:
            print('        {}: {}({})'.format(li, lfrom.name, lfrom.output_shape))
            for pfrom, pto in zip(lfrom.get_params(), lto.get_params()):
                if re.search(regexp, str(pto))!=None and ((not strictnames) or str(pfrom)==str(pto)):
                    print('            {} TRANSFERRED'.format(str(pto)))
                    pto.set_value(pfrom.get_value())
                else:
                    print('            {} SKIPPED {}'.format(str(pto), str(pfrom)))
