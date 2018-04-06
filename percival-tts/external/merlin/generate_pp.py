################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://svn.ecdf.ed.ac.uk/repo/inf/dnn_tts/
#                
#                Centre for Speech Technology Research                 
#                     University of Edinburgh, UK                       
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.                           
#                                                                       
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute  
#  this software and its documentation without restriction, including   
#  without limitation the rights to use, copy, modify, merge, publish,  
#  distribute, sublicense, and/or sell copies of this work, and to      
#  permit persons to whom this work is furnished to do so, subject to   
#  the following conditions:
#  
#   - Redistributions of source code must retain the above copyright  
#     notice, this list of conditions and the following disclaimer.   
#   - Redistributions in binary form must reproduce the above         
#     copyright notice, this list of conditions and the following     
#     disclaimer in the documentation and/or other materials provided 
#     with the distribution.                                          
#   - The authors' names may not be used to endorse or promote products derived 
#     from this software without specific prior written permission.   
#                                  
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK        
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE     
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       
#  THIS SOFTWARE.
################################################################################

#/usr/bin/python -u

# Reformated on 2017-04-08 gad27@cam.ac.uk

import sys, os, subprocess, commands
import numpy as np

import pulsemodel.sigproc as sp

def run_process(args,log=True):

    #print(args)

    # a convenience function instead of calling subprocess directly
    # this is so that we can do some logging and catch exceptions
    
    try:
        # the following is only available in later versions of Python
        # rval = subprocess.check_output(args)
        
        # bufsize=-1 enables buffering and may improve performance compared to the unbuffered case
        p = subprocess.Popen(args, bufsize=-1, shell=True, 
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        close_fds=True)
        # better to use communicate() than read() and write() - this avoids deadlocks
        (stdoutdata, stderrdata) = p.communicate()

        if p.returncode != 0:
            # for critical things, we always log, even if log==False
            print('exit status %d' % p.returncode )
            print(' for command: %s' % args )
            print('      stderr: %s' % stderrdata )
            print('      stdout: %s' % stdoutdata )
            raise OSError
            
        return (stdoutdata, stderrdata)
        
    except subprocess.CalledProcessError as e:
        # not sure under what circumstances this exception would be raised in Python 2.6
        print('exit status %d' % e.returncode )
        print(' for command: %s' % args )
        # not sure if there is an 'output' attribute under 2.6 ? still need to test this...
        print('  output: %s' % e.output )
        raise

    except ValueError:
        print('ValueError for %s' % args )
        raise
        
    except OSError:
        print('OSError for %s' % args )
        raise

    except KeyboardInterrupt:
        print('KeyboardInterrupt during %s' % args )
        try:
            # try to kill the subprocess, if it exists
            p.kill()
        except UnboundLocalError:
            # this means that p was undefined at the moment of the keyboard interrupt
            # (and we do nothing)
            pass

        raise KeyboardInterrupt

def mcep_postproc_sptk(mcep, fs, dftlen=4096):

    fmgcori = sp.gentmpfile('mgcori')
    mcep.astype('float32').tofile(fmgcori)

    mgc_dim = mcep.shape[1]

    fw_coef = sp.bark_alpha(fs)
    co_coef = dftlen/2+1
    
    pf_coef = 1.4
    
    fweight = sp.gentmpfile('fweight')
    
    line = "echo 1 1 "
    for i in range(2, mgc_dim):
        line = line + str(pf_coef) + " "

    # Write down the weights in float32
    run_process('{line} | {x2x} +af > {weight}'
                .format(line=line, x2x='x2x', weight=fweight))

    # Compute autocorr of decompressed cepstrum (unwarped cepstrum), i.e. original autocorr
    run_process('{freqt} -m {order} -a {fw} -M {co} -A 0 < {mgc} | {c2acr} -m {co} -M 0 -l {fl} > {base_r0}'
                .format(freqt='freqt', order=mgc_dim-1, fw=fw_coef, co=co_coef, mgc=fmgcori, c2acr='c2acr', fl=dftlen, base_r0=fmgcori+'_r0'))

    # Weight the warped cepstrum and get the resulting autocorr
    run_process('{vopr} -m -n {order} < {mgc} {weight} | {freqt} -m {order} -a {fw} -M {co} -A 0 | {c2acr} -m {co} -M 0 -l {fl} > {base_p_r0}'
                .format(vopr='vopr', order=mgc_dim-1, mgc=fmgcori, weight=fweight,
                        freqt='freqt', fw=fw_coef, co=co_coef, 
                        c2acr='c2acr', fl=dftlen, base_p_r0=fmgcori+'_p_r0'))

    # Weight the warped cepstrum and get the corresponding MLSA coefs
    run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 0 -e 0 > {base_b0}'
                .format(vopr='vopr', order=mgc_dim-1, mgc=fmgcori, weight=fweight,
                        mc2b='mc2b', fw=fw_coef, 
                        bcp='bcp', base_b0=fmgcori+'_b0'))

    # divide the original autocorr by the weighted-cep autocorr; apply log; divide by 2; add the weighted-cep MLSA coefs
    run_process('{vopr} -d < {base_r0} {base_p_r0} | {sopr} -LN -d 2 | {vopr} -a {base_b0} > {base_p_b0}'
                .format(vopr='vopr', base_r0=fmgcori+'_r0', base_p_r0=fmgcori+'_p_r0', 
                        sopr='sopr', 
                        base_b0=fmgcori+'_b0', base_p_b0=fmgcori+'_p_b0'))

    # re-extract weighted-cep MLSA coefs and ... and merge and ...
    run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 1 -e {order} | {merge} -n {order2} -s 0 -N 0 {base_p_b0} | {b2mc} -m {order} -a {fw} > {base_p_mgc}'
                .format(vopr='vopr', order=mgc_dim-1, mgc=fmgcori, weight=fweight,
                        mc2b='mc2b',  fw=fw_coef, 
                        bcp='bcp', 
                        merge='merge', order2=mgc_dim-2, base_p_b0=fmgcori+'_p_b0',
                        b2mc='b2mc', base_p_mgc=fmgcori+'_p_mgc'))

    mgcpp = np.fromfile(fmgcori+'_p_mgc', dtype='float32')
    mgcpp = np.reshape(mgcpp, (-1, mcep.shape[1]))

    #print(mgcpp.shape)

    return mgcpp
