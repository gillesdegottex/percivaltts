## Percival: CNN-GAN acoustic model for speech synthesis

Based on Python/Theano/Lasagne, using Wasserstein GAN and training
regularization to optimise 2D convolutional layers.

It uses the [PML vocoder](https://github.com/gillesdegottex/pulsemodel) for
the waveform representation.

### Legal
Copyright(C) 2017 Engineering Department, University of Cambridge, UK.

The code in this repository is released under the Apache License, Version 2.0. Please see LICENSE.md file for more details.
All source files of any kind (code source and any ressources), except
the content of the 'external' directory, are under the same license.
Please refer to the content of the 'external' directory for the legal issues
related to those code source.

### Inspired by

Wasserstein GAN [article](https://arxiv.org/abs/1701.07875)

    https://gist.github.com/f0k/f3190ebba6c53887d598d03119ca2066
    https://github.com/martinarjovsky/WassersteinGAN
    https://github.com/fairytale0011/Conditional-WassersteinGAN
    http://blog.richardweiss.org/2017/07/21/conditional-wasserstein-gan.html

Improved training for Wasserstein GAN [article](https://arxiv.org/abs/1704.00028)

    https://github.com/tjwei/GANotebooks/blob/master/wgan2-lasagne.ipynb
    https://github.com/ririw/ririw.github.io/blob/master/assets/conditional-wasserstein-gans/Improved.ipynb

Least Square mixing [article](https://arxiv.org/abs/1611.07004)

Other DNN pipelines

    [Merlin][https://github.com/CSTR-Edinburgh/merlin)

### Author/Contributor
Gilles Degottex <gad27@cam.ac.uk>


### Working versions

Dealing with the numerous dependencies between the libraries and tools can be
a nightmare. I strongly suggest to use a package manager [conda](https://conda.io/docs/) or [miniconda](https://conda.io/miniconda.html)
on top of the OS package manager.

Here are versions the are known to work using miniconda
libffi                    3.2.1                h4deb6c0_3  
libgcc-ng                 7.2.0                hcbc56d2_1  
libgpuarray               0.6.2                         0  
libstdcxx-ng              7.2.0                h24385c6_1  
numpy                     1.12.1                   py27_0  
pygpu                     0.6.2                    py27_0  
python                    2.7.13              hfff3488_13  
scipy                     0.19.1              np112py27_0  
theano                    0.9.0                    py27_0  
