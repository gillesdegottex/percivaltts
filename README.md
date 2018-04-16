[![Build Status](https://travis-ci.org/gillesdegottex/percivaltts.svg?branch=master)](https://travis-ci.org/gillesdegottex/percivaltts)
[![codecov](https://codecov.io/gh/gillesdegottex/percivaltts/branch/master/graph/badge.svg)](https://codecov.io/gh/gillesdegottex/percivaltts)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/3a6a54997f8e44ba9230fbd278e00d8a)](https://www.codacy.com/app/gillesdegottex/percivaltts?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=gillesdegottex/percivaltts&amp;utm_campaign=Badge_Grade)

## Percival: CNN-WGAN acoustic model for text-to-speech synthesis

<img align="right" width="300" src="./docs/spec_sample.png">

<br/><br/>

Based on Python/Theano/Lasagne, using Wasserstein GAN and training
regularization to optimise 2D convolutional layers.

It uses [festival](http://festvox.org/festival/) and [Merlin](https://github.com/CSTR-Edinburgh/merlin) scripts to generate
the input text labels. The [PML vocoder](https://github.com/gillesdegottex/pulsemodel) is used for the waveform synthesis.

<br/><br/>

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

PixelCNN [article](https://arxiv.org/abs/1606.05328)

    https://arxiv.org/pdf/1704.03809.pdf
    http://www.dtic.upf.edu/~mblaauw/MdM_NIPS_seminar/

<br/><br/>

### Legal

Copyright(C) 2017 Engineering Department, University of Cambridge, UK.

The code in this repository is released under the Apache License, Version 2.0.
Please see LICENSE.md file for more details.

All source files of any kind (code source and any ressources), except the
content of the 'external' directory, are under the same license. Please refer
to the content of the 'external' directory for the legal issues related to those
code source.

This project has received funding from the European Union's Horizon 2020
research and innovation programme under the Marie Skłodowska-Curie grant
agreement No 655764.

<br/>

### Dependencies and working versions

Percival is _not_ a standalone pipeline for TTS. It only trains an acoustic
model. Technically, it is dependent on a text-to-audio alignment system, which
usually provides context input labels (e.g. in HTS format; as label_state_align
in Merlin).

Dealing with the numerous dependencies between the libraries and tools can also
be a nightmare. We strongly suggest to use a package manager
[conda](https://conda.io/docs/) or [miniconda](https://conda.io/miniconda.html)
on top of the OS package manager. Here are versions the are known to work using
miniconda:
```
libffi                    3.2.1                h4deb6c0_3  
libgcc-ng                 7.2.0                hcbc56d2_1  
libgpuarray               0.6.2                         0  
libstdcxx-ng              7.2.0                h24385c6_1  
numpy                     1.12.1                   py27_0  
pygpu                     0.6.2                    py27_0  
python                    2.7.13              hfff3488_13  
scipy                     0.19.1              np112py27_0  
theano                    0.9.0                    py27_0  
```
And other version numbers
```
CUDA                      9.0
NVidia Drivers            384.111
```

If you want to use MLPG (disabled by default), you will also need to install:
```
$ sudo pip install bandmat
```

### Install and run the demo

First clone the git repository:
```
$ git clone git@github.com:gillesdegottex/percivaltts.git
```
Then, go into the repository directory and compile the various tools neccessary
to run Percival:
```
$ cd percivaltts
$ make build
```

Go into the Python module directory:
```
$ cd percivaltts
```

Edit `setenv.sh` according to your CUDA/Theano installation (see above).

Download the demo data:
```
$ make tests/slt_arctic_merlin_full
```

And finally run the demo!
```
$ make run
```

[A web page with the expected resulting demonstration samples can be found here](http://gillesdegottex.eu/Demos/PercivalTTS/compare_percival_0.9.5/)

### Preparing a new voice
Like in Merlin, you basically need the three following elements from any corpus:
* `label_state_align`

    The same directory that is imported in Merlin. It is produced by an
    alignement tool, as in HTS. Merlin provides the necessary scripts for
    generating the text labels from text inputs and align these text labels on
    some given waveform.

* `wav`

    The waveforms directory aligned with the labels above

* `file_id_list.scp`

    The same file that you find in Merlin, which contains the basenames of each
    file in `label_state_align` and `wav`.    

Put this somehwere in a common directory and point the `cp` variable in the
`run.py` file to this directory.


### Formats

The are a few assumptions across the code about data formats.

First, floating point precision values are always saved on disc using the `float32` format.

#### Data set
The basenames of the corpus files are listed in a file (e.g. file_id_list.scp).
This list is then split into [traning; validation; test] sets, always in this order.
The validation start starts at `id_valid_start` and contains `id_valid_nb` files.
The test set directly follows the validation set and contains `id_test_nb`.

Because the size of the training set is always an interger multiple of the batch size, the training set might have less than `id_valid_start` files.
The `id_valid_start-1` last files right before `id_valid_start` might thus be completely ignored by the training process.

A last set exists, the demo set, which is a subset of the test set. This is convenient for generating and listening quickly to a few known sentences after a training. By default it is the first 10 sentences of the test set.
`id_test_demostart` can be used to select the starting index (relative to the test set) in order to chose where the demo set starts within the test set.

#### File access and shapes
To represent multiple files in a directory, file paths are usually defined with a wildcard (e.g. `waveforms/*.wav`).
Because input and output data of the network (lab and cmp files) are saved in raw `float32` format, without header, it is not possible to know the actual dimensions of the data
inside each file.
In Percival, the trick is to specify the shape of the data as a suffix of the file path, e.g. `spectrum/*.spec:(-1,129)`. this suffix will be used, as is, to reshape the numpy array using `np.reshape(.)`.

#### Batches

A batch has a shape: [size, length, features_dim], that represent the number of samples in the batches, the number of time frames in the batch and the feature dimensionality, respectively. Because Theano/Lasagne needs a "channel" dimension (as in pictures), batches' shape often become temporarily [size, 1, length, features_dim] so that the last two dimensions define a picture of size [length, features_dim].


### Results repeatability

The seed of the random number generator is forced by default.
In order to have non-deterministic runs, replace the line `np.random.seed(123)` in utils.py by `pass`

When the seed is forced, and while using the CPU, all training runs are supposed to output the exact same numerical results, no matter the model.

When using the GPU, training a model that is based on Fully-Connected layers or Recurrent layers only should also output the exact same numerical results for each run.
However, when using the GPU and 2D Convolutional layers, the results will differ (c.f. [Lasagne thread](https://github.com/Lasagne/Lasagne/issues/6) ).
It should be possible to make them repeatable by uncommenting some lines in `setenv.sh` under repeatability section, at the expense of some computational time.


### Cloning

Managing multiple experiments in parallel, fixing bugs and try new ideas while keeping track of all the changes this can imply can become quite problematic. There is a simple script in Percival `clone.sh` that allows you to clone the whole code source and place it in an experiment directory:
```
$ ./clone.sh /path/to/experiment1
```
or clone and run the `run.py` script file directly:
```
$ ./clone.sh /path/to/experiment1 bash ../percivaltts/setenv.sh python ../percivaltts/run.py
```
when a script is runned using `clone.sh`, the working directory is `/path/to/experiment1/out` so that all of the experiment results are in this `out` sub-directory and all the code is in a separate sub-directory `percivaltts`
You can also replace the `bash` command by a script for submitting Sun Grid Engine (SGE) job.
The `Makefile` has also all the corresponding commands (clone, run, clone_run_grid, etc.)


### Notes
Before the waveform synthesis, there is no post-processing of the generated spectral amplitudes (e.g. no formant enhancement or similar).

### Author/Contributor
Gilles Degottex <gad27@cam.ac.uk>

### Contact
To raise bug reports, questions, suggestions, etc. please use the [issue managment](https://github.com/gillesdegottex/percivaltts/issues) only.
