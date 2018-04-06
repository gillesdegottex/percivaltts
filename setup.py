import os
import re
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

try:
    # obtain version string from __init__.py
    # Read ASCII file with builtin open() so __version__ is str in Python 2 and 3
    with open(os.path.join(here, 'percival-tts', '__init__.py'), 'r') as f:
        version = re.search('__version__ = \'(.*)\'', f.read()).groups()[0]
except Exception:
    version = ''

print('Percival-TTS version: '+version)

with open('README.md') as f:
    long_description =  f.read()

setup(name='percival-tts',
    version=version,
    description='Percival and the quest for the holy waveform - Acoustic model for DNN-based TTS',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/gillesdegottex/percival-tts',
    author='Gilles Degottex',
    author_email='gad27@cam.ac.uk',
    license='Apache License (2.0)',

    # Classifiers help users find your project by categorizing it.
    #
    # For a list of valid classifiers, see
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish
        'License :: OSI Approved :: Apache Software License',
    ],

    packages=['percival-tts', 'percival-tts/external/Lasagne/lasagne', 'percival-tts/external/merlin', 'percival-tts/external/pulsemodel', 'percival-tts/external/pfs'], #find_packages(), #exclude=['docs', 'tests']
    data_files=[('.',['LICENSE.md'])],
    package_data={'percival-tts': ['Makefile', 'clone.sh', 'setenv*.sh', 'external/*.py', 'external/*.hed', 'tests/slt_arctic_merlin_test.tar.gz']},
    # include_package_data=True, # TODO Needed ?
    zip_safe=False
    )
