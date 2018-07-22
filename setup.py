import os
import re
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

try:
    # obtain version string from __init__.py
    # Read ASCII file with builtin open() so __version__ is str in Python 2 and 3
    with open(os.path.join(here, 'percivaltts', '__init__.py'), 'r') as f:
        version = re.search('__version__ = \'(.*)\'', f.read()).groups()[0]
except Exception:
    version = ''

print('percivaltts version: '+version)

with open('README.md') as f:
    long_description =  f.read()

setup(name='percivaltts',
    version=version,
    description='Percival and the quest for the holy waveform - Acoustic model for DNN-based TTS',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.com/gillesdegottex/percivaltts',
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

    packages=['percivaltts', 'percivaltts/external/Lasagne/lasagne', 'percivaltts/external/merlin', 'percivaltts/external/pulsemodel', 'percivaltts/external/pfs'], #find_packages(), #exclude=['docs', 'tests']
    data_files=[('.',['LICENSE.md'])],
    package_data={'percivaltts': ['Makefile', 'clone.sh', 'setenv*.sh', 'external/*.py', 'external/*.hed', 'tests/slt_arctic_merlin_test.tar.gz']},
    include_package_data=True,
    zip_safe=False,

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/gillesdegottex/percivaltts/issues',
        'Source': 'https://github.com/gillesdegottex/percivaltts',
    },
)
