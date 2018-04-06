from setuptools import setup, find_packages

with open('README.md') as f:
    long_description =  f.read()

setup(name='percival-tts',
    version='0.9.0',
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

    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    package_data={'percival-tts': ['README.md', 'LICENSE.md']},
    zip_safe=False
    )
