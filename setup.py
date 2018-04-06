from setuptools import setup

def readme():
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
      packages=find_packages(exclude=['contrib', 'docs', 'tests']),
      zip_safe=False)
