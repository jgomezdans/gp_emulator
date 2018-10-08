#!/usr/bin/env python
import codecs
import os
import re
from os import path
from setuptools import setup


this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'README.md'), "rb") as f:
    long_description = f.read().decode('utf-8')

def read(*parts):
    with codecs.open(os.path.join(this_directory, *parts), 'r') as fp:
        return fp.read()
    
def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(name='gp_emulator',
      version=find_version("gp_emulator", "__init__.py"),
      description='A Python Gaussian Process emulator software package',
      classifiers=[
	'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Environment :: Console'],
      author='J Gomez-Dans',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author_email='j.gomez-dans@ucl.ac.uk',
      url='http://github.com/jgomezdans/gp_emulator',
      packages=['gp_emulator'],
      install_requires=["numpy", "scipy"]
     )
