#!/usr/bin/env python

from distutils.core import setup

setup(name='gp_emulator',
      version='1.5.5',
      description='A Python GaussianProcess emulator software package',
      classifiers=[
	'Development Status :: 4 - Beta',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Environment :: Console'],
      author='J Gomez-Dans',
      author_email='j.gomez-dans@ucl.ac.uk',
      url='http://github.com/jgomezdans/gp_emulator',
      packages=['gp_emulator'],
     )
