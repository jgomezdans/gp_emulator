#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A class for dumping a forest of emulators
"""
__author__  = "J Gomez-Dans"
__version__ = "1.0 (1.12.2013)"
__email__   = "j.gomez-dans@ucl.ac.uk"

import os
import shelve

import numpy as np

from GaussianProcess import GaussianProcess
from multivariate_gp import MultivariateEmulator

class EmulatorStorage ( object ):
    def __init__ ( self, fname ):
        self.fname = fname
    def dump_emulator ( self, emulator, tag ):
        """
        Dumps an emulator to storage file (a Python pickle). We need a "tag"
        in order to recover the emulator.
        
        Parameters
        ----------
        emulator: emulator object
            Either a scalar or a multiple output emulator
        tag: list
            A list that helps as a tag
        """
        if os.path.exists ( self.fname ):
            # File exists, so open and get a handle to it
            emulators = shelve.open ( self.fname )
        else:
            print "File doesn't exist, creating it"
            emulators = shelve.open ( self.fname )

        if type( tag ) != str:
            tag = repr ( tag )#.strip("()").split(",")
            

        if isinstance ( emulator, MultivariateEmulator ):
            emulator_dict = { "X": emulator.X_train, \
                              "y": emulator.y_train, \
                              "basis_functions": \
                                 emulator.basis_functions, \
                              "n_pcs": emulator.n_pcs, \
                              "thresh": emulator.thresh, \
                              "hyperparams": \
                                 emulator.hyperparams }

        elif isinstance ( emulator, GaussianProcess ):
            emulator_dict = { "input": emulator.inputs, \
                              "targets": emulator.targets, \
                              "theta": emulator.theta }

        emulators[tag] = emulator_dict
        emulators.close() # Flush!
        
    def _declutter_key ( self, tag ):
        return repr(tuple(tag))
    
    def get_emulator ( self, tag ):
        """
        Recovers an emulator from storage, and returns it to 
        the calller
        """
        if os.path.exists ( self.fname ):
            # File exists, so open and get a handle to it
            emulators = shelve.open ( self.fname )
        else:
            raise IOError:
                print "File doesn't exist!"
                

        if type(tag) != str:
            tag = self._declutter_key ( tag )
            
        if emulators[tag].has_key ( "basis_functions" ):
            gp = MultivariateEmulator ( \
                X = emulators[tag]["X"], \
                y=emulators[tag]["y"], \
                hyperparams = emulators[tag]["hyperparams"], 
                basis_functions = emulators[tag]["basis_functions"] )
        else:
            gp = GaussianProcess ( \
                emulators[tag]["inputs"], \
                emulators[tag]["targets"] )
            gp._set_params ( emulators[tag]['theta'] )
        emulators.close()
        return gp