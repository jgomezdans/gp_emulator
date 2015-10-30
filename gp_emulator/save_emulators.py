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
import h5py

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
        
    def get_keys ( self ):
        """Print out the keys"""

        if os.path.exists ( self.fname ):
            # File exists, so open and get a handle to it
            emulators = shelve.open ( self.fname )
        else:
            raise IOError ("File %s doesn't exist!" % self.fname )
        keys = emulators.keys()
        emulators.close()
        return keys
        
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
            raise IOError ("File %s doesn't exist!" % self.fname )
                

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

def convert_npz_to_hdf5 ( npz_file, hdf5_file ):
    """A utility to convert from the old and time honoured .npz format to the
    new HDF5 format."""
    
    f = np.load ( npz_file )
    X = f[ 'X' ]
    y = f[ 'y' ]
    hyperparams = f[ 'hyperparams' ]
    thresh = f[ 'thresh' ]
    basis_functions = f[ 'basis_functions' ]
    n_pcs = f[ 'n_pcs' ]
    f.close()
    fname = os.path.basename ( npz_file )
    fname = fname.replace("xx", "")
    sza, vza, raa, model = fname.split("_")
    sza = int ( float(sza) )
    vza = int ( float(vza) )
    raa = int ( float(raa) )
    model = model.split(".")[0]
    try:
        f = h5py.File (hdf5_file, 'r+')
    except IOError:
        print "The file %s did not exist. Creating it" % hdf5_file
        f = h5py.File (hdf5_file, 'w')
        f
    group = '%s_%03d_%03d_%03d' % ( model, sza, vza, raa )
    if group in f.keys():
        raise ValueError, "Emulator already exists!"
    f.create_group ("/%s" % group )
    f.create_dataset ( "/%s/X_train" % group, data=X, compression="gzip" )
    f.create_dataset ( "/%s/y_train" % group, data=y, compression="gzip"  )
    f.create_dataset ( "/%s/hyperparams" % group, data=hyperparams, compression="gzip"  )
    f.create_dataset ( "/%s/basis_functions" % group, data=basis_functions, 
                      compression="gzip"  )
    f.create_dataset ( "/%s/thresh" % group, data=thresh  )
    f.create_dataset ( "/%s/n_pcs" % group, data=n_pcs)
    f.close()
    print "Emulator safely saved"

