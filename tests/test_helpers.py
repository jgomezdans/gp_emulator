#!/usr/bin/env python
import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

import datetime
import numpy as np

import pytest
from pytest import fixture
from distutils import dir_util

from gp_emulator import MultivariateEmulator
from gp_emulator.emulation_helpers import integrate_passbands
from gp_emulator.emulation_helpers import create_single_band_emulators

@fixture
def datadir(tmpdir, request):
    '''
    Fixture responsible for locating the test data directory and copying it
    into a temporary directory.
    Taken from  http://www.camillescott.org/2016/07/15/travis-pytest-scipyconf/
    '''
    filename = request.module.__file__
    test_dir = os.path.dirname(filename)
    data_dir = os.path.join(test_dir, 'data') 
    dir_util.copy_tree(data_dir, str(tmpdir))

    def getter(filename, as_str=True):
        filepath = tmpdir.join(filename)
        if as_str:
            return str(filepath)
        return filepath

    return getter

def test_integratepassbands(datadir):

    f = np.load(datadir("prosail_000_000_0000.npz"))
    spectrum = f['X']
    
    # Process the S2A spectral response functions into something useable
    srf = np.loadtxt(datadir("S2A_SRS.csv"), skiprows=1,
                     delimiter=",")[100:, :]
    srf[:, 1:] = srf[:, 1:]/np.sum(srf[:, 1:], axis=0)
    srf_land = srf[:, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]]
    W = spectrum[:, None, :]*srf_land.T
    test_rho_s2a = W.sum(axis=2).T
    rho_s2a = integrate_passbands(np.atleast_2d(spectrum), srf_land.T)
    assert np.allclose(test_rho_s2a, rho_s2a)
    
def test_singlebandemulators(datadir):
    
    gp = MultivariateEmulator(dump=datadir("prosail_000_000_0000.npz"))
        # Process the S2A spectral response functions into something useable
    srf = np.loadtxt(datadir("S2A_SRS.csv"), skiprows=1,
                     delimiter=",")[100:, :]
    srf[:, 1:] = srf[:, 1:]/np.sum(srf[:, 1:], axis=0)
    srf_land = srf[:, [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]].T


    gps = create_single_band_emulators(gp, np.atleast_2d(srf_land[0, :]), n_tries=5)
    
    rho_pred, _, _ = gps[0].predict(np.atleast_2d(gp.y_train[10, :]))
    np.allclose(rho_pred, 0.10881985)
