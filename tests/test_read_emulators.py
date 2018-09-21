import os
import sys
sys.path.insert(0, '../')

import numpy as np


from pytest import fixture
from distutils import dir_util

import pickle

import gp_emulator

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


def test_scalar_gp_read(datadir):
    fname = datadir("isotropic_TERRA_emulators_0950_b01.npz")
    gp = gp_emulator.GaussianProcess(emulator_file=fname)
    y_pred, y_sigma, y_grad = gp.predict(np.atleast_2d(gp.inputs[200]))
    
    assert np.allclose( y_pred, gp.targets[200], atol=0.05)

def test_spectral_gp_read(datadir):
    fname = datadir("prosail_000_000_0000.npz")
    gp = gp_emulator.MultivariateEmulator(dump=fname)
    assert np.allclose(gp.predict(gp.y_train[200, :])[0],
                       gp.X_train[200, :], atol=0.05)

