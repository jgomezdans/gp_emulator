import numpy as np
import os

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


def test_gp_value(datadir):
    fname = datadir("tip_emulator_real.pkl")
    with open(fname, 'rb') as fp:
        gp = pickle.load(fp, encoding="latin-1")
    pred, unc, jac = gp.predict(np.atleast_2d([0.7, 2, 0.18, 4]))
    assert np.allclose( pred, 0.2392899)

def test_gp_variance(datadir):
    fname = datadir("tip_emulator_real.pkl")
    with open(fname, 'rb') as fp:
        gp = pickle.load(fp, encoding="latin-1")
    pred, unc, jac = gp.predict(np.atleast_2d([0.7, 2, 0.18, 4]))
    assert np.allclose(unc, 0.0006086)
    

def test_gp_jacobian(datadir):
    fname = datadir("tip_emulator_real.pkl")
    with open(fname, 'rb') as fp:
        gp = pickle.load(fp, encoding="latin-1")
    pred, unc, jac = gp.predict(np.atleast_2d([0.7, 2, 0.18, 4]))
    assert np.allclose(np.array([[0.62780396, 0.01440244, 0.03961139, 0.00089103]]),
                       jac)
    
def test_simple_fit(datadir):
    def f(x):
        """The function to predict."""
        return x * np.sin(x)
    X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    # Observations
    y = f(X).ravel()
    gp = gp_emulator.GaussianProcess(X, y)
    gp._set_params(np.array([ -1.03267399,   3.09976281, -34.31716978]))
    ypred, _, _ = gp.predict(np.atleast_2d(np.linspace(0, 10, 1000)).T)
    ytest = np.array([ 0.06315748,  0.39347948,  0.84242247,  1.28759591,  1.51707233,
        1.28415404,  0.41618402, -1.05591459, -2.81248657, -4.27631767,
       -4.79331401, -3.91388683, -1.64259409,  1.49545022,  4.63881107,
        6.94388076,  7.91938801,  7.56717339,  6.28340496,  4.61604946])
    assert np.allclose(ytest, ypred[::50], rtol=0.1)
