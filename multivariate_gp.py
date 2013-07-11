#/usr/bin/env python

import numpy as np

from GaussianProcess import GaussianProcess


class MultivariateEmulator ( object ):

    def __init__ ( self, X=None, y=None, basis_functions=None, thresh=0.95 ):
        if X is None and y is None:
            pass
        else:
            print "Decomposing the input dataset into basis functions...",
            self.calculate_decomposition ( X, thresh )
            print "Done!\n ====> Using %d basis functions" % self.n_pcs
            self.train_emulators ( X, y )



    def calculate_decomposition ( self, X, thresh ):
    
        U, s, V = np.linalg.svd ( X, full_matrices = True )
        pcnt_var_explained = s.cumsum()/s.sum()
        self.basis_functions = V [ pcnt_var_explained <= 0.95 ]
        self.n_pcs = np.sum ( pcnt_var_explained <= 0.95 )
    
    def train_emulators ( self, X, y ):
        self.emulators = []
        train_data = self.compress ( X )
        self.hyperparams = np.zeros ( ( 2, self.n_pcs ) )
        for i in xrange ( self.n_pcs ):
            print "\tFitting GP for basis function %d" % i
            self.emulators.append ( GaussianProcess ( X[i], y ) )
            self.hyperparams[ :, i] = \
                self.emulators[i].learn_hyperparameters ( n=5 )

        
    def compress ( self, X ):
        return X.dot ( X, self.basis_functions.T ).T

    def predict ( self, y, deriv=True ):
        """Infers the value of the full function (as well as its derivative)"""
        fwd = 0.
        y = np.atleast_2d ( y ) # Just in case
        if deriv:
            deriv = np.zeros ( ( y.shape[1], self.basis_functions.shape[0] ) )
        for i in xrange ( self.n_pcs ):
            pred_mu, pred_var, grad = self.emulators[i].predict ( y )
            fwd += pred_mu.T.dot ( self.basis_functions )
            if deriv:
                deriv += grad.T.dot ( self.basis_functions )
        if deriv:
            return fwd.squeeze(), deriv
        else:
            return fwd.squeeze()

if __name__ == "__main__":
    # read LUT to test this bits
    