# /usr/bin/env python
"""A multivariate Gaussian Process emulator through principal components compression.
This code provides a convenient way of emulating a multivariate model output (e.g. a
time series, or a spatial/temporal stack) by reducing it through principal component
analysis. Each of the selected PCs is then modelled through a Gaussian Process

Usage
-----

We need two arrays: one that stores the multivariate model output(`X`) and one that
stores the model parameters that produced said outputs (`y`). The sizes of these
arrays are important: `X` is `(N_training x N_full)` (i.e, `N_training` rows and
`N_full` elements in each column). `y` is `( N_train, N_params )`. Then simply

.. code-block:: python

   emulator = MultivariateEmulator( X=X, y=y )
   prediction, der_prediction = emulator.predict ( y[0] ) # Say

Saving emulators for future use
--------------------------------

If you want to save an emulator for re-use, you can do that easily by using the
`dump_emulator` method with a filename. You can then instantiate the
`MultivariateEmulator` class setting only the `dump` keyword in the constructor
to be the saved filename, and the emulator will be recreated.

"""

# import h5py

import numpy as np

from .GaussianProcess import GaussianProcess


class MultivariateEmulator(object):
    def __init__(
        self,
        dump=None,
        X=None,
        y=None,
        hyperparams=None,
        model="",
        sza=0,
        vza=0,
        raa=0,
        thresh=0.98,
        n_tries=5,
    ):
        """Constructor

        The constructor takes an array of model outputs `X`, and a vector
        of the parameters that served as the inputs for this model runs `y`.
        The sizes of these two arrays are `( N_training, N_full )` for `X` and
        `( N_train, N_params )` for `y`. `X` is decomposed into its PCs and
        the first `self.n_pcs` that explain `thresh` of the variance are
        selected. If `hyperparams` is set to `None`, normal training of the
        individual `self.n_pcs` GPs is carried out. If `hyperparams` is
        specified and it is the right shape `( N_params + 2, self.n_pcs )`,
        then there's no need for training the emulators (might be more
        efficient).

        Parameters
        ----------
        dump: str
            A filename that will be read to populate X, y and hyperparams
        X: array ( N_train, N_full )
            The modelled output array for training
        y: array (N_train, N_param )
            The corresponding training parameters for `X`
        hyperparams: array ( N_params + 2, N_PCs )
            The hyperparameters for the relevant GPs
        thresh: float
            The threshold at where to cutoff the percentage of
            variance explained.
        """
        if dump is not None:
            if X is None and y is None:
                if dump.find(".h5") > 0 or dump.find(".hdf5") > 0:
                    raise IOError("I can't be bothered working with HDF5 files")
                elif dump.find(".npz"):
                    f = np.load(dump)
                    X = f["X"]
                    y = f["y"]
                    hyperparams = f["hyperparams"]
                    thresh = f["thresh"]
                    if "basis_functions" in dict(f):
                        basis_functions = f["basis_functions"]
                        n_pcs = f["n_pcs"]
                        f.close()
                else:
                    pass
            else:
                raise ValueError("You specified both a dump file and X and y")
        else:
            if X is None or y is None:
                raise ValueError("Need to specify both X and y")
            else:
                assert X.shape[0] == y.shape[0]
                assert X.ndim == 2
                assert y.ndim == 2
                basis_functions = None

        self.X_train = X
        self.y_train = y
        self.thresh = thresh
        if basis_functions is None:
            print("Decomposing the input dataset into basis functions...")
            self.calculate_decomposition(X, thresh)
            print("Done!\n ====> Using %d basis functions" % self.n_pcs)
            basis_functions = self.basis_functions
            n_pcs = self.n_pcs

        self.n_pcs = n_pcs
        self.basis_functions = basis_functions

        if hyperparams is not None:
            assert (y.shape[1] + 2 == hyperparams.shape[0]) and (
                self.n_pcs == hyperparams.shape[1]
            )
        self.train_emulators(X, y, hyperparams=hyperparams, n_tries=n_tries)

    def dump_emulator(self, fname, model_name, sza, vza, raa):
        """Save emulator to file for reuse

        Saves the emulator to a file (`.npz` format) for reuse.

        Parameters
        ----------
        fname: str
            The output filename

        """
        sza = int(sza)
        vza = int(vza)
        raa = int(raa)
        if fname.find(".npz") < 0 and (
            fname.find("h5") >= 0 or fname.find(".hdf") >= 0
        ):
            raise IOError("I can't be bothered working with HDF5 files")
        else:
            np.savez_compressed(
                fname,
                X=self.X_train,
                y=self.y_train,
                hyperparams=self.hyperparams,
                thresh=self.thresh,
                basis_functions=self.basis_functions,
                n_pcs=self.n_pcs,
            )
            print("Emulator safely saved")

    def calculate_decomposition(self, X, thresh):
        """Does PCA decomposition

        This simply does a PCA decomposition using the SVD. Note that
        if `X` is very large, more efficient methods of doing this
        might be required. The number of PCs to retain is selected
        as those required to estimate `thresh` of the total variance.

        Parameters
        -----------
        X: array ( N_train, N_full )
            The modelled output array for training
        thresh: float
            The threshold at where to cutoff the percentage of
            variance explained.
        """
        U, s, V = np.linalg.svd(X, full_matrices=False)
        pcnt_var_explained = s.cumsum() / s.sum()
        self.basis_functions = V[pcnt_var_explained <= thresh]
        self.n_pcs = np.sum(pcnt_var_explained <= thresh)

    def train_emulators(self, X, y, hyperparams, n_tries=2):
        """Train the emulators

        This sets up the required emulators. If necessary (`hypeparams`
        is set to None), it will train the emulators.

        X: array ( N_train, N_full )
            The modelled output array for training
        y: array (N_train, N_param )
            The corresponding training parameters for `X`
        hyperparams: array ( N_params + 2, N_PCs )
            The hyperparameters for the relevant GPs
        """
        self.emulators = []
        train_data = self.compress(X)
        self.hyperparams = np.zeros((2 + y.shape[1], self.n_pcs))
        for i in range(self.n_pcs):

            self.emulators.append(GaussianProcess(np.atleast_2d(y), train_data[i]))
            if hyperparams is None:
                print("\tFitting GP for basis function %d" % i)
                self.hyperparams[:, i] = self.emulators[i].learn_hyperparameters(
                    n_tries=n_tries
                )[1]
            else:
                self.hyperparams[:, i] = hyperparams[:, i]
                self.emulators[i]._set_params(hyperparams[:, i])

    def hessian(self, x):
        """A method to approximate the Hessian. This method builds on the fact
        that the spectral emulators are a linear combination of individual
        emulators. Therefore, we can calculate the Hessian of the spectral
        emulator as the sum of the individual products of individual Hessians
        times the spectral basis functions.
        """
        the_hessian = np.zeros((len(x), len(x), len(self.basis_functions[0])))
        x = np.atleast_2d(x)
        for i in range(self.n_pcs):
            # Calculate the Hessian of the weight
            this_hessian = self.emulators[i].hessian(x)
            # Add this hessian contribution once it's been scaled by the
            # relevant basis function.
            the_hessian = (
                the_hessian
                + this_hessian.squeeze()[:, :, None] * self.basis_functions[i]
            )
        return the_hessian

    def compress(self, X):
        """Project full-rank vector into PC basis"""
        return X.dot(self.basis_functions.T).T

    def predict(self, y, do_unc=True, do_deriv=True):
        """Prediction of input vector

        The individual GPs predict the PC weights, and these are used to
        reconstruct the value of the function at a point `y`. Additionally,
        the derivative of the function is also calculated. This is returned
        as a `( N_params, N_full )` vector (i.e., it needs to be reduced
        along axis 1)

        Parameters:
        y: array
            The value of the prediction point
        do_deriv: bool
            Whether derivatives are required or not
        do_unc: bool
            Whether to calculate the uncertainty or not

        Returns:

        A tuple with the predicted mean, predicted variance and
        patial derivatives. If any of the latter two elements have
        been switched off by `do_deriv` or `do_unc`, they'll be returned
        as `None`.
        """
        fwd = np.zeros(self.basis_functions[0].shape[0])
        y = np.atleast_2d(y)  # Just in case
        deriv = None
        unc = None
        if do_deriv:
            deriv = np.zeros((y.shape[1], self.basis_functions.shape[1]))
        if do_unc:
            unc = np.zeros_like(fwd)
        for i in range(self.n_pcs):
            pred_mu, pred_var, grad = self.emulators[i].predict(
                y, do_unc=do_unc, do_deriv=do_deriv
            )
            fwd += pred_mu * self.basis_functions[i]
            if do_deriv:
                deriv += np.matrix(grad).T * np.matrix(self.basis_functions[i])
            if do_unc:
                unc += pred_var * self.basis_functions[i]
        try:
            return fwd.squeeze(), unc.squeeze(), deriv
        except AttributeError:
            return fwd.squeeze(), unc, deriv


###if __name__ == "__main__":
    #### read LUT to test this bits

    ###f = np.load("test_LUT.npz")
    ###angles = f["angles"]
    ###train_params = f["train_params"]
    ###train_brf = f["train_brf"]

    ###def unpack(params):
        ###"""Input a dictionary and output keys and array"""
        ###inputs = []
        ###keys = np.sort(list(params.keys()))
        ###for i, k in enumerate(keys):
            ###inputs.append(params[k])
        ###inputs = np.array(inputs).T
        ###return inputs, keys

    ###def pack(inputs, keys):
        ###"""Input keys and array and output dict"""
        ###params = {}
        ###for i, k in enumerate(keys):
            ###params[k] = inputs[i]
        ###return params

    ###train_paramsoot, keys = unpack(train_params.tolist())

    ###mv_em = MultivariateEmulator(X=train_brf, y=train_paramsoot)
    ###y_test = np.array([1., 0.5, 0.5, 1.65, 0.5, 0.5, 0.73, 0.89, 0.44, 0.42, 0.1])
    ###hypers = mv_em.hyperparams
    ###mv_em2 = MultivariateEmulator(X=train_brf, y=train_paramsoot, hyperparams=hypers)
    ###y_arr = y_test * 1
    ###for i in range(8):
        ###y_arr[-1] = 0.05 + 0.1 * i
        ###plt.plot(mv_em.predict(y_arr)[0], "-r", lw=2)
        ###plt.plot(mv_em2.predict(y_arr)[0], "-k", lw=1)
    ###mv_em.dump_emulator("emulator1.npz")
    ###plt.figure()
    ###new_em = MultivariateEmulator(dump="emulator1.npz")
    ###for i in range(8):
        ###y_arr[-1] = 0.05 + 0.1 * i
        ###plt.plot(mv_em.predict(y_arr)[0], "-r", lw=2)
        ###plt.plot(new_em.predict(y_arr)[0], "-k", lw=1)
