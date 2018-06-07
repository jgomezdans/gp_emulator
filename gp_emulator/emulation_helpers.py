# -*- coding: utf-8 -*-
import multiprocessing
import numpy as np
import scipy.stats as ss
from .lhd import lhd
from .GaussianProcess import GaussianProcess
from .multivariate_gp import MultivariateEmulator


def create_single_band_emulators(emulator, band_pass, n_tries=15):
    """This function creates per band emulators from the full-spectrum
    emulator. It requires passing an array of band pass functions of
    shape `(n_bands x n_wavelengths)` (e.g. (7 x 2101 for the 7 MODIS
    bands in the solar reflective domain 400-2500 nm with 1 nm spacing).
    The number of wavelengths should be compatible with that of the
    spectral emulator. Further, the sum of the band pass or spectral
    response function should be one (e.g. `np.sum(band_pass[i]) == 1`).

    Parameters
    ----------
    emulator: gp_emulator.MultivariateEmulator object
        A `gp_emulator.MultivariateEmulator` object that already exists.
    band_pass: iter
        A 2D array of size `(n_bands x n_wavelengths)`. The sum over the
        second dimension of the array should evaluate to 1.
    n_tries: int
        The number of random starts to train the single band emulator.
        Might be decreased for convenirnce to 5 or 10.

    Returns
    -------
    A list of `n_bands` `GaussianProcess` emulators.
    """

    n_bands = band_pass.shape[0]
    if band_pass.dtype == np.bool:
        x_train_pband = [emulator.X_train[:, band_pass[i, :]].mean(axis=1)
                         for i in range(n_bands)]
    else:
        band_pass = band_pass / (band_pass.sum(axis=1)[:, None])
        x_train_pband = [np.sum(emulator.X_train[:, :] * band_pass[i, :],
                                axis=1) for i in range(n_bands)]

    x_train_pband = np.array(x_train_pband)
    emus = []
    for i in range(n_bands):
        gp = GaussianProcess(emulator.y_train[:] * 1.,
                             x_train_pband[i, :])
        gp.learn_hyperparameters(n_tries=n_tries)
        emus.append(gp)
    return emus


def create_training_set(parameters, minvals, maxvals, fix_params=None, n_train=200):
    """Creates a traning set for a set of parameters specified by
    ``parameters`` (not actually used, but useful for debugging
    maybe). Parameters are assumed to be uniformly distributed
    between ``minvals`` and ``maxvals``. ``n_train`` input parameter
    sets will be produced, and returned with the actual distributions
    list. The latter is useful to create validation sets.

    It is often useful to add extra samples for regions which need to
    be carefully evaluated. We do this by adding a `fix_params` parameter
    which should be a dictionary indexing the parameter name, its fixed
    value, and the number of additional samples that will be drawn.

    Parameters
    -------------
    parameters: list
        A list of parameter names
    minvals: list
        The minimum value of the parameters. Same order as ``parameters``
    maxvals: list
        The maximum value of the parameters. Same order as ``parameters``
    fix_params: dictionary
        A diciontary indexed by the parameter name. Each item will have a
        tuple indicating the fixed value of the parameter, and how many
        extra LHS samples are required
    n_train: int
        How many training points to produce

    Returns
    ---------
    The training set and a distributions object that can be used by
    ``create_validation_set``
    """

    distributions = []
    for i, p in enumerate(parameters):
        distributions.append(
            ss.uniform(loc=minvals[i], scale=(maxvals[i] - minvals[i]))
        )
    samples = lhd(dist=distributions, size=n_train)

    if fix_params is not None:
        # Extra samples required
        for k, v in fix_params.items():
            # Check whether they key makes sense
            if k not in parameters:
                raise ValueError(
                    "You have specified '%s', which is" % k
                    + " not in the parameters list"
                )

            extras = fix_parameter_training_set(
                parameters, minvals, maxvals, k, v[0], v[1]
            )
            samples = np.r_[samples, extras]

    return samples, distributions


def fix_parameter_training_set(
    parameters, minvals, maxvals, fixed_parameter, value, n_train
):
    """Produces a set of extra LHS samples where one parameter
    has been fixed to a single value, whereas all other parameters
    take their usual boundaries etc."""
    from copy import deepcopy  # groan

    parameters = deepcopy(parameters)
    minvals = deepcopy(minvals)
    maxvals = deepcopy(maxvals)
    fix_param = parameters.index(fixed_parameter)
    reduced_parameters = [p for p in parameters if p != fixed_parameter]
    minvals.pop(fix_param)
    maxvals.pop(fix_param)
    dummy_param = np.ones(n_train) * value
    distributions = []
    for i, p in enumerate(reduced_parameters):
        distributions.append(
            ss.uniform(loc=minvals[i], scale=(maxvals[i] - minvals[i]))
        )
    samples = lhd(dist=distributions, size=n_train)

    extra_array = np.insert(samples, fix_param, dummy_param, axis=1)
    return extra_array


def create_validation_set(distributions, n_validate=500):
    """Creates a validation set of ``n_validate`` vectors, using the
    ``distributions`` list."""
    validate = []
    for d in distributions:
        validate.append(d.rvs(n_validate))
    validate = np.array(validate).T
    return validate


def create_emulator_validation(
    f_simulator,
    parameters,
    minvals,
    maxvals,
    n_train,
    n_validate,
    do_gradient=True,
    fix_params=None,
    thresh=0.98,
    n_tries=5,
    args=(),
    n_procs=None,
):

    """A method to create an emulator, given the simulator function, the
    parameters names and boundaries, the number of training input/output pairs.
    The function will also provide an independent validation dataset, both for
    the valuation of the function and its gradient. The gradient is calculated
    using finite differences, so it is a bit ropey.

    In order to better sample some regions of parameter space easily (you can
    change the underlying pdf of the parameters for LHS, but that's overkill)
    you can also add additional samples where one parameter is set to a fixed
    value, and an LHS design for all the other parameters is returned. This
    can be done usign the `fix_params` keyword.

    Parameters
    ------------
    f_simulator: function
        A function that evaluates the simulator. It should take a single
        parameter which will be made out of the input vector, plus whatever
        other extra arguments one needs (stored in ``args``).
    parameters: list
        The parameter names
    minvals: list
        The minimum value of the parameters
    maxvals: list
        The maximum value of the parameters
    n_train: int
        The number of training samples
    n_validate: int
        The number of validation samples
    thresh: float
        For a multivariate output GP, the threshold at which to cut the
        PCA expansion.
    n_tries: int
        The number of tries in the GP hyperparameter stage. The more the better,
        but also the longer it will take.
    args: tuple
        A list of extra arguments to the model
    do_gradient: Boolean
        Whether to do a gradient validation too.
    fix_params: dictionary
        A dictionary that allows the training set to be extended by fixing one
        or more parameters to one value, while still doing an LHS on the
        remaining parameters. Each parameter has a 2-element tuple, indicating
        the value and the number of extra samples.

    Returns
        The GP object, the validation input set, the validation output set, the
        emulated validation set, the emulated gradient set. If the gradient
        validation is also done, it will also return the gradient validation
        using finite differences.

    """

    # First, create the training set, using the appropriate function from
    # above...
    samples, distributions = create_training_set(
        parameters, minvals, maxvals, n_train=n_train, fix_params=fix_params
    )
    # Now, create the validation set, using the distributions object we got
    # from creating the training set
    validate = []
    for d in distributions:
        validate.append(d.rvs(n_validate))
    validate = np.array(validate).T

    # We have the input pairs for the training and validation. We will now run
    # the simulator function

    if n_procs is None:
        training_set = list(map(f_simulator, [((x,) + args) for x in samples]))
        validation_set = list(map(f_simulator, [((x,) + args) for x in validate]))

    else:
        pool = multiprocessing.Pool(processes=n_procs)

        training_set = pool.map(f_simulator, [((x,) + args) for x in samples])
        validation_set = pool.map(f_simulator, [((x,) + args) for x in validate])
    training_set = np.array(training_set).squeeze()
    validation_set = np.array(validation_set)

    if training_set.ndim == 1:
        gp = GaussianProcess(samples, training_set)
        gp.learn_hyperparameters(n_tries=n_tries)
    else:
        gp = MultivariateEmulator(
            X=training_set, y=samples, thresh=thresh, n_tries=n_tries
        )

    X = [gp.predict(np.atleast_2d(x)) for x in validate]
    if len(X[0]) == 2:
        emulated_validation = np.array([x[0] for x in X])
        emulated_gradient = np.array([x[1] for x in X])
    elif len(X[0]) == 3:
        emulated_validation = np.array([x[0] for x in X])
        emulated_gradient = np.array([x[2] for x in X])
    # Now with gradient... Approximate with finite differences...

    if do_gradient:
        val_set = [((x,) + args) for x in validate]
        validation_gradient = []
        delta = [(maxvals[j] - minvals[j]) / 10000. for j in range(len(parameters))]
        delta = np.array(delta)
        for i, pp in enumerate(val_set):
            xx0 = pp[0] * 1.
            grad_val_set = []
            f0 = validation_set[i]
            df = []
            for j in range(len(parameters)):
                xx = xx0 * 1
                xx[j] = xx0[j] + delta[j]
                grad_val_set.append(xx)
                df.append(f_simulator(((xx,) + args)))
            df = np.array(df)
            try:
                validation_gradient.append((df - f0) / delta)
            except ValueError:
                validation_gradient.append((df - f0) / delta[:, None])

        return (
            gp,
            validate,
            validation_set,
            np.array(validation_gradient),
            emulated_validation,
            emulated_gradient.squeeze(),
        )
    else:
        return gp, validate, validation_set, emulated_validation, emulated_gradient
