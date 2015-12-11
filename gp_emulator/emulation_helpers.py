# -*- coding: utf-8 -*-
import multiprocessing
import numpy as np
import scipy.stats as ss
from lhd import lhd
from GaussianProcess import GaussianProcess
from multivariate_gp import MultivariateEmulator

def create_training_set ( parameters, minvals, maxvals, n_train=200 ):
    """Creates a traning set for a set of parameters specified by 
    ``parameters`` (not actually used, but useful for debugging
    maybe). Parameters are assumed to be uniformly distributed
    between ``minvals`` and ``maxvals``. ``n_train`` input parameter
    sets will be produced, and returned with the actual distributions
    list. The latter is useful to create validation sets.

    Parameters
    -------------
    parameters: list
        A list of parameter names
    minvals: list
        The minimum value of the parameters. Same order as ``parameters``
    maxvals: list
        The maximum value of the parameters. Same order as ``parameters``
    n_train: int
        How many training points to produce

    Returns
    ---------
    The training set and a distributions object that can be used by
    ``create_validation_set``
    """

    distributions = []
    for i,p in enumerate(parameters):
        distributions.append ( ss.uniform ( loc=minvals[i], \
                            scale=(maxvals[i]-minvals[i] ) ) )
    samples = lhd ( dist=distributions, size=n_train )
    return samples, distributions

def create_validation_set ( distributions, n_validate=500 ):
    """Creates a validation set of ``n_validate`` vectors, using the
    ``distributions`` list."""
    validate  = []
    for d in distributions:
        validate.append ( d.rvs( n_validate ))
    validate = np.array ( validate ).T
    return validate


def create_emulator_validation ( f_simulator, parameters, minvals, maxvals, 
                                n_train, n_validate, do_gradient=True, 
                                thresh=0.98, n_tries=5, args=(), n_procs=None ):


    """A method to create an emulator, given the simulator function, the
    parameters names and boundaries, the number of training input/output pairs. 
    The function will also provide an independent validation dataset, both for 
    the valuation of the function and its gradient. The gradient is calculated
    using finite differences, so it is a bit ropey.
    
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
        
        
    Returns
        The GP object, the validation input set, the validation output set, the
        emulated validation set, the emulated gradient set. If the gradient
        validation is also done, it will also return the gradient validation 
        using finite differences.
        
    """
    
    # First, create the training set, using the appropriate function from
    # above...
    samples, distributions = create_training_set ( parameters, minvals, maxvals, 
                                                  n_train=n_train )
    # Now, create the validation set, using the distributions object we got
    # from creating the training set
    validate  = []
    for d in distributions:
        validate.append ( d.rvs( n_validate ))
    validate = np.array ( validate ).T
    
    # We have the input pairs for the training and validation. We will now run
    # the simulator function
    
    if n_procs is None:
        training_set = map  ( f_simulator, [( (x,)+args) for x in samples] )
        validation_set = map  ( f_simulator, [( (x,)+args) for x in validate] )
        
    else:
        pool = multiprocessing.Pool ( processes = n_procs)
        
        
        training_set = pool.map  ( f_simulator, [( (x,)+args) for x in samples] )
        validation_set = pool.map  ( f_simulator, [( (x,)+args) for x in validate] )
    training_set = np.array ( training_set ).squeeze()
    validation_set = np.array ( validation_set )

    if training_set.ndim == 1:
        gp = GaussianProcess( samples, training_set )
        gp.learn_hyperparameters( n_tries = n_tries )
    else:
        gp = MultivariateEmulator(X=training_set , \
                        y=samples, thresh=thresh, n_tries=n_tries )
    
    X = [ gp.predict ( np.atleast_2d(x) ) 
                        for x in validate ] 
    if len ( X[0] ) == 2:
        emulated_validation = np.array ( [ x[0] for x in X] )
        emulated_gradient = np.array ( [ x[1] for x in X] )
    elif len ( X[0] ) == 3:
        emulated_validation = np.array ( [ x[0] for x in X] )
        emulated_gradient = np.array ( [ x[2] for x in X] )
    # Now with gradient... Approximate with finite differences...
    
    

    if do_gradient:
        val_set = [( (x,)+args) for x in validate]
        validation_gradient = []
        delta = [(maxvals[j] - minvals[j])/10000. 
                    for j in xrange(len(parameters)) ]
        delta = np.array ( delta )
        for i, pp in enumerate( val_set ):
            xx0 = pp[0]*1.
            grad_val_set = []
            f0 = validation_set[i]
            df = []
            for j in xrange ( len ( parameters ) ):
                xx = xx0*1
                xx[j] = xx0[j] + delta[j]
                grad_val_set.append ( xx  )
                df.append ( f_simulator ( ( (xx,) + args ) ) )
            df = np.array ( df )
            try:
                validation_gradient.append (  (df-f0)/delta )
            except ValueError:
                validation_gradient.append (  (df-f0)/delta[:, None] )
                
        return gp, validate, validation_set, np.array(validation_gradient), \
            emulated_validation, emulated_gradient.squeeze()
    else:
        return gp, validate, validation_set,  emulated_validation, \
            emulated_gradient

