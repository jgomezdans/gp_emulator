Gaussian Process Emulators
****************************

Introduction
==============

Often, complex and numerically demanding computer codes are required in inverse modelling tasks. Such models might need to be invoked repeatedly as part of a minimisition task, or in order to provide some numerically-integrated quantity. This results in many applications being rendered impractical as the codes are too slow.

The concept of an emulator is simple: for a given model, let's provide a function that given the same inputs are the original model, gives *the same* output. Clearly, we need to qualify "the same", and maybe downgrade the expectations to "a very similar output". Ideally, with some metric of uncertainty on the prediction. So an emulator is just a fast, surrogate to a more established code.

Gaussian processes (GPs) have been used for this task for years, as they're very flexible throught he choice of covariance function that can be used, but also work remarkably well with models that are nonlinear and that have a reasonable number of inputs (10s). 

We have used these techniques for emulation radiative transfer models used in Earth Observation, and we even wrote a nice paper about it: `Gomez-Dans et al (2016) <http://dx.doi.org/10.3390/rs8020119>`_. Read it, it's pure gold.


Installing the package
============================

The package works on Python 3. With a bit of effort it'll probably work on Python 2.7. The only dependencies are `scipy <http://www.scipy.org/>`_ and `numpy <http://www.numpy.org/>`_ To install, use either **conda**: ::

    conda install -c jgomezdans gp_emulator

or **pip**: ::

    pip install gp_emulator
    
or just clone or download the source repository and invoke `setup.py` script: ::

    python setup.py install
    
