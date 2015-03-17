GP emulators
==============

:Info: Gaussian process (GP) emulators for Python
:Author: J Gomez-Dans <j.gomez-dans@ucl.ac.uk>
:Date: $Date: 2015-03-17 16:00:00 +0000  $
:Description: README file

This repository contains an implementation of GPs for emulation in Python. Although many different implementations exist, this particular one deals with fast GP predictions for large number of input vectors, where the training data sets are typically modest (e.g. less than 300 samples). Access to the emulation's partial derivatives and Hessian matrix is calculated, and training is also taken care of.

You can install with 

        python setup.py install

The only requirements are (if memory serves) numpy and scipy.
