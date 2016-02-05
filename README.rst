GP emulators
==============

.. image:: https://zenodo.org/badge/19469/jgomezdans/gp_emulator.svg
      :target: https://zenodo.org/badge/latestdoi/19469/jgomezdans/gp_emulator

:Info: Gaussian process (GP) emulators for Python
:Author: J Gomez-Dans <j.gomez-dans@ucl.ac.uk>
:Date: $Date: 2015-03-17 16:00:00 +0000  $
:Description: README file


.. image:: http://www.nceo.ac.uk/images/NCEO_logo_lrg.jpg
   :scale: 50 %
   :alt: NCEO logo
   :align: right
   
.. image:: http://www.esa.int/esalogo/images/logotype/img_colorlogo_darkblue.gif
   :scale: 20 %
   :alt: ESA logo
   :align: left

This repository contains an implementation of GPs for emulation of radiative transfer
models in Python. This particular implementation is focused on emulating univariate
output models (e.g. emulating reflectance or radiance for a single sensor band)
and multivariate outputs (e.g. emulating reflectance/radiance over the entire
solar reflective domain). The emulators also calculate the gradient of the
emulated model and the Hessian.

You can install the software with 

        python setup.py install

The only requirements are (if memory serves) numpy and scipy.

At some point, pointers to a library of emulators of popular vegetation and
atmospheric RT codes will be provided.

Citation
----------

If you use this code, we would be grateful if you cited the following paper:
	
	Gómez-Dans, J.L.; Lewis, P.E.; Disney, M. Efficient Emulation of Radiative Transfer Codes Using Gaussian Processes and Application to Land Surface Parameter Inferences. Remote Sens. 2016, 8, 119. DOI:`10.3390/rs8020119 <http://www.mdpi.com/2072-4292/8/2/119>`_
