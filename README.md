# GP emulators

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.45259.svg)](https://doi.org/10.5281/zenodo.45259)


## Gaussian process (GP) emulators for Python

### Author  
J Gomez-Dans <j.gomez-dans@ucl.ac.uk>


![NCEO logo](https://www.nceo.ac.uk/wp-content/themes/nceo/assets/images/logos/img_logo_purple.svg)

![ESA logo](http://www.esa.int/esalogo/images/logotype/img_colorlogo_darkblue.gif)

This repository contains an implementation of GPs for emulation of radiative transfer models in Python. This particular implementation is focused on emulating univariate output models (e.g. emulating reflectance or radiance for a single sensor band) and multivariate outputs (e.g. emulating reflectance/radiance over the entire solar reflective domain). The emulators also calculate the gradient of the emulated model and the Hessian.

You can install the software with

    python setup.py install

The only requirements are (if memory serves) numpy and scipy. It does have a vague requirement for h5py, but you can disable it, as that's still work in progress.

At some point, pointers to a library of emulators of popular vegetation and atmospheric RT codes will be provided.

Citation
--------

If you use this code, we would be grateful if you cited the following paper:

    Gómez-Dans, J.L.; Lewis, P.E.; Disney, M. Efficient Emulation of Radiative Transfer Codes Using Gaussian Processes and Application to Land Surface Parameter Inferences. Remote Sens. 2016, 8, 119. <DOI:10.3390/rs8020119> <http://www.mdpi.com/2072-4292/8/2/119>

