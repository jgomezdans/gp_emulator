# GP emulators

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.45259.svg)](https://doi.org/10.5281/zenodo.45259)


## Gaussian process (GP) emulators for Python

### Author  
J Gómez-Dans <j.gomez-dans@ucl.ac.uk>
<p><img src="http://www.multiply-h2020.eu/wp-content/uploads/2018/08/multiply_banner_2018_klein.jpg" align="center" \></p>
<p><img src="https://www.nceo.ac.uk/wp-content/themes/nceo/assets/images/logos/img_logo_purple.svg" align="left" />

<img src="http://www.esa.int/esalogo/images/logotype/img_colorlogo_darkblue.gif" scale="20%" align="right" />
</p>



This repository contains an implementation of GPs for emulation of radiative transfer models in Python. This particular implementation is focused on emulating univariate output models (e.g. emulating reflectance or radiance for a single sensor band) and multivariate outputs (e.g. emulating reflectance/radiance over the entire solar reflective domain). The emulators also calculate the gradient of the emulated model and the Hessian.

You can install the software with either [conda]<https://docs.anaconda.com/anaconda/>

    conda install -c jgomezdans gp_emulators

or using `pip`...

    pip install gp_emulator

or just clone/download the repository and invoke the `setup.py` script:

    python setup.py install

The only requirements are (if memory serves) numpy and scipy. 

At some point, pointers to a library of emulators of popular vegetation and atmospheric RT codes will be provided.

### Citation


If you use this code, we would be grateful if you cited the following paper:

    Gómez-Dans, J.L.; Lewis, P.E.; Disney, M. Efficient Emulation of Radiative Transfer Codes Using Gaussian Processes and Application to Land Surface Parameter Inferences. Remote Sens. 2016, 8, 119. <DOI:10.3390/rs8020119> <http://www.mdpi.com/2072-4292/8/2/119>

