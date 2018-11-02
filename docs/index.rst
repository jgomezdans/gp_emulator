.. gp_emulator documentation master file, created by
   sphinx-quickstart on Mon Oct 29 16:05:31 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Emulators for complex models using Gaussian Processes in Python: `gp_emulator`
=====================================================================================================================

The `gp_emulator` library provides a simple pure Python implementations of Gaussian Processes (GPs), with a view of using them as **emulators** of complex computers code. In particular, the library is focused on radiative transfer models for remote sensing, although the use is general. The GPs can also be used as a way of regressing or interpolating datasets.

If you use this code, please cite both the code and the paper that describes it.

* JL Gómez-Dans, Lewis PE, Disney M. Efficient Emulation of Radiative Transfer Codes Using Gaussian Processes and Application to Land Surface Parameter Inferences. Remote Sensing. 2016; 8(2):119. `DOI:10.3390/rs8020119 <https://doi.org/10.3390/rs8020119>`_
* José Gómez-Dans & Professor Philip Lewis. (2018, October 12). jgomezdans/gp_emulator (Version 1.6.5). Zenodo. `DOI:10.5281/zenodo.1460970 <http://doi.org/10.5281/zenodo.1460970>`_

Development of this code has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 687320, under project `H2020 MULTIPLY <http://www.multiply-h2020.eu/>`_.





.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction.rst
   quickstart.rst
   emulating_rt_model.rst
   gp_regression.rst
   user_reference.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
