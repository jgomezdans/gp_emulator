import sys


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss

from Py6S import *
from prosail import run_prosail
from gp_emulator import GaussianProcess, lhd



def inverse_transform ( x ):
    """Inverse transform the PROSAIL parameters"""
    x_out = x*1.
    # Cab, posn 1
    x_out[1] = -100.*np.log ( x[1] )
    # Cab, posn 2
    x_out[2] = -100.*np.log ( x[2] )
    # Cw, posn 4
    x_out[4] = (-1./50.)*np.log ( x[4] )
    #Cm, posn 5
    x_out[5] = (-1./100.)*np.log ( x[5] )
    # LAI, posn 6
    x_out[6] = -2.*np.log ( x[6] )
    # ALA, posn 7
    x_out[7] = 90.*x[7]
    return x_out

def cpled_model ( x, wavebands, sza, saa, vza, vaa, day, month, do_trans=True ):
    """A coupled land surface/atmospheric model, predicting TOA refl from
    land surface parameters and atmospheric composition parameters. This
    function provides estimates of TOA refl for 
    * A particular date
    * A particular illumination geometry
    
    The underlying land surface reflectance spectra is simulated using
    PROSAIL, and the atmospheric calculations are done with 6S. The coupling
    assumes a Lambertian land surface

        The input parameter ``x`` is a vector with the following components:
        
        * ``n``
        * ``cab``
        * ``car``
        * ``cbrown``
        * ``cw``
        * ``cm``
        * ``lai``
        * ``ala``
        * ``bsoil``
        * ``psoil``
        * ``aot_550``
        * ``o3_conc``
        * ``cwc``

    """
    from prosail import run_prosail
    # Invert parameter LAI
    
    if do_trans:
        x = inverse_transform ( x )
    ################# surface refl with prosail #####################
    surf_refl = np.c_[ np.arange(400, 2501)/1000., \
        run_prosail(x[0], x[1], x[2], x[3], \
        x[4], x[5], x[6], x[7], 0, x[8], x[9], 0.01, sza, vza, vaa, 2 )]
    ################# Atmospheric modelling with 6s ##################
    o3 = x[-2]
    cwc = x[-1]
    aot = x[-3]
    s, retvals = atcorr ( wavebands, o3, cwc, aot, surf_refl, sza, saa, vza, \
            vaa, day, month )

    ################ Per band coupling ###############################
    refl_toa = []
    for i, sixs_output in enumerate ( retvals ):
        rho_surf = np.interp( wavebands[i], \
                surf_refl[:,0], surf_refl[:,1])
        refl_toa.append ( to_refltoa ( sixs_output, rho_surf ) )
    return np.array ( refl_toa )


def to_refltoa ( o, surf_refl ):
    """Convert surface reflectance to TOA reflectance"""
    tgasm = o.transmittance_global_gas.total
    ainr = o.atmospheric_intrinsic_reflectance
    transmittances = o.transmittance_total_scattering.total
    sast = o.spherical_albedo.total
    refl_toa = (surf_refl*(transmittances*tgasm - ainr*sast) + ainr)/\
            (1.+sast*surf_refl)
    return refl_toa

def do_atcorr ( s, apparent_refl ):
    tgasm = s.outputs.transmittance_global_gas.total
    ainr = s.outputs.atmospheric_intrinsic_reflectance
    transmittances = s.outputs.transmittance_total_scattering.total
    sast = s.outputs.spherical_albedo.total
    rog = apparent_refl / tgasm
    rog = ( rog - ainr/tgasm ) / transmittances
    rog = rog/ ( 1 + rog*sast )
    return rog

def atcorr ( wavebands, o3, cwc, aot, surf_refl, sza, saa, vza, vaa, day, month, \
            atcorr_refl = None ):
    import os
    if os.path.exists ( "/home/ucfajlg/Data/python/SixS/sixsV1.1" ):
        s = SixS( path="/home/ucfajlg/Data/python/SixS/sixsV1.1")
    else:
        s = SixS ( )  
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(cwc, o3 ) 
    s.aero_profile = AeroProfile.Continental
    s.aot550 = aot 
    #s.wavelength = Wavelength( wv ) 
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian ( surf_refl )
    s.geometry = Geometry.User()
    s.geometry.month = month
    s.geometry.day = day
    s.geometry.solar_z = sza
    s.geometry.solar_a = saa
    s.geometry.view_z = vza
    s.geometry.view_a = vaa
    if atcorr_refl is not None:
        s.atmos_corr = AtmosCorr.AtmosCorrLambertianFromReflectance( atcorr_refl )
    #s.run()
    retval = SixSHelpers.Wavelengths.run_wavelengths ( s, wavebands, n=2 )
    #    if atcorr_refl is not None:
    #    print "Rog: %g" % do_atcorr ( s, atcorr_refl )
    #    print s.outputs.fulltext
    return s,retval[1]
