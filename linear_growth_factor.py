"""
calculate the linear growth factor for perturbations, which affects how the power spectrum changes with redshift.

see eqs (2,3,10,11) in Mo and White, 2002, MNRAS 336,112

http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?2002MNRAS.336..112M&amp;data_type=PDF_HIGH&amp;whole_paper=YES&amp;type=PRINTER&amp;filetype=.pdf

"""

import numpy as np
from cosmo_consts import cosmoconsts as consts

def E(z):
    """
    Redshift dependence of the Hubble constant (i.e. H(z)=H0*E(z) )
    """
    
    omega_lambda_0 = consts.omega_lambda_0
    omega_0        = consts.omega_0
    omega_m_0      = consts.omega_m_0
    Ez = np.sqrt((omega_lambda_0)+((1-omega_0)*(1+z)**2)+(omega_m_0*(1+z)**3))
    return Ez

def OMEGA_M(z):
    """
    The energy density of matter at any redshift
    """

    omega_m_0 = consts.omega_m_0
    omega_m = omega_m_0 * (1+z)**3 / E(z)**2
    return omega_m

def H(z):
    """
    The Hubble constant at any redshift
    """

    H0 = consts.H0
    Hz = H0*E(z)
    return Hz

def OMEGA_LAMBDA(z):
    """
    The dark energy density at any redshift
    """

    omega_lambda_0 = consts.omega_lambda_0
    omega_lambda = omega_lambda_0 / E(z)**2
    return omega_lambda

def G(z):
    """
    Function within D(z)
    """

    o_m = OMEGA_M(z)
    o_l = OMEGA_LAMBDA(z)
    gz = 5./2 * o_m * ((o_m**(4./7))-(o_l)+((1+o_m/2)*(1+o_l/70)))**-1
    return gz

def D(z):
    """
    The linear growth factor for fluctuations
    """

    gz = G(z)
    g0 = G(0)
    Dz = gz/(g0*(1+z))
    return Dz


