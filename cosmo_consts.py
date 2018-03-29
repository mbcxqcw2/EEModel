class cosmoconsts:
    """
    Cosmology constants that get used repeatedly
    """
    
    H0= 67.3                                   #Hubble constant at z=0 [km/s/Mpc]                   (Planck 2016b)
    h=H0/100                                   #reduced Hubble constant                             (Mo and White)
    #rho_crit_0 = 2.775 * h**2 * 10**11         #critical density of Universe at z=0
    omega_m_0 = 0.309#0.313                          #energy density of matter at z=0               (Planck 2016b)
    #rho_m_0 = rho_crit_0 * omega_m_0           #the comoving matter density at z=0
    omega_lambda_0 = 0.691#0.687                     #dark energy density at z=0                    (Planck 2016b)
    omega_0 = 1.                               #total energy density of the Universe (always 1)     (Mo and White)
    c = 299792458.                             #the speed of light in m/s
