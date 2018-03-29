import numpy as np
from scipy.interpolate import interp1d as i1d
from matplotlib import pyplot as plt
from linear_growth_factor import E
#from WMB_scratch import Plognormal
from scipy.special import gamma as gamma




###############################################################################################################
##IGM stuff

################################
###Mean of Gaussian IGM model###
################################

def DM_(zs=1,nz=201):
    
    """
    Returns the DM for a cosmological source at redshift zs
    nz is the number of intermediate redshifts to be included
    in the trapz integration.
    
    Inputs: 
    
    zs (source redshift)
    nz (number of intermediate redshifts for trapz integration)
    
    Outputs:
    
    dm (dm in pc cm^-3)
    """
    #####
    #nz=1201 #may delete this
    #####
    def integrand(z):
        return (1+z)/E(z)
    
    z = np.linspace(0,zs,nz) #intermediate redshifts for integration

    i = np.array([integrand(i) for i in z]) #integrand at said redshifts
    
    dm = 950.5 * np.trapz(i,z)
    
    return dm


################################
###variance of Gaussian model###
################################

mps_zs = np.arange(26.)/10 #list of redshifts we have matter power spectrum for

#Calculate discrete version of Integral I (see yinzhe's notes)
#Matter power spectrum computed from public code CAMB with Planck best-fitting cosmological parameters
# Planck 2015 results. XIII. Cosmological parameters: http://adsabs.harvard.edu/abs/2016A%26A...594A..13P

I = []

for j in mps_zs:
    data = np.loadtxt('./lin_mp_files/planck_lin_mp%s.dat'%(j))
    ktilde = data[:,0]
    powspec = data[:,1]  
    I.append(np.trapz(ktilde*powspec,ktilde))
    
#Create interpolated function I(z)

class Icontinuous:
    zmin = np.min(mps_zs)
    zmax = np.max(mps_zs)
    func = i1d(mps_zs,I)
    
def Icont(z):
    """
    Continuous version of Integral, I (see yin-zhe's notes)
    """
    if (z<=Icontinuous.zmax) & (z>=Icontinuous.zmin):
        
        out = Icontinuous.func(z)
        
    else:
        out = 0.0
        
    return out

# Calculate variance of gaussian model by integrating over continuous I

def variance(zs=1,nz=201):
    """
    Returns the sight-line to sightline variance in DM for source redshift zs.
    
    nz is the number of intermediate redshifts to be included
    in the trapz integration.
    
    Inputs: 
    
    zs (source redshift)
    nz (number of intermediate redshifts for trapz integration)
    
    Outputs:
    
    v (variance)
    """
    
    #####
    #nz=1201 #may delete this
    #####

    def integrandv(z):
        return ((1+z)**2)*Icont(z)/E(z)
    
    z = np.linspace(0,zs,nz) #intermediate redshifts for integration
    
    i = np.array([integrandv(i) for i in z]) #integrand at said redshift
    
    v = 47.91 * np.trapz(i,z)
    
    return v


####################################################
###Gaussian IGM Probability Distribution Function###
####################################################

def Prob_IGM(dm=100,zs=1):
    """
    Returns P(zs|dm) for fixed dm and list of source redshifts zs

    Alternatively

    Returns P(dm|zs) for fixed source redshift zs and list of dms.

    Inputs:

    zs (source redshift)

    dm (dispersion measure)

    Returns:

    P (probability P(zs|dm) or P(dm|zs) with shape like the input list)
    
    """
    
    if isinstance(zs,(list,np.ndarray,tuple)): #check if redshifts are a list (dm fixed)
        
        mean = np.zeros_like(zs)
        var = np.zeros_like(zs)

        for i in range(len(zs)):

            mean[i]=DM_(zs[i])
            var[i]=variance(zs[i])
            
    else: #otherwise dm is list and redshift is fixed
            
        mean = np.array(DM_(zs))
        var = np.array(variance(zs))
        #print mean,var

    P = ((2.*np.pi*var)**(-1./2)) * (np.exp((-1.*((dm-mean)**2/(2.*var)))))
    
    #find places where P is infinite and replace with zero
    indices = np.isfinite(P)
    P[~indices] = 0 #tilde is the NOT operator, which returns the inverse of indices
    
    return P


########################################################################################################
##host stuff
 
def Observed_Host_PDF(probfunc,dm,zs):
    """
    Transforms a reference frame host galaxy PDF into an observed PDF.

    Transformation necessary due to redshift effects.

    See the transformation of coordinates equations in my notebook.

    INPUTS:

    probfunc (the probability distribution function to use)
    
    dm (list of dms to feed probability function)

    zs (frb source redshift)   
    """
    
    #if probfunc == Plognormal: #special case for Plognormal, as it requires mean, std inputs
    #    mean=5
    #    std=0.5
    #    PDF = probfunc(dm*(1+zs),mean,std)*(1+zs)

    #else:
    PDF = probfunc(dm*(1+zs))*(1+zs)

    return PDF


#########################################################################################################
##convolution stuff

############################
############################
####Convolution Function####
############################
############################

def Convolve(IGM_Prob_Func, Host_Prob_Func,DM,zs):
    """
    Convolves an IGM PDF function with a Host Galaxy PDF function.

    Inputs:

    IGM_Prob_Func  -- Model for IGM PDF. Must have inputs DM,z
    Host_Prob_Func -- Model for Host PDF at rest frame. Must have DM input.
    DM             -- List of DMs to calculate P(DM|z) for.
    zs             -- FRB source redshift.

    Outputs:

    conv           -- Convolution of IGM & Host Probability functions for DMs in DM at zs
    newDMs         -- New list of DMs which go with conv (artifact of np.convolve)
    """
    
    #convolve two functions
    conv = np.convolve(IGM_Prob_Func(DM,zs),Observed_Host_PDF(Host_Prob_Func,DM,zs),mode='full')
    #increase range of DMs to equal length of the convolution via concatenation
    newDMs = np.r_[DM,DM[1:]+np.max(DM)]
    
    return conv,newDMs


def NormConv(Convolution_Matrix,DMlist):
    """
    Normalises a 2D array of P(DM|zs) probability distributions along the DM axis.
    Utilises the trapezoidal rule for a uniform grid -- https://en.wikipedia.org/wiki/Trapezoidal_rule
    Note: see my notebook 05/03/18 for a written explanation of the normalisation. For an old version of
    this NormConv (for debugging purposes) see NormConv in WMB_Model_Notebook.ipynb
    """

    if len(DMlist)==Convolution_Matrix[0].shape[0]: #check to make sure DM list input was used to make convolution matrix: DM dimension should match
	print 'True'
    else:
	print "Error: DM list doesn't match matrix"

    a = DMlist[0]-float(DMlist[1]-DMlist[0])/2 # edge of first dm bin
    b = DMlist[-1]+float(DMlist[1]-DMlist[0])/2 # edge of last dm bin
    N = len(DMlist)                             # number of dm bins

    DeltaX  = (float(b)-float(a))/float(N)      # Delta X (see wikipedia page)

    print Convolution_Matrix.shape

    N_z = Convolution_Matrix.shape[0] #number of redshifts there are P(DM|z) curves to be normalised for

    for z in range(N_z): #implement trapezoidal rule over each redshift

        norm=(DeltaX/2)*((2*np.sum(Convolution_Matrix[z]))-(Convolution_Matrix[z][0]+Convolution_Matrix[z][-1])) #trapezoidal rule: get area under P(DM|z) curve
        Convolution_Matrix[z]/=norm                                                                              #normalise P(DM|z)

        #hack: divide by the sum to make the cumulative sum unity (05/03/18)
        #without doing this, low DM cuts of P(z|DM) sum to too large values
        #note: this means you are not normalising using real units...
        
        Convolution_Matrix[z]/=np.sum(Convolution_Matrix[z])

    return Convolution_Matrix

def ErlangDist(x,k=2,l=1):
    """
    The Erlang distribution for variable x
    
    (see http://iopscience.iop.org/article/10.1088/0004-637X/738/1/19/pdf)
    """
    
    f = (l**k) * (x**(k-1)) * np.exp(-1.*l*x) / (gamma(k))
    
    return f

def NormTranspose(Convolution_Matrix,zlist,erlang=False):

    """
    Normalises a 2D array of P(DM|zs) probability distributions along the redshift axis to obtain P(z|DM).
    
    Utilises Bayes' Theorem: P(z|DM)=P(DM|z)P(z)/P(DM)
    
    Note: if erlang=False, the probability of an FRB having a certain redshift is constant out to any redshift.
    
    If erlang=True, P(z) is assumed to be the Erlang distribution, as has been postulated for GRBs.
    
    Utilises the trapezoidal rule for a uniform grid -- https://en.wikipedia.org/wiki/Trapezoidal_rule
    Note: see my notebook 05/03/18 for a written explanation of the normalisation.
    
    
    """
    
    #input check
    
    if len(zlist)==Convolution_Matrix[:,0].shape[0]: #check to make sure redshift list input was used to make convolution matrix: DM dimension should match
        print 'True'
    else:
        print "Error: redshift list doesn't match matrix"
        
        
    #if applicable, apply erlang distribution before renormalising
    
    N_z = Convolution_Matrix.shape[0] #number of redshifts there are P(DM|z) curves for
    
    if erlang==True:
        print 'ERLANG DISTRIBUTION FOR FRBS IMPLEMENTED'
        for i in range(len(zlist)): #loop over redshifts
            z = zlist[i]            #extract redshift
            Convolution_Matrix[i,:]*=ErlangDist(z) #P(DM|z)*P(z)
        
    
    #perform renormalisation
    
    a = zlist[0]-float(zlist[1]-zlist[0])/2  # edge of first z bin
    b = zlist[-1]+float(zlist[1]-zlist[0])/2 # edge of last z bin
    N = len(zlist)                             # number of z bins
    

    print a,b,N
    
    DeltaX  = (float(b)-float(a))/float(N)      # Delta X (see wikipedia page)
    
    N_DM = Convolution_Matrix.shape[1] #number of DMs there are P(z|DM) curves to be normalised for
    
    for d in range(N_DM): #implement trapezoidal rule over each DM
        
        norm=(DeltaX/2)*((2*np.sum(Convolution_Matrix[:,d]))-(Convolution_Matrix[:,d][0]+Convolution_Matrix[:,d][-1])) #trapezoidal rule: get area under P(DM|z) curve
        Convolution_Matrix[:,d]/=norm                                                                             #normalise P(DM|z)
        
        #hack: divide by the sum to make the cumulative sum unity (05/03/18)
        #without doing this, low DM cuts of P(z|DM) sum to too large values
        #note: this means you are not normalising using real units...
        
        Convolution_Matrix[:,d]/=np.sum(Convolution_Matrix[:,d])                                                                             #normalise P(DM|z)

        
    return Convolution_Matrix


#########################################################################################
##Extras##

def find_nearest(array,value):
    """
    function to find the location of the nearest value in an array to the chosen value
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    idx=(np.abs(array-value)).argmin()
    return array[idx]


def FindErrorRange(pdf,x,ref,p):
    """
    Function to find x-values associated with the minimum and maximum bounds for a probability
    density function (PDF), centered on a reference x-value, within which p% of the probability
    lies.
    
    https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
    
    Example: for PDF P(x),
             with a reference x-value of the mode of the PDF:
             FindErrorRange will return the x-values within which
             99.7% (3sigma) of the probability lies, distributed
             about the mode,
             if p = 0.997
    
    INPUTS:
    
    pdf : (array-like) the probability distribution to analyse
    
    x   : (array-like) the x range associated with the input PDF
    
    ref : (value)      reference value (i.e. x-value of mode/median of pdf)
                       to take as center of error bars (NOTE: may not be at center
                       if errors are not symmetric due to PDF behaviour)
                       
    p   : (value: 0-1) percentage of values to lie within the band about ref.
    
    """
    
    ref_idx = np.where(x==ref)[0] #get index of the reference value
    
    cumsum = np.cumsum(pdf) #create cumulative sum
    
    print cumsum[0],cumsum[ref_idx]-p/2,cumsum[ref_idx]+p/2
    
    #if there aren't sufficient points at beginning of distribution then error bars will not be symmetrical
    if cumsum[0]>cumsum[ref_idx]-(p/2):
        print 'nonsymmetric'
        
        #Min of error bar will be first x value
        #remainder = cumsum[0]-(cumsum[ref_idx]-(p/2))
        
        xMin = x[0]
        
        #Maximum of error bar:
        
        #max value is where CDF = p
        CDFmax = p
        #find nearest discrete point on CDF to upper bound
        CDF_nearest = find_nearest(cumsum,CDFmax)
        #find idx of nearest discrete point
        CDF_nearest_idx = np.where(cumsum==CDF_nearest)[0]
        #find whether nearest point lies above or below true value:
        if CDF_nearest<CDFmax:
            low_max = CDF_nearest_idx
            high_max = CDF_nearest_idx+1
        elif CDF_nearest>CDFmax:
            low_max=CDF_nearest_idx-1
            high_max=CDF_nearest_idx
        #find CDF values for boundaries
        CDF_bounds = [cumsum[low_max],cumsum[high_max]]
        #find x values for boundaries
        x_bounds = [x[low_max],x[high_max]]        
        #find where true max lies wrt discrete range
        maxfrac = (CDFmax-CDF_bounds[0])/(CDF_bounds[1]-CDF_bounds[0])
        #calculate true x-value
        xMax = x_bounds[0]+maxfrac*(x_bounds[1]-x_bounds[0])
        
        return [xMin,xMax]
    
    #if probability curve falls to zero too fast at end of dist. then error bars will also be nonsymmetrical
    elif 1.0<cumsum[ref_idx]+(p/2): 
        print 'nonsymmetric'
        
        #remainder from overflow (think as though water level)
        remainder = cumsum[ref_idx]+(p/2)-1.0
        
        #Max of error bar: where cumulative sum reaches 1       
        xMax_idx = np.where(cumsum==np.max(cumsum))[0][0]
        xMax = x[xMax_idx]
        print 'max x value: ',xMax
        #Minimum of error bar: 
        
        #find minimum value of CDF
        CDFmin = cumsum[ref_idx]-(p/2)-remainder
        #find nearest discrete point on CDF to lower bound
        CDF_nearest = find_nearest(cumsum,CDFmin)
        #find idx of nearest discrete point
        CDF_nearest_idx = np.where(cumsum==CDF_nearest)[0]
        #find whether nearest point lies above or below true value:
        print 'CDF minimum value: ',CDFmin
        if CDF_nearest<CDFmin:
            low_min = CDF_nearest_idx
            high_min = CDF_nearest_idx+1
        elif CDF_nearest>CDFmin:
            low_min=CDF_nearest_idx-1
            high_min=CDF_nearest_idx
        #fix in case minimum is found to be less than zeroth index of array
        if low_min<0:
            print 'Fix needed'
            xMin=x[0]
        else:
            print 'boundary indexes: ',low_min,high_min
            #find CDF values for boundaries
            CDF_bounds = [cumsum[low_min],cumsum[high_min]]
            print 'CDF minimum boundary values: ', CDF_bounds
            #find x values for boundaries
            x_bounds = [x[low_min],x[high_min]]
            print 'Xmin boundary values: ', x_bounds
            #find where true min lies wrt discrete range
            minfrac = (CDFmin-CDF_bounds[0])/(CDF_bounds[1]-CDF_bounds[0])
            print 'fraction: ', minfrac
            #calculate true x-value
            xMin = x_bounds[0]+minfrac*(x_bounds[1]-x_bounds[0])

        return [xMin,xMax]
        
    #else errors will be symmetrical about the reference position
    else:
        print 'symmetric'
        
        CDFmin = cumsum[ref_idx]-(p/2)         #true minimum CDF value in desired range
        CDFmax = cumsum[ref_idx]+(p/2)         #true maximum CDF value in desired range
        print 'CDF reference point: ',cumsum[ref_idx]
        print 'CDF boundaries: ',CDFmin,CDFmax
        
        #find nearest discrete CDF points
        CDF_nearest = [find_nearest(cumsum,CDFmin),find_nearest(cumsum,CDFmax)] 
        print 'CDF discretes: ',CDF_nearest
        
        #get indexes of discrete points
        CDF_nearest_idx = [np.where(cumsum==i)[0][0] for i in CDF_nearest]
        print 'CDF discretes indexes: ',CDF_nearest_idx
        
        #find boundaries around discrete pdf points
        if CDF_nearest[0]<CDFmin:
            low_min = CDF_nearest_idx[0]
            low_max = CDF_nearest_idx[0]+1
        elif CDF_nearest[0]>CDFmin:
            low_min = CDF_nearest_idx[0]-1
            low_max = CDF_nearest_idx[0]

        if CDF_nearest[1]<CDFmax:
            high_min = CDF_nearest_idx[1]
            high_max = CDF_nearest_idx[1]+1
        elif CDF_nearest[1]>CDFmax:
            high_min = CDF_nearest_idx[1]-1
            high_max = CDF_nearest_idx[1]
            
        CDF_bounds_idx = [[low_min,low_max],[high_min,high_max]]
        print 'CDF index boundaries: ', CDF_bounds_idx
        
        #find CDF values for boundaries
        CDF_bounds = [[cumsum[low_min],cumsum[low_max]],[cumsum[high_min],cumsum[high_max]]]
        print 'CDF boundaries: ', CDF_bounds
        
        #find x values for boundaries
        x_bounds = [[x[low_min],x[low_max]],[x[high_min],x[high_max]]]
        print 'x-array boundaries: ',x_bounds
        
        #find where the true min/maxes lie wrt. the discrete ranges
        #note: see 06/03/18 notes for derivation
        minfrac = (CDFmin-CDF_bounds[0][0])/(CDF_bounds[0][1]-CDF_bounds[0][0])
        maxfrac = (CDFmax-CDF_bounds[1][0])/(CDF_bounds[1][1]-CDF_bounds[1][0])
        
        print 'fractions: ',minfrac,maxfrac
        
        #calculate true x-values
        #note: see 06/03/18 notes for derivation
        xMin = x_bounds[0][0]+minfrac*(x_bounds[0][1]-x_bounds[0][0])
        xMax = x_bounds[1][0]+maxfrac*(x_bounds[1][1]-x_bounds[1][0])
                
        return [xMin,xMax]
