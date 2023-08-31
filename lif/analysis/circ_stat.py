# copyright Errol Lloyd 2015

import numpy as np

import scipy.stats as stats
import scipy.misc as misc

#######
# test from same distribution??  Maybe to iceberged geniculate units
 



def mean(r, theta, modes=2, bin = False, bwdth=0, ci=0.95, single_angle=False):
    """
    Returns, for a dataset of responses (spikes/s) for given
    orientations, values for the mean vector.

    r:    responses
    theta:    corresponding orientations
    modes:    number of modes; ie, whether uni, bi, or, tri or quadri-
        modal.  The default is bimodal (ie, not direction 
        selective).  See note below
    bin:    Boolean.  Whether bin compensation is requested.  This
        only affects the mean vector length(l) and derivative
        statistics.  Where the number of bins > 12, compensation
        is generally not necessary (Batschelet)
    bwdth:    radial length of arc of the bins.  necessary only if
        bin=True
    ci:    percentage for which confidence intervals will be
        calculated

    Returns:    

        angle:    mean angle or angles of mean axis
        d:    confidence interval (ie, angle +/-d)
        l:    length of mean vector / sum of vector lengths (ie, Circ Var / Rayleigh's NUm)
        		(0 - 1) ... 1 --> straight line; 0 --> perfect circle
        L:    absolute length of mean vector (= l*n)

    Notes:

    -- Modes --        
    Modes argument identifies how many modes the dataset is believed
    to possess.  This is important as the majority of circular
    statistics presume a unimodal dataset.  To compensate, the 
    angles of the dataset are multiplied by the amount of modes 
    to generate a unimodal dataset.  This compensation PRESUMES THAT
    THE DIFFERENCE IN ANGLE BETWEEN THE MODES IS EQUAL/SYMMETRICAL.

    The angle calculated is not that of the mean vector, but that of
    the mean axis (or diameter) which passes through both modes of
    data.  Thus angle = (a, a+pi).  For this reason, a dataset with
    more than one mode is said to be "axial".

    -- Conf Int --
    Calculation employs an approximation (see Zar) that is accurate
    only to one degree and that is based on the von Mises.

    Where a NaN is returned, this is most likely due to the q value
    being negative such that there is no square root.  This is
    equates to the confidence interval being greater than 90 degrees
    (ie, arccos of a negative is between 90 & 270 degrees).
    Analytically, it can be shown that this will occur when
    l < (chi / 2*n)**0.5

    References:
        Batschelet (1981)
        Zar, Biostatistical Analysis (2010)

    """

    m = float(modes)
    theta_m = (m*theta) % (2*np.pi)

    # required for Conf Int calculation
    chi = stats.chi2.ppf(ci, 1)
    n = r.sum()


    y = np.sum(r * np.sin(theta_m)) / r.sum()
    x = np.sum(r * np.cos(theta_m)) / r.sum()

    l = (x**2 + y**2)**0.5
    L =  ((x*r.sum())**2 + (y*r.sum())**2)**0.5

    if x > 0 and y > 0:
        a = 1/m * np.arctan(y/x)

    if x < 0 and y > 0:
        a = 1/m * (np.arctan(y/x) + np.pi)

    if x < 0 and y < 0:
        a = 1/m * (np.arctan(y/x) + np.pi)

    if x > 0 and y < 0:
        a = 1/m * (np.arctan(y/x) + 2*np.pi)



    # Conf Interval calculation

    if l < (chi / (2*n))**0.5:
        d = np.pi/2

    elif l <= 0.9:
        q = (2*n * (2*(L**2) - n*chi) / (4*n - chi) )**0.5
        d = np.arccos(q / L)

    elif l > 0.9:
        q = (n**2 - (n**2 - L**2)*np.exp(chi/n))**0.5
        d = np.arccos( q / L)
        

    angle = a + np.arange(m)*np.pi

    if single_angle:
        angle = angle[0]


    # bin compensation & final output
    if bin:
        c = (bwdth/2.) / np.sin(bwdth/2.)
        lc = c * l    
        return angle, d, lc, L
    else:
    
        return angle, d, l, L




def disp(l, modes=2, bin = False, bwdth=0):

    """
    """

    m = float(modes)
    var=0
    varc = 0
    dev=0
    devc = 0
    

    var = 1 - l

    # divide by number modes as per batschelet 1981
    dev = (2*(1-l))**0.5 / m

    if bin:

        c = (bwdth/2.) / np.sin(bwdth/2.)
        lc = c * l
        
        varc = 1-lc
        devc = (2*(1-lc))**0.5 / m

        return var, dev, varc, devc

    else:
        return var, dev


def Rayleigh(r, theta, modes=2, bin = False, bwdth=0, ci=0.95, mod=False, a=0, mu=0):

    """
    n:    is the sum of responses ... ei, the total amount 
        of vectors
    l:    length of the mean vector [0,1]
    L:    length of the sum of vectors [0, R]
    mod:    Boolean.  Whether to employ the modified Rayleigh
        test, also known as a V test, 
        which gains more power by testing against
        a predetermined mean angle, in which case the test
        hypothesis is that the mean angle is mu, and the 
        null, as with the standard test, is circularity.
    a:    the predetermined angle of the data which is used to test
        for circularity with more power
    mu:    test angle for a modified Rayleigh test
    
    
    Reports the p-value of the dataset under a null-hypothesis of 
    circularity and a test hypothesis of non-circularity with a
    predicted mean angle (mu). 

    In the case of a modified Rayleigh test, the test statistic,
    the u value, is related to a p-value by approximating to
    a one-tailed normal deviation, which is appropriate for
    large sample sizes and test probabilities of 0.05.
    Ie, the u value can be taken as the z number (number of std
    deviations), in the positive direction, such that 1 - 
    one-tailed norm.CDF approximates the p-value of obtaining
    said u value under the null hypothesis.  This approximation
    is more accurate for higher n and the closer the p-value
    is to 0.05

    Caculation is an approximation that is good to three decimals,
    for n >=10, and to two decimals for n >=5 (Zar).

    NOTE:
    
    Test presumes sampling from a UNIMODAL von Mises distribution.
    Thus, the input arguments must be derived from a multimodal 
    calculation of the mean vector
    """
    a, ci, l, L = mean(r, theta, modes=modes, bin=bin, bwdth=bwdth, ci=ci)
    n = r.sum()
     
    if not mod:    
        p = np.exp( (( 1 + 4*n + 4*(n**2 - L**2))**0.5) - (1 + 2*n))

        if n > 10:
            return np.around(p, decimals=3)
        elif n > 5:
            return np.around(p, decimals=2)

        else:
            print("sample size is too small")
    elif mod:
        V = L * np.cos(a - mu)
        u = V * np.power( (2/n), 0.5)

        p = 1 - stats.norm.cdf(u)        

        return np.around(p, decimals=3)

    else:
        print("invalid 'mod' kwarg")
    

def omnibus(r, theta):
    """
    less powerful alternative to the Rayleigh that is also suitable
    for samples from populations that may be multimodal.

    for n > 50, an approximation is used that is inaccurate, at the
    very highest, to ~0.003 for a p-value of ~0.5; otherwise, the
    inaccuracy is <~ 0.001

    Where m/n ~ 0.5, the formulae appear to breakdown.  Thus, this ratio
    is returned also as a gauge.
    """

    def f(int):
        int = np.ceil(int).astype('int')
        return misc.factorial(int, exact=True)


    half = np.ceil(r.size / 2.)
    sums = []
    for i in range(r.size):
        s = np.sum(np.roll(r, i)[:half])
        sums.append(s)

    
    m = min(sums)
    n = r.sum()

    if n > 50:

        A = (np.pi * (n**0.5)) / (2. * (n - 2.*m))

        p = (((2*np.pi)**0.5) / A) * np.exp((-np.pi**2) / (8*A**2))
    
        return p, m/float(n)


    elif n < 50:
        p = ( (n - 2*m) * ( f(n) / float((f(m) * f(n-m))))) / float(2**(n-1))

        return p, m/float(n)
        
    
    
def WW():
    """
    Two or multi-sample test of whether the mean direction of the 
    samples are identical or not
    Requires approximation of k from Fisher 1995.

    """



def ang_cor_nonpara(a, b):
    """
    to reject the null hypothesis, v should be bigger than V1(one tailed)
    and/or V2
    """

    rank = stats.mstats.rankdata

    C = 2*np.pi / a.size

    r1 = ( (np.cos( C * (rank(a)-rank(b))).sum())**2 + (np.sin( C * (rank(a)-rank(b))).sum())**2) / a.size**2

    r2 = ( (np.cos( C * (rank(a)+rank(b))).sum())**2 + (np.sin( C * (rank(a)+rank(b))).sum())**2) / a.size**2

    r_aa = r1 - r2

    v = (a.size - 1) * r_aa

    # alpha = 0.05
    V1 = 2.3 + 2.0/a.size
    V2 = 2.99 + 2.16/a.size

    print("\n\nnegative values indicate failed test. DO NOT reject null hypothesis\n")

    print("alpha = 0.05")
    print("1tail:    %f;    2tail:    %f\n" % ( (v-V1), (v-V2)))

    # alpha = 0.1
    
    print("alpha = 0.1")
    V1 = 1.61 + 1.52/a.size
    V2 = 2.3 + 2.0/a.size

    print("1tail:    %f;    2tail:    %f" % ( (v-V1), (v-V2)))

    # alpha = 0.1
    
    print("alpha = 0.1")
    V1 = 1.61 + 1.52/a.size
    V2 = 2.3 + 2.0/a.size

    print("1tail:    %f;    2tail:    %f" % ( (v-V1), (v-V2)))


    

    