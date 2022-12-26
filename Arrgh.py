#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 05:12:51 2020
    Arrgh
@author: ed
"""


import numpy as np
import copy
from scipy.optimize import curve_fit
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
Low_Temp = 30.  # Cut-off deg C. Methane clathrates shallow?


def EAandPlotCoords(temp, Low_Temp, RTinv, xs, m, n, phi0, Dtdeposit):
    #### Estimate E & A from the SLOPE of actual data points alone: ###########
    # Chop off irrelevent porosities == xs, for temperatures LT "Low_Temp", maybe ~ 40 C?, and\
    #    porosities > phi0:

    I = Chop_Low_T(temp, Low_Temp, phi0, xs)

    RTinvx = copy.deepcopy(RTinv[I:])
    Dtdepo = copy.deepcopy(Dtdeposit[I:])
    xsx = copy.deepcopy(xs[I:])
    lenx = len(RTinvx)

    # Get a straight line fit to these hotter data points in (lnLS11,RTinv)     LNS11xxGen
    #
    lnLS11x = np.zeros(lenx, dtype=float)
    for i in range(lenx):
        lnLS11x[i] = pow(xsx[i], -4.*m/3.)*pow((1.-xsx[i]), -4.*n/3.)*\
            (np.log((1.-xsx[i])/(1.-phi0))/Dtdepo[i])
        if lnLS11x[i] <= 0.0000001  :
            print('  Arrgh! ZERO lnLS11x[i]  ', i, lnLS11x[i] ) 
        lnLS11x[i] = max(0.0000001, lnLS11x[i])                    
        lnLS11x[i] = np.log( lnLS11x[i] )                    
    
    def func(RTinvx, u, E) :
        return u+E*RTinvx

    sig = np.ones(2)*.5  # Error estimates for popt variables

    
    popt, pcov = curve_fit(func, RTinvx, lnLS11x, sig)

    E = popt[1]

    # Also return axis length-vectors needed by the plotting program:
    axisRT = np.array([RTinvx[0], RTinvx[-1], .0001])
    axislnLS = np.array([lnLS11x[0], lnLS11x[-1], .001])
    
    # Return the shortened RTinvx & generated lnLS11xGen to main program for plotting:
    lnLS11xGen = np.ones(lenx, dtype=float)
    lnLS11xGen = popt[0] + popt[1]*RTinvx

    sec_per_my = 3.1557*pow(10., 13)
    
    Apop0 = np.exp(popt[0])/sec_per_my

    return E, Apop0, axisRT, axislnLS, lnLS11xGen, RTinvx

    pass


def Chop_Low_T(t, Low_Temp, phi0, xs):
    l, I, visits = 0, 0, 0
    for l in range(len(t)):
        tt = t[l]
        xst = xs[l]
        if tt < Low_Temp or xst > phi0:
            visits = visits+1
            I = l
    I = I + (visits > 0)
    return I
    pass
