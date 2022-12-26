###!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:01:07 2020
Plotz2
https://matplotlib.org/gallery/api/two_scales.html
https://stackoverflow.com/questions/53531429/valueerror-invalid-rgba-argument-what-is-causing-this-error
https://futurestud.io/tutorials/matplotlib-save-plots-as-file
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
@author: ed
"""

import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
#import copy


def Plotz2( vtk, htk, vlabl, hlabl,lab,titl, vo, ho, v, h,\
           name, temperature,ans )   :

    #print 'len(popt), len(vtk), len(htk), len(vo), len(ho), len(v), len(h), len(temperature) = ',\
    #    len(popt), len(vtk), len(htk), len(vo), len(ho), len(v), len(h), len(temperature)
    #print
    fig, ax1 = plt.subplots() 
    ax1.autoscale()

    v = np.array(v)  # porosity fit
    h = np.array(h)  # sigma or lambda

    vo = np.array(vo)  # porosity data or trend picks 
    ho = np.array(ho)  # sigma or lambda
    
    #print ' len(ho), len(vo) = ', len(ho), len(vo)            
    vtk = np.array(vtk)
    htk = np.array(htk)
    v = np.array(v)  # porosity fit
    h = np.array(h)  # sigma or lambda

    vo = np.array(vo)  # porosity data or trend picks 
    ho = np.array(ho)  # sigma or lambda
    
    #print ' len(ho), len(vo) = ', len(ho), len(vo)            
    vtk = np.array(vtk)
    htk = np.array(htk)
    lab = str(lab)+'.pdf'

    #fig, ax1 = plt.subplots()
       
    ax1.set_xlabel(hlabl)
    ax1.set_ylabel(vlabl )
    plt.title( str( titl ) ) 
    ax1.plot( ho, vo, 'ko' )
    ax1.plot( h, v,  )  
    ax1.tick_params(axis='y')
    ax1.set_ylim( [0.,.85] )
    plt.grid()
    
    ax2 = ax1.twinx()   # instantiate a second axes that shares the same x-axis
    if ans == 'yes' :   # Add the temperature on the second axis
        ax2.autoscale()
        ax2.set_ylabel('T deg C')  # we already handled the x-label with ax1
        ax2.plot(h, temperature)  
        ax2.set_ylim( [ temperature[0],temperature[-1] ] )
        ax2.tick_params(axis='y') 
    if ans == 'age'  :  #
        ax2.autoscale() 
        ax2.set_ylabel('Time since deposition, My')
        ax2.plot(h, temperature)   # actually 'age' = time since deposition
        ax2.set_ylim( [ temperature[0],temperature[-1] ] )
        ax2.tick_params(axis='y')         
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    #https://stackoverflow.com/questions/43508142/matplotlib-plots-turn-out-blank 
    ax2.autoscale()
    #https://stackoverflow.com/questions/42875357/deformed-rectangulars-with-decreasing-trend
    #plt.show()
    plt.savefig(lab)
    plt.show()
    
    pass







#https://matplotlib.org/gallery/pyplots/pyplot_text.html#sphx-glr-gallery-pyplots-pyplot-text-py
def phi_histigram( porosity, name )  :
    
    #print 'len(porosity), len(phi), max(porosity), max(phi) = ',len(porosity), len(phi), max(porosity), max(phi)
    # the histogram of the data
    n, bins, patches = plt.hist(porosity, 50, density=True, facecolor='k', alpha=0.75)
    #print ' n, bins, patches = ',   n, bins, patches
    plt.xlabel('Transformed porosity, '+name+' ' )
    plt.ylabel('Number of porosity-fit points per bin')
    plt.title('Histogram of transformed porosity')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    plt.xlim(0., .82 )
    plt.ylim(0, len(porosity) )
    plt.grid(True)
    plt.show()
 
    
    
 


