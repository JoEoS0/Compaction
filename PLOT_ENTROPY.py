#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:41:10 2020
PLOT_ENTROPY.py
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
@author: ed
"""

print('                      ')

import numpy as np
import matplotlib.pyplot as plt



def  Plotz( vtk, htk, vlabl, hlabl,lab,titl, vo, ho, v, h, name  )  :
    plotx = '/home/ed/Desktop/LatexFiles/'+lab+'.pdf'
    v = np.array(v).tolist()
    h = np.array(h).tolist()

    vo = np.array(vo).tolist()
    ho = np.array(ho).tolist()
    
    #print ' len(ho), len(vo) = ', len(ho), len(vo)            
    vtk = np.array(vtk).tolist()
    htk = np.array(htk).tolist()
    lab = str(lab)+'.pdf'
###############################################################################
    fig = plt.figure(1,figsize=(30.0,15.0 ) )
    plt.ion()
    fig, ax1 = plt.subplots()
    ax1.autoscale()
    ax1.set_ylim( [ htk[0], htk[1] ] )   
    ax1.set_xlim( [ vtk[0], vtk[1] ] )   
    #ax1.set_xticks(htk)
    #ax1.set_yticks(vtk)
    plt.grid()
    ax1.set_xlabel( str(hlabl) )
    ax1.set_ylabel( str(vlabl) ) 
    plt.title( str( titl ) )    
    #
    ax1.plot( ho, vo, 'ko' )  #, label= str(lab)  )
    #ax1.plot.scatter( ho, vo )
    ax1.plot( h, v, 'k-' )  #, label = str(lab) )
    ax1.autoscale()

    plt.savefig(lab)
    plt.savefig(plotx)    
    plt.show
###############################################################################    
   






    
#https://futurestud.io/tutorials/matplotlib-save-plots-as-file
#https://datatofish.com/export-matplotlib-pdf/

def Entropyx( name, xss,xssTP )  :
    """
    Entropy of porosities which go from .00 to .84 in boxes .04 wide
    https://code-examples.net/en/q/ebc050
    https://search.yahoo.com/yhs/search?hspart=ddc&hsimp=yhs-linuxmint&type=__
    alt__ddc_linuxmint_com&p=compute+entropy+python
    """
    entropyx, entropxx    = 0., 0.
    xssTP        = xssTP.tolist()
    slot         = [0]*len(xssTP)
    xss          = xss.tolist()
    box         = [0]*21 
 
    AllContents = -.0000001
    for i in range( len(xssTP) ) :
        slot[i] = slot[i] + int( 100.*xssTP[i] )  #Sets xssTP = <1=0, 1>2 = 1, 2>3 = 2 ..
    for k in range( 21)  :
            n1,n2,n3,n4 =  1+4*k, 2+4*k, 3+4*k, 4+4*k
            a,b,c,d = slot.count(n1), slot.count(n2), slot.count(n3),slot.count(n4)
            box[k]= a+b+c+d
            AllContents += box[k]
    #print' AllContents, type(box), box = ' ,cf.f_lineno, int(AllContents+.002), type(box), box      
    for j in range( 21 ) :
        if box[j] > 0 and AllContents > 0 :
            entropyx = entropyx - (box[j]/AllContents)*np.log(box[j]/AllContents)    
    print('                                                                     ')
    print('                                                                     ')
            
    AllContens = -.0000001
    for i in range( len(xss) ) :
        slot[i] = slot[i] + int( 100.*xss[i] )  #Sets xss = <1=0, 1>2 = 1, 2>3 = 2 ..
    for k in range( 21)  :
            n1,n2,n3,n4 =  1+4*k, 2+4*k, 3+4*k, 4+4*k
            a,b,c,d = slot.count(n1), slot.count(n2), slot.count(n3),slot.count(n4)
            box[k]= a+b+c+d
            AllContens += box[k]            
    #print' AllContens, type(box), box = ' ,cf.f_lineno, int(AllContens+.002), type(box), box      
    for j in range( 21 ) :
        if box[j] > 0 and AllContens > 0 :
            entropxx = entropxx - (box[j]/AllContens)*np.log(box[j]/AllContens)    

    #print 'name,entropyx, entropxx = ' ,cf.f_lineno, name,  '%6.3f'  %  entropyx, entropxx
    eta_ratio = entropyx/entropxx
    return name, entropyx, entropxx, eta_ratio
    


















