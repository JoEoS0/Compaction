#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 06:52:56 2020 
porosity0.py
@author: ed
"""
#Test Case:
#sigma, tau, phi = 1379., 1000., .45
def porosity0(sigma0, tau0, q=1., r=1.,u=1.) :
    """
    This computess the empirical porosity,ln(phi/.8) = -.1(tau+1.5u)**{.23q}sigma**{.25r}
    where [u,q,r] = [1,1,1] for i nitial guesses.   
    Smith, J. E., Dysinger, G. C., and Borst, R. L., Shales and Abnormal Pressures.
    Geotechnical and Environmental Aspects of Geopressure Energy, ed. S. K. Saxena,
    1980, pp.69-91.        
    """        
    import math
    a = (tau0+1.5*u)**(.23*q)
    b = (sigma0)**(.25*r)
    c = -0.1*a*b
    phiest = 0.8*math.exp( c )
    #print "a,b,c,u,q,r =", a,  b,  c, u, q, r
    #print "bar, tau0,phiest =", sigma0,  tau0,   phiest
    return phiest

# Test Case    
#porosity0(sigma, tau, 1., 1., 1. )
def porosity0Plot( q=1.,r=1., u= 1. ) :
    """
    This plots the empirical porosity,ln(phi/.8) = -.1(tau+1.5u)**{.23q}sigma**{.25r}
    where [u,q,r] = [1,1,1] for i nitial guesses.
    (x, y; z) = (sigma , phi ; tau )
    https://www.tutorialspoint.com/matplotlib/
    matplotlib_contour_plot.htm.
    https://www.youtube.com/watch?v=XLJHkCn48lM
    """  
    #import math
    import copy
    import matplotlib.pyplot as pl
    #import numpy as np
    fig = pl.figure() 
    pl.title( 'Porosity vs Geologic Age and Effective Pressure with Geologic Age' )
    pl.xlabel( 'Pressure difference sigma (bar)' )
    pl.ylabel( 'Fractional Porosity' )   
    pl.legend(loc = 'upper right')         
    pl.axes = pl.gca()      #GCA = get current axes
    pl.axes.set_ylim( [ .0, .8 ] )         
    pl.axes.set_xlim( [ -10, 1510 ] )
    #pl.grid(True)
    label0 = ['1my','20my','100my','300my']
    clr = ['b','g','k','r']       #colors
    #mkr = ['^','-','v','o']       #data point markers
    #ltp = ['--','.','-',':']      #line type
    #graph, = pl.plot( [], [], 'o' )
    
    tau =[1.,20.,100.,300. ]
    lt  = len(tau)
    sig =  range( 10,1510,10 )    #sigma  
    ll = len(sig)
    s = [ [0.]*ll, [0.]*ll, [0.]*ll, [0.]*ll ]
    phix = [ 0. ]*ll

    for i in range( lt )        :       #tau
        for j in range( ll )    :       #sigma
            ta = float(tau[i])
            si = float(sig[j])
            sig[j] = float(sig[j])

            phix[j] = porosity0(ta,si, q=1., r=1.,u=1.)
        s[i] = phix[:]
        #
        print 'max(phix), min(phix), ll = ', '%10.3f' % max(phix),'%10.3f' % min(phix), ll
        print 'max(sig) ,  min(sig), ll = ', '%10.1f' % max(sig) ,'%10.1f' % min(sig), ll

        #if i >= 0  :
        pl.plot( sig, phix, color = clr[i],label= label0[i] )
        if i == lt-1  :                    
            pl.legend()
            #fig = pl.plot( sig, phix, color = clr[i],label= label0[i] )
            fig.savefig("Figure1.pdf", bbox_inches='tight')    
    pl.show()
    #f.savefig("foo.pdf", bbox_inches='tight')    
              
              
porosity0Plot( q=1.,r=1., u= 1. )     
  


def porosity1Plot( q=1.,r=1., u= 1., n=1.,m=1. ) :
    """(x, y, z,R) = (sigma , phi , tau, Eq7rhs )
    https://www.tutorialspoint.com/matplotlib/
    matplotlib_contour_plot.htm.
    https://www.youtube.com/watch?v=XLJHkCn48lM
    """  
    import math
    import copy
    import matplotlib.pyplot as pl
    import numpy as np
    fig = pl.figure() #+str( ) 
    params = 'RHS of Equation 7 with m,n,q,r,u= ' +str(m)+'  '+str(n)+'  '+str(q)+'  '+str(r)+'  '+str(u) 
    
    pl.title( params )
    pl.xlabel( 'Pressure difference sigma (bar)' )
    pl.ylabel( 'RHS of Equation 7' )   
    pl.legend(loc = 'upper right')         
    pl.axes = pl.gca()      #GCA = get current axes
    
    pl.axes.set_ylim( [ 0., 1. ] )   
          
    pl.axes.set_xlim( [ -10, 1510 ] )
    #pl.yscale( 'log' )
    #pl.grid(True)
    label0 = ['1my','20my','100my','300my']
    clr = ['b','g','k','r']       #colors
    #mkr = ['^','-','v','o']       #data point markers
    #ltp = ['--','.','-',':']      #line type
    #graph, = pl.plot( [], [], 'o' )
    
    tau =[ 1. ,20. ,100. ,300. ]
    lt = len(tau)
    
    sig =  range( 10,1510,10 )    #sigma, bar =    
    ll = len(sig)
    sigs = copy.deepcopy(sig)
    # bar*1.0197 = kgf/m**2
    sigs = map( lambda x:x*1.0197, sigs )  #bar = kgf/m**2 = .1*gf/cm^2  
    #
    print 'max(sigs), min(sigs) = ','%10.1f' % max(sigs), '%10.1f' % min(sigs)
    print '                                                                      '
    #
    # Assign a Kelvin temperature to every point on each tau(sig,phi) curve.
    # T = f(dTpalioSurface,sigs,depth) :   
    #dTpaleoSurface  is deg C above 14 deg C for the Holocene.
    #https://en.wikipedia.org/wiki/Geologic_temperature_record#Overall_view
    # https://www.britannica.com/science/seawater/Density-of-seawater-and-pressure
    dTpaleoSurface = [ 0. ,4., 13., 13., 25. ]    # deg C
    #  quartz:  2.32 g/cm^2 =  23.2 kg/m^2
    #grav  = 5.574*10**(-11)          # m^3/kg/sec^2 
    drhof  = (2.75 - 1.02)*10         # gf/cm^3 = 1000*kgf/m^3
    R     = 8.314                     # kj/degK/mole
    #
    gradT = .04                       # deg C / m
    Tav = 350                         # deg K; makes Quartzexp close to 1.
    #
    Equartz = 89                      # quartz activation energy, kj/mole 
    Quartzexp = math.exp(-Equartz/R/ Tav)  # =.9698 for T = 350.
    print 'exp**(-E/RT) (=~1 so lhs =~ A !)  =', Quartzexp
    print '                                                                    '
    #
    depthz = [ 0. ]*ll
    Tz     = [ 0. ]*ll       
    rhs    = [ 0. ]*ll
    #
    depths = [ [0.]*ll, [0.]*ll, [0.]*ll, [0.]*ll, [0.]*ll ]
    Tzs    = [ [0.]*ll, [0.]*ll, [0.]*ll, [0.]*ll, [0.]*ll ]    
    rhss   = [ [0.]*ll, [0.]*ll, [0.]*ll, [0.]*ll, [0.]*ll ]
    Lambda = [ [0.]*ll, [0.]*ll, [0.]*ll, [0.]*ll, [0.]*ll ]
    Lambdax= [ 0. ]*ll    
    phiy   = [ 0. ]*ll

    for i in range( lt )        :       #tau
        ta = float(tau[i])
        mudlineT = 273.2 + 14. +  dTpaleoSurface[i]  # deg K
        #
        for j in range( ll )    :       #sigma
            # The same series of sig is used for every tau[i] :
            sig[j] = float( sig[j] )
            si     = float( sig[j] )
            
            phiy[j] = porosity0( si, ta, q=1., r=1. ,u=1.)
            Lambdax[j] = si/( 1. -phiy[j] )
            phi     = phiy[j]
            
            # Compute the approximate depth/m, assuming normal pressures.
            # Used to compute T kelvin for average paleo-surface.
            # depth=(kg/m**2)/(kg/m**3)
            if j == 0 :
                depthz[j] = sigs[0]/( drhof*(1.-phi) )
                zdT = mudlineT + depthz[j]*gradT
            else  :
                depthz[j] = depthz[j-1] + (sigs[j] - sigs[j-1 ] ) /(drhof*(1.-phi) ) 
            zdT   = zdT + gradT*depthz[j]  # temperature increase from surface to depthz:   
            Tz[j] = zdT            
           
            # Compute the estimated expression for the Arrhenius exponent rhs:
            rhs[j] = .023*(ta+1.5*u)**(.23*q-1.)*si**(.25*r)*phi**(-m/3.+1.)* \
                    (1.-phi)**(-float(n)/3.-1.)*np.exp(Tz[j]/Tav) 

        depths[i] = depthz[:]               
        Tzs[i]     = Tz[:]         
        rhss[i]   = rhs[:]
        
        print 'max(Lambda), min(Lambda), ll = ', '%10.1f' % max(Lambdax),'%10.1f' % min(Lambdax), ll        
        print 'max(sigs), min(sigs), ll     = ', '%10.1f' % max(sigs), '%10.1f' % min(sigs), ll                
        print 'max(depthz), min(depthz), ll = ', '%10.1f' % max(depthz),'%10.1f' % min(depthz), ll        
        print 'max(rhs), min(rhs), ll       = ', '%10.4f' % max(rhs), '%10.4f' % min(rhs), ll        
        print 'max(phiy), min(phiy), ll     = ', '%10.3f' % max(phiy),'%10.3f' % min(phiy), ll
        print 'max(sig), min(sig), ll       = ', '%10.1f' % max(sig), '%10.1f' % min(sig), ll
        print '                                                                                '

        #if i >= 0  :
        pl.plot( sig, rhs, color = clr[i],label= label0[i] )
        if i == len(tau)-1  :                    
            pl.legend()
            #fig = pl.plot( sig, rhs, color = clr[i],label= label0[i] )
            fig.savefig("Figure2.pdf", bbox_inches='tight')    
    pl.show()
       
#           
porosity1Plot( q=1.,r=1., u= 1., n=3., m=3. )
    
    
    
    