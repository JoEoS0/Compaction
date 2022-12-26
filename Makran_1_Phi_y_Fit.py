#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 19:07:07 2020
Makran_1_Phi_y_Fit.py
@author: ed
"""

from Arrgh import EAandPlotCoords
#from PLOTZ2 import phi_histigram
from PLOTZ2 import Plotz2
#from PLOT_ENTROPY import Entropyx
from PLOT_ENTROPY import Plotz 
from propertys import shale
import numpy as np
from scipy.optimize import curve_fit
from inspect import currentframe, getframeinfo
#import math
from pandas import DataFrame
import pandas as pd




cf = currentframe()
filename = getframeinfo(cf).filename
print("This is ", filename, ", code line = ", cf.f_lineno) 

ShaleNames       = [ 'Akita',  'Makran1','Makran2', 'SuluSea', 'Oklahoma', 'Maracaibo', 'NETailand' ]
#def fit( 'name' )  :

#vvvvvvvvvvvvvvvvvv_Makran_Input_Data_from_'def shale'_vvvvvvvvvvvvvvvvvvvvvvvv
#Create some lists to import data into:
AgeRange1            = [ 0., 0. ]
DepthRange1          = [ 0., 0. ]
PorosityDepthParams1 = [ 0., 0., 0. ]
ShaleName1, ShaleNumber1, AgeRange1, DepthRange1, PorosityDepthParams1, \
PaleoMudlineT1, GeothermalGradientMax1, TKavg1, TKavgavg, dy_steps,AuthorDate,\
   Low_Temp,Deposition_Rate,AgeName = shale( 'Makran1' )

#print 'ShaleName1, ShaleNumber1, AgeRange1, DepthRange1, PorosityDepthParams1, PaleoMudlineT1, \
# GeothermalGradientMax1, TKavg1, TKavgavg,dy_steps, AuthorDate,\
#   Low_Temp,Deposition_Rate  = ', \
#       ShaleName1, ShaleNumber1, AgeRange1, DepthRange1, PorosityDepthParams1, PaleoMudlineT1, \
#       GeothermalGradientMax1, TKavg1, TKavgavg,dy_steps,AuthorDate,\
#       Low_Temp,Deposition_Rate

#^^^^^^^^^^^^^^^^^^_Makran_Input_Data_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
# x1 is abyssal plain porosity:
x1 = np.array( [ .46,.31,.11,.46,.31,.29,.21,.15,.12,.35,.20,.14,.07,\
  .38,.16,.53,.36,.22,.11,.46,.24,.08,.07,.12,.07,.69,.41,.23,.13 ] )
# y1 is abyssal plain depth in km:
y1 = np.array( [ 0.30,0.97,1.86,0.30,0.97,1.53,1.71,2.37,3.23,0.92,1.84,\
  2.52,4.47,0.50,3.08,0.28,1.66,2.03,3.61,0.44,1.45,3.97,4.21,2.55,3.78,\
  0.05,0.31,1.72,3.04 ] )

#Variables for tinkering. These over-ride those imported from 'shale': 
GeothermalGradientMax1  =  .03            #.03  #.023 # .019
PaleoMudlineT1          = PaleoMudlineT1  # PaleoMudlineT1
m,n                     = .85 , 1.        #.85, 1.   
KmSpMy                  = .2
Tau0_1                  = 1. # Add more geologic time, following section deposition.
#------------------------------------------------------------------------------ 

# Here ys increases monotonically. The matching xs aren't monotonic:
ys,xs   = list(map( np.array, list(zip(*sorted( zip(y1 ,x1 ) ) )) ))
ys = [w*1000. for w in ys]   #km to m depth
#print( 'Macran1_67', '  Depth:',len(ys), ys, '  phi:', xs ) 
#vvvvvvvvv_Abyssal_Plain_Porosity_vs_Depth_Fit_vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv 

#Compute the  log of experimental porosities, 'map' = Fehily p-242:
phi0  = .55
xp = [np.log( w/phi0 ) for w in xs]#titl =  'T deg C = a + b*depth;  a,b ='+str('%4.0f' % PaleoMudlineT1)+ ',  ' +str('%4.3f' % GeothermalGradientMax1) 
  
# Curve fit abyssal plain porosity data :
a = -.796
b = .065    
#______________________________________________________________________________

# Scipy uses y = func(x); I use x = func(y) :
a,b,bx = -7.98*np.exp(-4), 6.53*np.exp(-8), 2.

#ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
def func(ys, a, b )  :
    bx = 2.
    return a*ys +b*pow(ys, bx)  #   #< gives ln(phi/phi0)
#ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
#Fit for the parameters a, b, .. of the function func: Marsland p376
sig = np.ones( 2 )*.5
popt, pcov = curve_fit(func, ys, xp, sig  )
print(' phi0, popt = ' , phi0, popt)
#______________________________________________________________________________


#Plot the fitted curve with dy_steps points:
yss = np.linspace(0., max(ys), dy_steps )
xss = np.linspace(0.,0., dy_steps )

for q in range( len(xss) ) : 
    xss[q] = phi0*np.exp(( func(yss, *popt)[q] ) )
#print ' xss = ', cf.f_lineno , xss    
#
# Get std_phi_error = sqrt( sum( ( (phi - phi_fit)**2/( samples-1 ) )  )
s = 0.
estphi = np.zeros( len(xs) )
for q in range( len(xs) ) : 
    estphi[q] = phi0*np.exp( func(ys[q], *popt) )
    s = s + pow( ( xs[q]- estphi[q] ), 2 )    
std_phi = np.sqrt( s/( len(xs)-1.) ) 
fit_std = '%4.3f' % std_phi                 
print(' fit_std  ', cf.f_lineno,  fit_std)  #= .056
# Fehily p-114-5. Google python scientific formating
titl1  =  'ln(porosity/'+str(phi0)+') = a + b*depth^'+repr(bx)+' \n a=%4.2e, b=%4.2e' % tuple(popt)+ ' ,porosity std='+str(fit_std)+')'
print(' titl1 = ',    titl1)
titl1  =  'ln(porosity/0.55) = -5.70*10^-4 + 2.27*10^-8*depth^2 \n porosity standard deviation = .049'
#_____________________________________________________________________________
###############################################################################

#vvv_Get_ Sigma,_MatrixMass,_Temperature,Lambda & Arrhenius Profiles_ln(LHS) Eq11 vvv
################### before slicing off the top 363 m ##########################
#______________________________________________________________________________

#vvvvvvvvvvvvvvvvvvvvvvvv Create all arrays of maximum length VVVVVVVVVVVVVVVVV New

#For every phi and Z compute sigma = sum of of (rhog-rhow)*(1.-phi)*dZ= 1.67*(1.-phi)*dZ.
#https://www.convertunits.com/from/kgf/sq.cm/to/bar, (10 Metres of water = 0.9806 Bars)
#Convert gf/cm^3*km to bars. 1 kilograms-force/square centimeter = 0.980665 bars.
#(1gf/cm^3)*km*(100000cm/km)*(.oo1 kgf/gf)*( .980665 bar/( kgf/cm^2 )) = 98.0665 bar

#Data Fits:
sigma = np.array([0.]*len(yss))  # Fitted porosities vs depth
temperature = np.array([0.]*len(yss))  # Temperature deg C vs depth
temperature[0] = PaleoMudlineT1
Lamda = np.array([0.]*len(yss))   # grain_to_grain force per unit contact area
lnLHS11 = np.array([0.]*len(yss))  # Ln(LHS) Eq 11
xssP = np.array([0.]*len(yss))  # Porosity with 'porosity correction'
RTinverse = np.array([0.]*len(yss))      # - 1/RT  1/kJ mol
Matrix = np.array([0.]*len(yss))      # sum (1-phi)*dz
# millions of years after section deposition began
Dtdeposit = np.array([0.]*len(yss))


#Data:
sig = np.array([0.]*len(ys))    # Fitted porosities vs depth
temp = np.array([0.]*len(ys))  # Temperature deg C vs depth
temp[0] = PaleoMudlineT1
Lam = np.array([0.]*len(ys))    # grain_to_grain force per unit contact area
lnLS11 = np.array([0.]*len(ys))  # Ln(LHS) Eq 11
xsP = np.array([0.]*len(ys))
RTinv = np.array([0.]*len(ys))
Matrx = np.array([0.]*len(ys))      # sum (1-phi)*dz
# millions of years after section deposition began Mya M
Dtdepo = np.array([0.]*len(ys))


for i in range(1, len(xss), 1):
    #https://www.convertunits.com/from/bar/to/grams+per+(square+meter)
    #conversion = [gf/cm^2*(10^4cm^2/m^2)]*[9.80665bar/(gf/m^2)] \
    # multiplier = 9.80665*10^4
    #Note: 'yss' = DEPTH & 'xss' = POROSITY:
    sigma[i] = sigma[i-1] + 1.67*(1. - xss[i])*(yss[i] - yss[i-1])\
        * .00980665  # MPa
    temperature[i] = temperature[i-1] + GeothermalGradientMax1 *\
        (yss[i] - yss[i-1])  # deg C
    Lamda[i] = Lamda[i-1] + sigma[i] / \
        (1. - xss[i])                              #
    Matrix[i] = Matrix[i-1] + (1. - (xss[i-1]+xss[i])/2.) *\
        (yss[i] - yss[i-1])          # total meters of solids
Lamda[0], Matrix[0] = Lamda[1]/2., Matrix[1]/2.     # Must not divide by zeros

for i in range(len(xss)):
    Dtdeposit[i] = (Matrix[i]*.001)/KmSpMy    \
        # million years to accumulate matrix to this km depth
    RTinverse[i] = -1./(273.2+temperature[i])/(8.314*.001)  # -1/kJ/mo

GeoT_O_DepT = (AgeRange1[1]-AgeRange1[0])/Dtdeposit[-1]      # Has to be > 1.

if GeoT_O_DepT < 1.:  # Mya
    print('THE DEPOSITION TIME IS TOO LONG. MUST KEEP "GeoT_o_DepT" > 1.\
          AND HENCE "KmSpMy" BIGGER.')

for i in range(len(xss)):
    # Center the deposition time in the geologic time interval:
    Dtdeposit[i] = Dtdeposit[i] + Dtdeposit[-1]/2. +\
        Tau0_1*(AgeRange1[1] + AgeRange1[0] - Dtdeposit[-1])/2.

    for k in range(len(ys)):  # Interpolate for plotting
        if (yss[i-1] <= ys[k]) and (yss[i] >= ys[k]):
            sig[k] = sigma[i]
            temp[k], Lam[k] = temperature[i], Lamda[i]
            Lam[k] = Lamda[i]  # Ln(LHS) Eq 11
            Matrx[k] = Matrix[i]      # meters of solids
            # millions of years after section deposition began
            Dtdepo[k] = Dtdeposit[i]
            RTinv[k] = RTinverse[i]  # -1/kJ/mo


##Quantities for getting meters of matrix per million years == Rate for this well:
TD = ys[-1]
Matrix_to_TD = Matrix[-1] / TD
# km/My from start of deposition until now
Rate = Matrix[-1]*.001 / Dtdeposit[-1]


############# Firstly__Correct_Porosity_for_Surface_Areas_#####################

# Normalization only  for plotting 'corrected' porosity function
PhiAv = np.mean(xss)
PhiNorm = pow(PhiAv, -4./3.*m) * pow((1.-PhiAv), -
                                     4./3.*n)  # Normalization constant

for i in range(len(xss)):
    xssP[i] = xss[i] * pow(xss[i], -4./3.*m) * \
        pow((1.-xss[i]), -4./3.*n)/PhiNorm


for i in range(len(xs)):
    xsP[i] = xs[i] * pow(xs[i], -4./3.*m) * pow((1.-xs[i]), -4./3.*n)/PhiNorm


########## Porosity and Temperature C, vs Depth, m ####################
 ############## PLOT - 1 : porosity vs depth #################################
fig = 'Figure'+'  '+repr(ShaleNumber1+1)

ans = 'yes'  # Do you want t deg C  plotted?
lab = 'Fig'+repr(ShaleNumber1+1)
titl = titl1
ylabl = str(ShaleName1)+' porosity'
xlabl = fig+str(ShaleName1)+' well porosity data trend (Velde 1996)'
xDim = ' Depth, m '

Plotz2(np.array([0., .9, .1]), np.array([0., 400., 50.]), ylabl,
       xlabl, lab, titl, xs, ys, xss, yss, ShaleName1, temperature, ans)

################ PLOT 2:  Porosity and temperature C  vs Lambda MPa ###########
fig = 'Figure'+' '+repr(ShaleNumber1+7)
anss = 'yes'  # Do you want t deg C  plotted?
labl, lab = '', 'Fig'+str(ShaleNumber1+7),
titl = ' Age interval is'+str('%4.0f' % AgeRange1[0]) + ' -' + str('%4.0f'
                                                                   % AgeRange1[1])+' Ma.'
xlabl = fig+' '+str(ShaleName1)+''
ans = 'yes'
xDim = ' Grain-to-grain stress Lambda, MPa '

Plotz2(np.array([0., .9, .1]), np.array([0., 400., 50.]), ylabl,
       xDim, lab, titl, xs, Lam, xss, Lamda, ShaleName1, temperature, ans)

##### PLOT 3: 'surface-corrected'-Porosity and temperature C  vs_Lambda MPa ###
fig = 'Figure'+' '+repr(ShaleNumber1+13)

labl, lab = ' ', 'Fig'+' '+repr(ShaleNumber1+13)

titl = 'T = a + b*depth;  a,b ='+str('%3.1f' % PaleoMudlineT1) + '\
    ' + str('%4.3f' % GeothermalGradientMax1) +\
    '.  m,n = '+str('%3.2f' % m) + ',  ' + str('%3.2f' % n)

ans = 'yes'  # Do you want t deg C  plotted?
ylabl = 'Surface-corrected'+'  porosity'
xlabl = fig+ShaleName1+'.'
Plotz2(np.array([0., .9, .1]), np.array([0., .400, .05]), ylabl,
       xDim, lab, titl, xsP, Lam, xssP, Lamda, ShaleName1, temperature, ans)


########## PLOT 4:  Age of sediment My since start of deposition vs Depth m ###
fig = 'Figure' + repr(ShaleNumber1+25)
labl, lab = '', 'Fig'+repr(ShaleNumber1+25)
titl = 'Shale age with matrix deposition rate'+str('%4.1f' % KmSpMy)+' km/My'
ans = 'age'
xlabel = 'Depth, km, ' + fig+ShaleName1+'.'
ylabl = str(ShaleName1)+' porosity'
xDim = ' Depth, m '
Plotz2(np.array([0., .9, .1]), np.array([0., 400., 50.]), ylabl,
       xDim, lab, titl, xs, ys, xss, yss, ShaleName1, Dtdeposit, ans)


############### Plot 5: Transformed  Porosity Histogram #######################
#phi_histigram( xssP, ShaleName1 )
#______________________________________________________________________________
phi00 = 0.5   # probably chaotic if greater
#name, entropyx, entropxx, eta_ratio = Entropyx( ShaleName1,xss, xssP  )
#______________________________________________________________________________
#______________________________________________________________________________


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# output = F( input).    SIX  'REAL' LDATA POINTS for SuluSea:

Eslopx, Apop0, axisRT, axislnLS, lnLS11xGen, RTinvx = \
    EAandPlotCoords(temp, Low_Temp, RTinv, xs, m, n, phi00, Dtdepo)

Eslopexx, Apop00, axisRTx, axislnLHSx, lnLS11xxGen, RTinvxx = \
    EAandPlotCoords(temperature, Low_Temp, RTinverse,
                    xss, m, n, phi00, Dtdeposit)


avE, avA = (Eslopx + Eslopexx)/2., (Apop0 + Apop00)/2.

print("( See code lines ~ 49-53 to change GeothermalGradientMax1,PaleoMudlineT1, m, n, Method,\
and KmSpMy My_per_Km_Sediment). )")
print(' E kJ/mol from '"data"' and from curve fit:, ',
      repr(round(Eslopexx, 1)), repr(round(Eslopx, 1)))
print(' A/sec from '"data"' and from curve fit:, ',  (Apop0, Apop00))
############## PLOT - 0 : Arrhenius plot ######################################
fig = 'Figure '+' '+str(ShaleNumber1+19)
ans = 'no'  # Do you want t deg C  plotted?
lab = 'Fig'+str(ShaleNumber1+19)
titl = ' Arrhenius plot with slope E ='+repr(round(avE, 1)) + ' kJ/mol '
ylabl = 'ln( LHS of Equation 11 )  /Myr'
xlabl = fig+'-1/RT.  '+str(ShaleName1)+', ' + \
    str(AuthorDate[ShaleNumber1])+'    /kJ/mol'
Plotz(axisRT, axislnLS, ylabl,
      xlabl, lab, titl, lnLS11xGen, RTinvx, lnLS11xxGen, RTinvxx, ShaleName1)
##############################################################################

# p-116, W. McKinney "Python for data analysis" Also p-112-133
#https://stackoverflow.com/questions/31983341/using-scientific-notation-in-pandas
pd.set_option('display.float_format', '{:.1E}'.format)
pd.Series(data=[10.0])
print()
print(' ',    str(ShaleName1) + ' Summary :')
print('Known shale age limits, My = ', AgeRange1, ', ', AgeName)
print()
data = {'Parameters': ['Total Depth', 'Solids Thickness',
                       'Geothermal Gradient', 'Surface temperature', 'Surface porosity', 'Minimum porosity',
                       'Minimum Temperature', 'Solids deposition rate', 'm', 'n', 'E', 'A'],

        'Value': [str(round(yss[-1])), str(round(Matrix[-1])),
                  str(GeothermalGradientMax1), repr(
                      PaleoMudlineT1), repr(phi0), repr(phi00),
                  repr(Low_Temp), repr(KmSpMy), repr(round(m, 2)), repr(round(n, 2)), repr(round(avE, 1)), avA],

        'Units': ['m', 'm',
                  'deg C/m', 'deg C', 'fraction', 'fraction', 'deg C', 'km/My',
                  '', '', 'kJ/mole', '/s'],

        'Comments': ['', '', '', '', '', 'Arbitrary limit', 'Arbitrary limit',
                     'Keep > ' + repr(round(Rate, 2))+'km/My', 'Derived', 'Derived', 'Derived', 'Derived']}
frame = DataFrame(data)

#https://www.google.com/search?client=firefox-b-1-lm&q=print+DataFrame+in+Ipython
#export DataFrame to CSV file
nam = str(ShaleName1)
frame.to_csv(r'/home/ed/Desktop/Z_Code0/'+ nam +'Data.csv', index=False)
print(frame)
pass
print(' THE END')

















































#Data Fiits:
#sigma       =  np.array( [ 0.]*len(yss) )  # Fitted porosities vs depth
#temperature =  np.array( [ 0.]*len(yss) )  # Temperature deg C vs depth
#temperature[0] = PaleoMudlineT1
#Lamda       =  np.array( [ 0.]*len(yss) )   # grain_to_grain force per unit contact area
#lnLHS11     = np.array( [ 0.]*len(yss) )    #Ln(LHS) Eq 11
#RTinverse = np.array( [ 0.]*len(yss) )      # - 1/RT  1/kJ mol
#xssP      = np.array( [ 0.]*len(yss) )      #Porosity with 'porosity correction'
#Matrix    = np.array( [ 0.]*len(yss) )      # sum (1-phi)*dz
#Dtdeposit = np.array( [ 0.]*len(yss) )      # For estimates of activation E 

#Data:
#sig         =  np.array( [ 0.]*len(ys) )    # Fitted porosities vs depth
#temp        =  np.array( [ 0.]*len(ys)  )   #  Temperature deg C vs depth
#temp[0]     =  PaleoMudlineT1
#Lam         =  np.array( [ 0.]*len(ys) )    # grain_to_grain force per unit contact area
#lnLS11      =  np.array( [ 0.]*len(ys) )    #Ln(LHS) Eq 11
#RTinv       = np.array( [ 0.]*len(ys) )
#xsP         = np.array( [ 0.]*len(ys) )
#Matrx       = np.array( [ 0.]*len(ys) )      # sum (1-phi)*dz 
#Dtdepo      = np.array( [ 0.]*len(ys) )      # For estimates of activation E 

 
#for i in range( 1,len( xss ), 1 ) :
#https://www.convertunits.com/from/bar/to/grams+per+(square+meter)   
    #conversion = [gf/cm^2*(10^4cm^2/m^2)]*[9.80665bar/(gf/m^2)] # multiplier = 9.80665*10^4 
#    sigma[i] = sigma[i-1] + 1.67*(1. - xss[i] )*( yss[i] - yss[i-1] )*.00980665 #MPa*9.80665*pow(10.,1)
#    temperature[i] = temperature[i-1] + GeothermalGradientMax1*( yss[i] - yss[i-1] )       #deg C 
#    Lamda[i]       = Lamda[i-1]       +sigma[i]/(1. - xss[i])            #MPa
#    Matrix[i]      = Matrix[i-1]      +(1. - xss[i] )*( yss[i] - yss[i-1] )            #meters of solids    
#    Dtdeposit[i]   = ( Matrix[i]*.001 )/KmSpMy    # million years to accumulate matrix to this km depth
#    RTinverse[i]   = -1./( 273.2+temperature[i])/(8.314*.001 )  # -1/kJ/mo
#GeoT_O_DepT    = (AgeRange1[1]-AgeRange1[0])/Dtdeposit[-1]      # About 1/6 for SuluSea 

#if   GeoT_O_DepT < 1. :  # Mya
#    print( 'THE DEPOSITION TIME IS TOO LONG. MUST KEEP "GeoT_o_DepT" > 1.\
#          AND HENCE "KmSpMy" BIGGER.')
#  
#for i in range(  len( xss ) ) :
#    Dtdeposit[i]   =  Dtdeposit[i] + Tau0_1*((AgeRange1[1] + AgeRange1[0])/2. - Dtdeposit[-1 ])
    
#    for k in range( len(ys) )  :    #Interpolate for plotting
#        if  (yss[i-1] <= ys[k]) and (yss[i] >= ys[k] )  :
#            sig[k]          = sigma[i]
#            temp[k], Lam[k] = temperature[i], Lamda[i]
#            Lam[k]          = Lamda[i]
#            Matrx[k]        = Matrix[i]   #meters of solids
#            Dtdepo[k]       = Dtdeposit[i]  # millions of years after section deposition began           
#            RTinv[k]        = -1./( 273.2+temp[k]       )/(8.314*.001 )  # -1/kJ/mo          


##Quantities for getting meters of matrix per million years == Rate for this well:     
#TD  =  yss[ -1 ]
#Matrix_to_TD = Matrix[ -1 ]/ TD
#Rate  =  Matrix[ len(xss)-1 ]/ ( AgeRange1[1] - AgeRange1[0] )      


############ Firstly__Correct_Porosity_for_Surface_Areas_##################### 
#PhiAv = np.mean(xss)      #Normalization constant
#PhiNorm = pow( PhiAv,-1.33*m)* pow( (1.-PhiAv) ,-1.33*n )  #Normalization constant
#print()
#print('m, n, PhiAv, PhiNorm = ', m, n,  '% 5.4f % 5.4f' % ( PhiAv, PhiNorm ))
#print()
#for i in range( len(xss) )   :
#    xssP[i]        = xss[i]* pow( xss[i],-1.33*m)* pow( (1.-xss[i]) ,-1.33*n )/PhiNorm 
#    #lnxssP[i]      = np.log( xssP[i] )      
#    #dPhi_dz[i]     = popt[0] +2.*yss[i]*popt[1]           # For computing E    
#
#for i in range( len(xs) )   : 
#    xsP[i]        = xs[i]* pow( xs[i],-1.33*m)* pow( (1.-xs[i]) ,-1.33*n )/PhiNorm
#    #lnxsP[i]      = np.log( xsP[i] )
#    #dPh_dz[i]     = popt[0] +2.*ys[i]*popt[1]           # For computing E  
#    
#print('max(xssP), min( xssP) = \
#', '% 6.3e %6.3g ' % (np.max(xssP), np.min( xssP)   ))    
#print('max(xsP), min( xsP) = ', '% 6.3e %6.3g ' % ( np.max(xsP), np.min( xsP) ))              


########## Secondly,_Correct_ for Porosity and Temperature #################### New

#for i in range( len(xss) )   :
#    RTinverse[i]  = -1./( 273.2+temperature[i])/(8.314*.001 )  #/kJ mol
#avRTinverse = np.mean(RTinverse)

#for i in range( len(xss) )   :    
#    lnLHS11[i]    = np.log( pow(xss[i],-4.*m/3.)*pow( (1.-xss[i]), -4.*n/3.)*np.log( (1.-xss[i])/(1.- phi0))/Dtdeposit[i] )     
    
#for i in range( len(xs) )   :     
#    RTinv[i]      = -np.array(1./( 273.2 + temp[i])/(8.314*.001))  #/kJ/mol
#avRTinv = np.mean( RTinv )    

#for i in range( len(xs) )   : 
#    lnLS11[i]     = np.log( pow(xs[i],-4.*m/3.)*pow( (1.-xs[i]), -4.*n/3.)*np.log( (1.-xs[i])/(1.- phi0))/Dtdepo[i] )  

#print('Makran 226 max(RTinv), len(RTinv) = ', ' %6.3g %6.3g ' %  ( max(RTinv), len(RTinv) ))  
#print('Makran 227 max(DeDeposit), len(Dtdeposit) = ', ' %6.3g %6.3g ' %  ( max(Dtdeposit), len(Dtdeposit) ))      
#print('Makran 228 max(DtDepo), len(Dtdepo) = ', ' %6.3g %6.3g ' %  ( max(Dtdepo), len(Dtdepo) )) 
 ############## PLOT - 1 : porosity vs depth ################################## New
#fig =  'Figure '+repr(ShaleNumber1+1)+' '

#ans = 'yes'  #Do you want t deg C  plotted?
#lab = 'Fig'+repr(ShaleNumber1+1)
#titl  = titl1
#ylabl = str(ShaleName1)+' porosity'
#xlabl =fig+str(ShaleName1)+' well porosity data trend (Fowler 1985) '
#xDim = ' Depth, m '
#Plotz2( np.array([0., .9, .1]), np.array([0., 400., 50.]), ylabl,\
#      xDim,lab,titl,xs, ys, xss, yss, ShaleName1,temperature, ans  )    
    
################ PLOT 2:  Porosity vs_Lambda_################################### New
#fig = 'Figure '+str(ShaleNumber1+7)+' '
#anss = 'yes'  #Do you want t deg C  plotted?
#labl, lab = '', 'Fig'+str(ShaleNumber1+7)
#titl = ' Age interval is'+str('%4.0f' % AgeRange1[0]) +' -'+ str('%4.0f' % AgeRange1[1])+' Ma.'

#ylabl = 'Porosity '
#xlabl = fig+str(ShaleName1)+''
#ans = 'yes'
#xDim = ' Grain-to-grain stress Lambda, MPa '

#Plotz2( np.array([0., .9, .1]), np.array([0., 400., 50.]), ylabl,\
#      xDim,lab,titl,xs, Lam, xss, Lamda, ShaleName1,temperature,ans  )    
    
################ PLOT 3:  Porosity vs_Lambda, 'surface-corrected' ############ New

#fig = 'Figure '+repr(ShaleNumber1+13)+'     '
#print 'len(xss), len(yss), len(xs), len(ys) = ',    len(xss), len(yss), len(xs), len(ys) 
#print 'len(sigma), len(sig) = ',  len(sigma), len(sig)
#print ' len(temperature), len(Lamda) = ',  len(temperature), len(Lamda)  
#print ' len(Lam) len(Lamda) = ',  len(Lam), len(Lamda)
#labl, lab = '', 'Fig'+repr(ShaleNumber1+13)
#titl =  'T = a + b*depth;  a,b ='+str('%3.1f' % PaleoMudlineT1)+ ',  ' +str('%4.3f' % GeothermalGradientMax1)+\
#    '.  m,n = '+str('%3.2f' % m)+ ',  ' +str('%3.2f' % n)
#ans = 'yes' #Do you want t deg C  plotted?
#ylabl = 'Surface-corrected'+'  porosity'
#xlabl = fig+ShaleName1+'.'
#Plotz2( np.array([0., .9, .1]), np.array([0., .400, .05]), ylabl,\
#      xDim,lab,titl,xsP, Lam, xssP, Lamda, ShaleName1,temperature,ans  )
#########################################################################

####### Porosity Histogram ############################################
#phi_histigram( xssP, ShaleName1 )
#______________________________________________________________________________    
#phi00 = 0.5
#name, entropyx, entropxx, eta_ratio = Entropyx( ShaleName1,xss, xssP  )     
#print('name, entropyx, entropxx, eta_ratio  = ' name,\
#'%6.3e %6.3e %6.3e'  % ( entropyx , entropxx, eta_ratio ))
#______________________________________________________________________________ New

#______________________________________________________________________________ New name

#### Estimate E from the SLOPE of the data/trend 'corrected' points alone: #### New
#Low_Temp = 10  # deg C
#
#print('Makran 310 max(RTinv), len(RTinv) = ', ' %6.3g %6.3g ' %  ( max(RTinv), len(RTinv) ))  
#print('Makran 311 max(DeDeposit), len(Dtdeposit) = ', ' %6.3g %6.3g ' %  ( max(Dtdeposit), len(Dtdeposit) ))      
#print('Makran 312 max(DtDepo), len(Dtdepo) = ', ' %6.3g %6.3g ' %  ( max(Dtdepo), len(Dtdepo) )) 



#Eslopx, Aslopx, axisRT, axislnLS, lnLS11xGen, RTinvx = \
#EAandPlotCoords( temp, Low_Temp, RTinv,  xs, m, n, phi00, Dtdepo)  
#print('Macron 320 ', len(temp), len(RTinv), len(xs), len(Dtdepo))

#print()
#print(' Eslopx,Aslopx, len(axisRT), len(axislnLS), len(lnLS11xGen), len(RTinvx) = ', \
#    ' %5.2e, %5.3e, %5.2e, %5.3e, %5.3e, %5.3e ' % \
#    ( Eslopx,Aslopx, len(axisRT), len(axislnLS), len(lnLS11xGen), len(RTinvx) ))
#print()  

#### Estimate E from the SLOPE of the fitted curve 'corrected' points alone: ##
            
#Eslopexx, Aslopexx, axisRTx, axislnLHSx, lnLS11xxGen, RTinvxx = \
#EAandPlotCoords( temperature, Low_Temp, RTinverse, xss ,m, n,phi00, Dtdeposit)
#print('Macran 334', len(temperature ), len(RTinverse), len(xss), len(Dtdeposit))

#ratex =  Eslopx*Aslopx 
#ratexx = Eslopexx*Aslopexx
#avrate = (ratex+ratexx)/2.


#avE, avA = (Eslopx+ Eslopexx)/2., (Aslopx+ Aslopexx)/2.


#print('  ')

#Aslopxsec = Aslopx/(1000000.*31556952.)
#Aslopxxsec = Aslopexx/(1000000.*31556952.)
#avA = avA/(1000000.*31556952.)

#print 'Aslopxsec = ', '%5.3e' %(Aslopxsec) 
#print('                                                                    ')
#print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#print('________________________________________________________________________')

#print(" See lines 42-45 to change m, n, time derivative, and gradient of T. ")
#print()
#print('    Input-  Low_Temp,  m, n, phi0, phi00, GeothermalGradientMax1,AgeRange1 = ',\
#                  Low_Temp,  m, n, round(phi0,2), phi00, round(GeothermalGradientMax1,3),AgeRange1)
#print()
#print(' Thickness of solid matrix, total depth, and ratio==c  m/m = ', round(Matrix[ len(xss) -1],), round(TD,1), round(Matrix_to_TD,3))    
#print(' Thickness of solid matrix/ Geologic age span m/My == Av Rate for this well = ', round(Rate,3))  
#print()   
#print()
#print('Derived:  Eslopx,  Aslopxsec,   ratex     =   ', ' %6.2e,  %5.2e,  %5.2e ' \
#             % ( Eslopx, Aslopxsec,   ratex ))    
#print('Derived:  Eslopexx,  Aslopxxsec ratexx =  ',  ' %6.2e,  %5.2e, %5.2e ' \
#             % ( Eslopexx,  Aslopxxsec, ratexx))    
#print('Derived:  avE,      avA,      avrate  =  ',  ' %6.2e,  %5.2e, %5.2e ' \
#             % ( avE,     avA,      avrate  ))

#print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx') 

############## PLOT - 0 : Arrhenius plot ###################################### 
#fig =  'Figure '+' '+str(ShaleNumber1+19)

#ans = 'no' #Do you want t deg C  plotted?
#lab = 'Fig'+str(ShaleNumber1+19)
#titl  =  ' Arrhenius plot with slope E ='+repr(round(avE,1)) +' kJ/mol '
#ylabl =  'ln( LHS of Equation 11 )  /Myr'  
#xlabl =fig+'-1/RT.  '+str(ShaleName1)+', '+str(AuthorDate[ShaleNumber1])+'    /kJ/mol'      
#Plotz( axisRT, axislnLS, ylabl,\
#      xlabl,lab,titl,lnLS11xGen , RTinvx,lnLS11xxGen , RTinvxx, ShaleName1  )     
 ##############################################################################

 # p-116, W. McKinney "Python for data analysis" Also p-112-133   
 #https://stackoverflow.com/questions/31983341/using-scientific-notation-in-pandas
#pd.set_option('display.float_format', '{:.1E}'.format)
#pd.Series(data=[10.0])
#print() 
#print(' ',    str(  ShaleName1  )+ ' Summary :'   )
#print( 'Known shale age limits, My = ', AgeRange1, ', ',AgeName )
#print()
#data = {'Parameters':['Total Depth','Solids Thickness',\
#          'Geothermal Gradient','Surface temperature','Surface porosity','Minimum porosity',\
#          'Minimum Temperature', 'Solids deposition rate','m','n','E', 'A'],\
        
#        'Value':[ str(round(yss[-1])), str(round(Matrix[-1])),\
#                 str(GeothermalGradientMax1),repr(PaleoMudlineT1),repr(phi0),repr(phi00),\
#                 repr(Low_Temp),repr(KmSpMy),repr(round(m,2)), repr(round(n,2)),repr(round(avE,1)),avA ],\
            
#        'Units' :[ 'm','m', \
#                  'deg C/m', 'deg C', 'fraction','fraction','deg C','km/My',\
#                  '','', 'kJ/mole','/s'],
            
#        'Comments':[ '', '', '', '','', 'Arbitrary limit', 'Arbitrary limit',\
#                   'Keep > '+ repr(round(Rate,2))+'km/My','Derived', 'Derived', 'Derived', 'Derived']    }
    
#frame=DataFrame(data)

#https://www.google.com/search?client=firefox-b-1-lm&q=print+DataFrame+in+Ipython
#export DataFrame to CSV file
#frame.to_csv(r'/home/ed/Desktop/Z_Code0/SuluSeaData.csv', index=False)
#print(frame)    
#print()
#print() 

#pass































 



















