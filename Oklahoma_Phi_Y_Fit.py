#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 19:05:53 2020
Oklahoma_Phi_Y_Fit.py
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
#import math
from pandas import DataFrame
import pandas as pd


from inspect import currentframe, getframeinfo
cf = currentframe()
filename = getframeinfo(cf).filename
print("This is ", filename, ", code line = ", cf.f_lineno) 
print()
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
ShaleNames       = [ 'Akita',  'Makran1','Makran2', 'SuluSea', 'Oklahoma', 'Maracaibo', 'NETailand' ]
#def fit( 'name' )  :


#vvvvvvvvvvvvvvvvvv Oklahoma Input_Data_from_'def shale'_vvvvvvvvvvvvvvvvvvvvvvvv
#Create some lists to import data into:
AgeRange1            = [ 0., 0. ]
DepthRange1          = [ 0., 0. ]
PorosityDepthParams1 = [ 0., 0., 0. ]
ShaleName1, ShaleNumber1, AgeRange1, DepthRange1, PorosityDepthParams1, \
PaleoMudlineT1, GeothermalGradientMax1, TKavg1, TKavgavg,dy_steps, AuthorDate,\
   Low_Temp,Deposition_Rate,AgeName = shale( 'Oklahoma' )

#print 'ShaleName1, ShaleNumber1, AgeRange1, DepthRange1, PorosityDepthParams1, PaleoMudlineT1, \
# GeothermalGradientMax1, TKavg1, TKavgavg,dy_steps,AuthorDate,\
#   Low_Temp,Deposition_Rate  = ', \
#       ShaleName1, ShaleNumber1, AgeRange1, DepthRange1, PorosityDepthParams1, PaleoMudlineT1, \
#       GeothermalGradientMax1, TKavg1, TKavgavg,dy_steps,AuthorDate,\
#   Low_Temp,Deposition_Rate

    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx_Athy_Oklahoma_xxxxxxxxxxxxxxxxxxxxxxxxxx    
#
#L. F. Athy, DENSITY, POROSITY, AND COMPACTION OF SEDIMENTARY ROCKS,BULLETIN
#of the AMERICAN ASSOCIATION OF PETROLEUM GEOLOGISTS v14, Number 1, pp 1-24.
#Figure 2:   
#y =Depth ft*0.3048 = m, x=fractional porosity. rhog = 2.7, rhow = 1.
Density = 1.40, 2.08, 2.245, 2.38, 2.480,  2.56, 2.60,  2.615, 2.63
Depth   =    0., 1750.,2000.,  3200., 4000., 5200., 6000., 6400.   
                                                                                       
# Athy Figure 3, depth ft. 'Data' shallower than 1190' (363 m) are extrapolated.
x     = .48,  .40,  .311,  .26,   .20,  .10,  .126,   .084,  .05,   .03,   .022  
yft   =  0.,  407.,  1000.,1190., 2000., 3496.,3000.,  4000., 5200., 6000., 6800. 

#Variables for tinkering. These over-ride those imported from 'shale': 
GeothermalGradientMax1  = .06             # .06,.135   .12
PaleoMudlineT1          = PaleoMudlineT1  # PaleoMudlineT1
m,n                     =  .85, 1.        #  .85, 1.
KmSpMy                 = .2 # km solids deposited per My
Tau0_1                 = 1. # Add? (=0,1) remaining geologic time, following section deposition.
#------------------------------------------------------------------------------




y = [w*.3048 for w in yft]   #meters depth

# Make y increase monotonically:
ys,xs   = list(map( np.array, list(zip(*sorted( zip(y ,x ) ) )) ))
#______________________________________________________________________________
#print  "Line# =", cf.f_lineno, ' ys,xs = ', ys, xs


#______________________________________________________________________________

# Scipy uses y = func(x); I use x = func(y) :
a,b,c,bx,cx = .49, .0008, -.00302, 1.05, .9

#print "Line# =", cf.f_lineno, ' porosity = a+b*ys+c*pow(ys,', repr(cx),') ' 
def func(ys, a, b, c):
    bx, cx = 1.05, .9
    return a+b*pow(ys,bx) +c*pow(ys, cx)

#Fit for the parameters a, b,.. of the function func: Marsland p376
sg = np.ones( 3 )*.5  # Error estimates for popt variables
popt, pcov = curve_fit(func, ys, xs, sg  )
#______________________________________________________________________________
#Plot the curve with dy_steps points:
maxys = np.max(ys)
yss = np.linspace(0., maxys,dy_steps )
xss = np.zeros( dy_steps )
for k in range(len(yss)) :
    xss[k] = func(yss[k],*popt)
titl1  =  'Porosity = a + b*depth^('+repr(bx)+')+c*depth^(' +repr(cx)+'  ) \n a=%4.2e, b=%4.2e, c=%4.2e' % tuple(popt) 
titl1 = ' titl1 = ', titl1
titl1  =  'Porosity = 0.490 + 8.03*10^-4*depth^1.05 -3.01*10^-3 *depth^(0.9 )' 
phi0 = xss[0]
############### Find where to Slice Off Extrapolated data######################

## Oklahoma only: Must not treat 'data' at less than 363m as real in entropy 
#analyses.  Slice it off after using it to compute temperature, sigma, geologic age, etc. 

slice,slic = 0,0
for i in range( len(yss) )  :
    if  yss[i] < 363 :  #Only extrapolated 'data' exists below 363m
        slice = i
    else  :
        pass
for k in range( len(ys) )  :
    if  ys[k] < 363 :  #Only extrapolated 'data' exists below 363m
        slic = k
    else :
        pass
    
#print 'name, slice, slic = ', cf.f_lineno, ShaleName1, slice, slic  # 34, 3

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
    #Center the deposition time in the geologic time interval:
    Dtdeposit[i] = Dtdeposit[i] + Dtdeposit[-1]/2. +\
         Tau0_1*(AgeRange1[1] + AgeRange1[0] - Dtdeposit[-1])/2.

    for k in range(len(ys )): 
        # Interpolate for plotting
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

phi00 = min(phi0, 0.5)   # probably chaotic if greater
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
                     '(Keep>' + repr(round(Rate, 2))+'km/My)', 'Derived', 'Derived',\
                         'Derived', 'Derived']}

frame = DataFrame(data)

#https://www.google.com/search?client=firefox-b-1-lm&q=print+DataFrame+in+Ipython
#export DataFrame to CSV file
nam = str(ShaleName1)
frame.to_csv(r'/home/ed/Desktop/Z_Code0/'+ nam +'Data.csv', index=False)
print(frame)
print( ' THE END  ')  
pass



#Data Fiits:
#sigma       =  np.array( [ 0.]*len(yss) )  # Fitted porosities vs depth
#temperature =  np.array( [ 0.]*len(yss) )  # Temperature deg C vs depth
#temperature[0] = PaleoMudlineT1
#Lamda       =  np.array( [ 0.]*len(yss) )   # grain_to_grain force per unit contact area
#lnLHS11     = np.array( [ 0.]*len(yss) )    #Ln(LHS) Eq 11
#d1mphidt  = np.array( [ 0.]*len(yss) ) 
#xssP      = np.array( [ 0.]*len(yss) )      #Porosity with 'porosity correction'
#RTinverse = np.array( [ 0.]*len(yss) )      # - 1/RT  1/kJ mol
#lnxssP    = np.array( [ 0.]*len(yss) )      # ln(xssP)
#Matrix    = np.array( [ 0.]*len(yss) )      # sum (1-phi)*dz
#dPhi_dz   = np.array( [ 0.]*len(yss) )      # For estimates of activation E 

#Data:
#sig         =  np.array( [ 0.]*len(ys) )    # Fitted porosities vs depth
#temp        =  np.array( [ 0.]*len(ys)  )   #  Temperature deg C vs depth
#temp[0]     =  PaleoMudlineT1
#Lam         =  np.array( [ 0.]*len(ys) )    # grain_to_grain force per unit contact area
#lnLS11      =  np.array( [ 0.]*len(ys) )    #Ln(LHS) Eq 11
#xsP         = np.array( [ 0.]*len(ys) )
#RTinv       = np.array( [ 0.]*len(ys) )
#lnxsP       = np.array( [ 0.]*len(ys) )
#Matrx       = np.array( [ 0.]*len(ys) )      # sum (1-phi)*dz 
#dPh_dz      = np.array( [ 0.]*len(ys) )      # For estimates of activation E 

#for i in range( 1,len( xss ), 1 ) :
#https://www.convertunits.com/from/bar/to/grams+per+(square+meter)   
    #conversion = [gf/cm^2*(10^4cm^2/m^2)]*[9.80665bar/(gf/m^2)] # multiplier = 9.80665*10^4 
#    sigma[i] = sigma[i-1] + 1.67*(1. - xss[i] )*( yss[i] - yss[i-1] )*.00980665 #MPa
#    temperature[i] = temperature[i-1] + GeothermalGradientMax1*( yss[i] - yss[i-1] )       #deg C 
#    Lamda[i]       = Lamda[i-1]       +sigma[i]/(1. - xss[i])                            #kilobar force
#    Matrix[i]      = Matrix[i-1]      +(1. - xss[i] )*( yss[i] - yss[i-1] )                #meters of solids    
#    dPhi_dz[i]     = 1.05*pow( yss[i], .05 )*popt[1] + .9*pow( yss[i], -.1 )*popt[2]            # For computing E   
    
#    for k in range( len(ys) )  :    #Interpolate for plotting
#        if  (yss[i-1] <= ys[k]) and (yss[i] >= ys[k] )  :
#            sig[k] = sigma[i]
#            temp[k], Lam[k] = temperature[i], Lamda[i]
#            Lam[k]  = Lamda[i]
#            Matrx[k]  = Matrix[i]   #meters of solids
            
#Quantities for getting meters of matrix per million years == Rate for this well:            
#TD  =  yss[ len(xss)-1 ]
#Matrix_to_TD = Matrix[ len(xss)-1 ]/ TD
#Rate  =  Matrix[ len(xss)-1 ]/ ( AgeRange1[1] - AgeRange1[0] )      
#dz_dt      = yss[-1]*2./(AgeRange1[0]+AgeRange1[1])  # For estimates of activation E
#Average_Age = ( AgeRange1[0]+AgeRange1[1])/2  
############# Firstly__Correct_Porosity_for_Surface_Areas_#####################
#PhiAv = np.mean(xss)      #Normalization constant
#PhiNorm = pow( PhiAv,-1.33*m)* pow( (1.-PhiAv) ,-1.33*n )  #Normalization constant
#print('m, n, PhiAv, PhiNorm = ',cf.f_lineno , m, n,  '% 5.4f % 5.4f' % ( PhiAv, PhiNorm ), end=' ')
#print('                                                               ')
#for i in range( len(xss) )   :
#    xssP[i]        = xss[i]* pow( xss[i],-1.33*m)* pow( (1.-xss[i]) ,-1.33*n )/PhiNorm 
#    lnxssP[i]      = np.log( xssP[i] )      
#    dPhi_dz[i]     = 1.05*pow( yss[i], .05 )*popt[1] + .9*pow( yss[i], -.1 )*popt[2]             # For computing E   
    
#for i in range( len(xs) )   : 
#    xsP[i]        = xs[i]* pow( xs[i],-1.33*m)* pow( (1.-xs[i]) ,-1.33*n )/PhiNorm
#    lnxsP[i]      = np.log( xsP[i] )
#    dPh_dz[i]     = 1.05*pow( ys[i], .05 )*popt[1] + .9*pow( ys[i], -.1 )*popt[2]       # For computing E   
    
#print('max(xssP), min( xssP) = \
#', '% 6.3e %6.3g ' % (np.max(xssP), np.min( xssP)   ))  
#print('   ')  
#print('max(xsP), min( xsP) = ', '% 6.3e %6.3g ' % ( np.max(xsP), np.min( xsP) ))              
#print('   ')

########## Secondly,_Correct_ for Porosity and Temperature #################### New


#for i in range( len(xss) )   :
#    RTinverse[i]  = -1./( 273.2+temperature[i])/(8.314*.001 )  #kJ/mo
#avRTinverse = sum(RTinverse)/len(RTinverse)
#for i in range( len(xss) )   :     
#    lnLHS11[i]    = np.log( pow(xss[i],-4.*m/3.)*pow( (1.-xss[i]), -4.*n/3.)*np.log( (1.-xss[i])/(1.- .81))/Average_Age )     
    
#for i in range( len(xs) )   :
#    RTinv[i]      = -np.array(1./( 273.2 + temp[i])/(8.314*.001))  #kJ/mol
#    lnLS11[i]     = np.log( pow(xs[i],-4.*m/3.)*pow( (1.-xs[i]), -4.*n/3.)*np.log( (1.-xs[i])/(1.- .81))/Average_Age )  
#
#print(' max(RTinv), min(RTinv) = ', ' %6.3g %6.3g ' %  ( np.max(RTinv), np.min(RTinv) ))  
      
############## Oklahoma: Thirdly slice off the top 1200 ft of data: ########### New
############## Oklahoma: Thirdly slice off the top 1200 ft of data: ########### New   

#print 'len(xss), len(yss), len(xs), len(ys) = ', cf.f_lineno ,    len(xss), len(yss), len(xs), len(ys)        
#sigma, temperature = sigma[slice:],temperature[slice:]
#xssP        = xssP[slice:]
#RTinverse          = np.array(RTinverse[slice:])
#lnLHS11            = lnLHS11[slice:]

#print ' len(sigma), len(temperature) = ', len(sigma), len(temperature)

#Lamda, sig, temp, Lam = Lamda[slice:],sig[slic:], temp[slic:], Lam[slic:]
#yss, xss, ys, xs      = yss[slice:], xss[slice:], ys[slic:], xs[slic:]
#xsP, RTinv     =   xsP[slic:], RTinv[slic:]
#lnLS11   = lnLS11[slic:] 

#print('len(xss), len(yss), len(xs), len(ys) = ', cf.f_lineno ,    len(xss), len(yss), len(xs), len(ys)) 
#print('                                                   ')
#print('len(sigma), len(sig) = ', cf.f_lineno ,  len(sigma), len(sig))
#print('                                    ')
#print(' len(temperature), len(Lamda) = ',cf.f_lineno ,  len(temperature), len(Lamda))   
#print('                              ') 
############## Oklahoma: Thirdly slice off the top 1200 ft of data: ########### New
############## Oklahoma: Thirdly slice off the top 1200 ft of data: ########### New


 ############## PLOT - 1 : porosity vs depth ################################## New

#fig =  'Figure'+' '+repr(ShaleNumber1+1)+' '

#ans = 'yes'  #Do you want t deg C  plotted?
#lab = 'Fig'+repr(ShaleNumber1+1)
#print("This is ", filename, ", code line = ", cf.f_lineno) 
#print('lab = ' , lab)
#print()
#titl  = titl1
#ylabl = str(ShaleName1)+' porosity'
#xlabl =fig+' '+str(ShaleName1)+' well porosity data trend (Athy 1930)'
#maxdepth = yss[-1] + 50. - np.divmod( (yss[-1]+50.),50.)[1]
#htk = np.array([0., maxdepth+50, 50.])
#htk = np.array([0., 500., 50.])
#print("This is ", filename, ", code line = ", cf.f_lineno) 
#print('len(lab), len(ylabl), len(xlabl),maxdepth, len(titl1),len(htk) = ',\
#len(lab), len(ylabl), len(xlabl), maxdepth, len(titl1),len(htk))
#xDim = ' Depth, m '

#Plotz2( np.array([0., .9, .1]), htk, ylabl,\
#      xDim,lab,titl,xs, ys, xss, yss, ShaleName1,temperature, ans  )    
#print("This is ", filename, ", code line = ", cf.f_lineno)     
################ PLOT 2:  Porosity vs_Lambda_################################### New
#fig = 'Figure'+' '+str(ShaleNumber1+7)+' '
#anss = 'yes'  #Do you want t deg C  plotted?
#lab = 'Fig'+str(ShaleNumber1+7)
#titl = ' Average of age interval is'+str('%4.0f' % Average_Age)+' Ma.'

#ans = 'yes'
#xlabl = fig+' '+ShaleName1+'.'
#print(' len(xlabl), len(fig), len(ylabl), len(titl) = ',  len(xlabl), len(fig), len(ylabl), len(titl))
#print()
#print('len(lab), len(ylabl), len(xlabl),maxdepth, len(titl1),len(htk) = ',\
#len(lab), len(ylabl), len(xlabl), maxdepth, len(titl1),len(htk))
#print()
#xDim = ' Grain-to-grain stress Lambda, MPa '

#Plotz2(  np.array([0., .9, .1]), np.array([0., 400., 50.]), ylabl,\
#      xDim,lab,titl,xs, Lam, xss, Lamda, ShaleName1,temperature,ans  )    
    
################ PLOT 3:  Porosity vs_Lambda, 'surface-corrected' ############ New

#fig = 'Figure'+' '+repr(ShaleNumber1+13)+'     '

#labl, lab = '','Fig'+repr(ShaleNumber1+13)

#titl =  'T deg C = a + b*depth;  a,b ='+str('%4.0f' % PaleoMudlineT1)+ ',  ' +str('%4.3f' % GeothermalGradientMax1) 
#titl =  'T = a + b*depth;  a,b ='+str('%3.1f' % PaleoMudlineT1)+ ',  ' +str('%4.3f' % GeothermalGradientMax1)+\
#    '.  m,n = '+str('%3.2f' % m)+ ',  ' +str('%3.2f' % n)

#ans = 'yes' #Do you want t deg C  plotted?
#ylabl = 'Surface-corrected'+'  porosity'
#xlabl = fig+' '+ShaleName1+'. '
#print('len(lab), len(ylabl), len(xlabl),maxdepth, len(titl1),len(htk) = ',\
#len(lab), len(ylabl), len(xlabl), maxdepth, len(titl1),len(htk))
#print()
#xDim = ' Grain-to-grain stress Lambda, MPa '

#Plotz2( np.array([0., .9, .1]), np.array([0., .400, .05]), ylabl,\
#      xDim,lab,titl,xsP, Lam, xssP, Lamda, ShaleName1,temperature,ans  )


################ PLOT 4:  Porosity vs_Lambda, T and surface-corrected ######### New

#fig = 'Fig'+' '+repr(ShaleNumber1+19)+'     '
#ans = 'yes' #Do you want t deg C  plotted?
#titl =   'Exponents m and n are '+repr(m)+','+repr(n)
#labl, lab =  '', 'Fig'+repr(ShaleNumber1+19)
#ylabl, xlabl = 'Fully corrected porosity', fig+' '+' '+ShaleName1+'.    Lambda kbar'    
#Plotz2( popt, np.array([0., .9, .1]), np.array([0., 400., 50.]), ylabl,\
#      xlabl,lab,titl,xsTP, Lam, xssTP, Lamda, ShaleName1,temperature,ans  ) 

############### Plot 5: Transformed  Porosity Histogram ####################### New

#( xssP, ShaleName1 )


#______________________________________________________________________________ 
#phi00 = min(phi0, 0.5)
#name, entropyx, entropxx, eta_ratio = Entropyx( ShaleName1,xss, xssP  )     
  
#print()
#______________________________________________________________________________ New
#
#______________________________________________________________________________ New

 ############## PLOT - 0 : Arrhenius plot ##################################### 
#fig =  'Figure '+' '+str(ShaleNumber1+19)
#
#print()
#ans = 'no' #Do you want t deg C  plotted?
#lab = 'Fig'+str(ShaleNumber1+19)
#titl  =  ' Arrhenius plot with slope E kj/mol '
#ylabl =  'ln( LHS of Equation 11 )  /Myr'  
#xlabl =fig+'    -1/RT.  '+str(ShaleName1)+', '+str(AuthorDate[ShaleNumber1])+''
#xDim = ' -1/RT,   1/(kJ/mol) '
#### Estimate E from the SLOPE of the data/trend 'corrected' points alone: #### New
#### Estimate E from the SLOPE of the data/trend 'corrected' points alone: #### New

#
#Eslopx,lnTAUx, Aslopx, axisRT, axislnLS, lnLS11xGen, RTinvx = \
#EAandPlotCoords( temp,       Low_Temp,     RTinv,  xs, m, n, Average_Age, phi00, Method,dPh_dz, ys)  

#print("This is ", filename, ", code line = ", cf.f_lineno) 
#print()
#print(' Eslopx,Aslopx, len(axisRT), len(axislnLS), len(lnLS11xGen), len(RTinvx) = ', \
#    ' %5.2e, %5.3e, %5.2e, %5.3e, %5.3e, %5.3e ' % \
#    ( Eslopx,Aslopx, len(axisRT), len(axislnLS), len(lnLS11xGen), len(RTinvx) ))
#print()  

#### Estimate E from the SLOPE of the fitted curve 'corrected' points alone: ## New    
#### Estimate E from the SLOPE of the fitted curve 'corrected' points alone: ## New 
            
#Eslopexx, lnTAUxx, Aslopexx, axisRTx, axislnLHSx, lnLS11xxGen, RTinvxx = \
#EAandPlotCoords( temperature, Low_Temp, RTinverse, xss ,m, n, Average_Age,phi00, Method,dPhi_dz, ys )

#ratex =  Eslopx*Aslopx 
#ratexx = Eslopexx*Aslopexx
#avrate = (ratex+ratexx)/2.
#avE, avA = (Eslopx+ Eslopexx)/2., (Aslopx+ Aslopexx)/2.
#avlnTAU    = (lnTAUx+lnTAUxx)/2.

#print("This is ", filename, ", code line = ", cf.f_lineno) 
#print('  ')
#print('avE, avA,Eslopx, Eslopexx, Aslopx, Aslopexx = ',cf.f_lineno,\
#' %5.3e, %5.3e, %5.3e, %5.3e, %5.3e, %5.3e'\
# % ( avE,avA, Eslopx, Eslopexx, Aslopx, Aslopexx ))

#print('                                                                    ')
#print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#print('________________________________________________________________________')
#print("This is ", filename, ", code line = ", cf.f_lineno) 
#print(" See lines 42-45 to change m, n, time derivative, and gradient of T. ")
#print()
#print('    Input-  Low_Temp,  m, n, phi0, phi00, GeothermalGradientMax1,AgeRange1 = ',\
#                  Low_Temp,  m, n, round(phi0,2), phi00, round(GeothermalGradientMax1,3),AgeRange1)
#print()
#print(' Thickness of solid matrix, total depth, and ratio==c  m/m = ', round(Matrix[ len(xss) -1],), round(TD,1), Matrix_to_TD)    
#print(' Thickness of solid matrix/ Geologic age span m/My == Av Rate for this well = ', round(Rate,3))  
#print()
#print(' Method # = Code in "EAandPlotCoords" that evaluates dln(1-phi)/dt  =   ', Method)   
#print()
#print('Derived:  Eslopx, lnTAUx  Aslopx,   ratex     =   ', ' %6.2e, %6.2e,  %5.2e,  %5.2e ' \
#             % ( Eslopx, lnTAUx, Aslopx,   ratex ))    
#print('Derived:  Eslopexx, lnTAUxx, Aslopexx, ratexx =  ',  ' %6.2e, %6.2e,  %5.2e, %5.2e ' \
#             % ( Eslopexx, lnTAUxx, Aslopexx, ratexx))    
#print('Derived:  avE,  avlnTAU,    avA,      avrate  =  ',  ' %6.2e, %6.2e,  %5.2e, %5.2e ' \
#             % ( avE,  avlnTAU,    avA,      avrate  ))
#print()
#print('Derived:  name,                             entropyx, entropxx, eta_ratio  = ',\
#                 name, ' %6.2e  %6.2e  %6.2e'  % ( entropyx, entropxx, eta_ratio ))  
#
#print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx') 

#Plotz( popt, axisRT, axislnLS, ylabl,\
#      xDim,lab,titl,lnLS11xGen , RTinvx,lnLS11xxGen , RTinvxx, ShaleName1  )      

