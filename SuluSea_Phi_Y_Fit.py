#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:31:11 2020
SuluSea_Phi_Y_Fit
@author: ed
"""
import numpy as np
from Arrgh import EAandPlotCoords
#from PLOTZ2 import phi_histigram
from PLOTZ2 import Plotz2
#from PLOT_ENTROPY import Entropyx
from PLOT_ENTROPY import Plotz 
from propertys import shale
import math
from pandas import DataFrame
from scipy.optimize import curve_fit
import pandas as pd
#import shelve

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html 
#Tannant, D.D., 1991. Index properties and compressional-wave velocity variations \
#and correlations for Leg 124. In Silver, E.A., Rangin, C., von Breymann, M.T., et al.,\
#Proc. ODP, Sci. Results, 124: College Station, TX (Ocean Drilling Program), 507–510.\
#doi:10.2973/odp.proc.sr.124.164.1991
#Rutherford, K.J. & M.K. Qureshi, 1981. Geothermal gradient
#map of Southeast Asia 2nd edition – 1981, South East Asia
#Petroleum Exploration Society and Indonesian Petroleum   Copy at McFarlin Library, TU 41 mi.
#ShaleNames       = [ 'Akita',  'Makran1','Makran2', '\"SuluSea\"', 'Oklahoma', 'Maracaibo', 'ODP-DSDP' ] tau
#def fit( 'name' )  :
#

#vvvvvvvvvvvvvvvvvv_Makran_Input_Data_from_'def shale'_vvvvvvvvvvvvvvvvvvvvvvvv
#Create some lists to import data into:
AgeRange1            = [ 0., 0. ]  #My
DepthRange1          = [ 0., 0. ]  #Km
PorosityDepthParams1 = [ 0., 0., 0. ]
ShaleName1, ShaleNumber1, AgeRange1, DepthRange1, PorosityDepthParams1, \
PaleoMudlineT1, GeothermalGradientMax1, TKavg1, TKavgavg,dy_steps,AuthorDate,\
   Low_Temp,Deposition_Rate,AgeName = shale( '\"SuluSea\"' )


#Velde Fig 2b & 1c; x = porosity, y = depth m:                                          
y = np.array([  0.1, 1000.,2000.,3000., 4000.,5000.])
x = np.array([ .6, .29,  .19,  .108, .093, .065 ])
#
#-----------------------------------------------------------------------------
#Variables for tinkering. These over-ride those imported from 'shale' in 'propertys':  
GeothermalGradientMax1 = .03  #.03       #.0185   # .022 
PaleoMudlineT1         = PaleoMudlineT1  # Paleo Mudline t deg C
m,n                    = .9,.8           #  .9,.8
KmSpMy                 = .2 # km solids deposited per My
Tau0_1                 = 1. # Add? (=0,1) remaining geologic time, following section deposition.
#------------------------------------------------------------------------------ 

# Make y increase monotonically:
ys,xs   = list(map( np.array, list(zip(*sorted( zip(y ,x ) ) )) ))
#______________________________________________________________________________ 
   
phi0  = xs[0]

xp = [math.log( x/phi0 ) for x in xs]

#ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
# Scipy uses y = func(x); I use x = func(y) :
a,b,c,d,= -.02395, -.001006, 1.615*np.exp(-5), -9.969*np.exp(-8) 

def func(ys, a, b )   :          #, c, d):
    return a*pow( ys, .4 ) + b*ys  # + c*pow(ys, 1.5) + d*pow(ys, 2.)
#ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
    
#Fit for the parameters a, b,.. of the function func: Marsland p376
sig = np.array( [ .5, .5 ] )
popt, pcov = curve_fit(func, ys, xp, sig  )


#______________________________________________________________________________
#Plot the curve with dy_steps points:
yss = np.linspace(0., np.max(ys),dy_steps )  # Depth m
xss = np.linspace(0.,0., dy_steps  )         # Porosity, fractional

for i in range( len(xss) ) :
    xss[i] = phi0*math.exp( func( yss[i], *popt) )  # Porosity
titl1  =  'ln(porosity/'+repr(phi0)+') = a*depth^0.4 + b^(depth) \n a=%4.2e, b=%4.2e '   % tuple(popt) #, c=%4.2e, d =%4.2e ' % tuple(popt)     

titl1  =  'ln(porosity/0.6) = -3.29*10^-2*depth^0.4  -(2.51*10^-4)^(depth)'     

#______________________________________________________________________________
#print 'len(xss), xss = ', len(xss), xss
#______________________________________________________________________________
#vvvvvvvvvvvvvvvvvvvvvvvv Create all arrays of maximum length VVVVVVVVVVVVVVVVV

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

     for k in range(len(ys)) :  
        # Interpolate for plotting
         if (yss[i-1] <= ys[k]) and (yss[i] >= ys[k]):
             sig[k] = sigma[i]
             temp[k], Lam[k] = temperature[i], Lamda[i]
             Lam[k] = Lamda[i]  # Ln(LHS) Eq 11
             Matrx[k] = Matrix[i]      # meters of solids
             # millions of years after section deposition began:
             Dtdepo[k] = Dtdeposit[i]
             RTinv[k] = RTinverse[i]  # -1/kJ/mo
print(' len RTinv, Dtdepo, xs, temp = ', len(RTinv), len(Dtdepo), len(xs), len(temp)) 

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
                  repr(Low_Temp), repr(KmSpMy), repr(round(m, 2)),\
                      repr(round(n, 2)), repr(round(avE, 1)), avA],

        'Units': ['m', 'm',
                  'deg C/m', 'deg C', 'fraction', 'fraction', 'deg C', 'km/My',
                  '', '', 'kJ/mole', '/s'],

        'Comments': ['', '', '', '', '', 'Arbitrary limit', 'Arbitrary limit',
                     '(Keep>' + repr(round(Rate, 2))+'km/My)', 'Derived', 'Derived',\
                         'Derived', 'Derived']}

frame = DataFrame(data)
print(frame)
#https://www.google.com/search?client=firefox-b-1-lm&q=print+DataFrame+in+Ipython
#export DataFrame to CSV file
#nam = str(ShaleName1)
#frame.to_csv(r'/home/ed/Desktop/Z_Code0/'+ nam +'Data.csv', index=False)

#Chris Fehily,Python, p-323, "shelve":
#nam1 = nam+' '+ str(KmSpMy)
#db = shelve.open(  nam , 'c')
#db[ nam1 ] =  [ avE, avA  ] 
#db.close()
#db = shelve.open( nam )
#print( nam1,'= KmSpMy; E,A =', db[ nam1 ] )
#db.close()


















































