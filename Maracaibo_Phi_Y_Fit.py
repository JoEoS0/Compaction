#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 11:35:59 2020
Maracaibo_Phi_Y_Fit.py
@author: ed
"""

from Arrgh import EAandPlotCoords
#from PLOTZ2 import phi_histigram
from PLOTZ2 import Plotz2
#from PLOT_ENTROPY import Entropyx # bar
from PLOT_ENTROPY import Plotz 
from propertys import shale
import numpy as np
#import math
from scipy.optimize import curve_fit
from pandas import DataFrame
import pandas as pd


#    ShaleNames       = [ 'Akita',  'Makran1','Makran2', ''SuluSea'', 'Oklahoma', 'Maracaibo' ]

#vvvvvvvvvvvvvvvvvv_Makran_Input_Data_from_'properties/def shale'_vvvvvvvvvvvvv Rate
#Create some lists to import data into:
AgeRange1            = [ 0., 0. ]
DepthRange1          = [ 0., 0. ]
PorosityDepthParams1 = [ 0., 0., 0. ]
ShaleName1, ShaleNumber1, AgeRange1, DepthRange1, PorosityDepthParams1, \
PaleoMudlineT1, GeothermalGradientMax1, TKavg1, TKavgavg,dy_steps,AuthorDate,\
   Low_Temp,Deposition_Rate, AgeName = shale( 'Maracaibo' )

#print 'ShaleName1, ShaleNumber1, AgeRange1, DepthRange1, PorosityDepthParams1, PaleoMudlineT1, \
# GeothermalGradientMax1, TKavg1, TKavgavg,dy_steps,AuthorDate,Low_Temp,Deposition_Rate  = ', \
#       ShaleName1, ShaleNumber1, AgeRange1, DepthRange1, PorosityDepthParams1, PaleoMudlineT1, \
#       GeothermalGradientMax1, TKavg1, TKavgavg,dy_steps,AuthorDate,Low_Temp,Deposition_Rate


#HHHHHHHHHHHHHHHHHHHHHH_Hedberg Maracaibo:Grain density, Depth, PorosityHHHHHHH    
#Hedberg, H.D., 1936. Gravitational compaction of clays and shales.
#American Journal of Science 31 (184), 241–287. p 279 for rho_g and p-254 for porosity-depth Maricaibo:
#y =Depth ft*0.3048 = m, x=fractional porosity. rhog = 2.666, rhow = 1.
rhogs = [2.636,2.665,2.671,2.623,2.525, 2.371,2.588,2.716,2.714,2.657, 2.719,2.652,2.597,2.681,2.701, \
                 2.686,2.687,2.735,2.732,2.720, 2.751,2.701,2.701,2.733,2.690, 2.733,2.642,2.645,2.633,2.672, \
                 2.654,2.703,2.732,2.648,2.607 ]
rhog = np.mean(rhogs)
print('rhog = ' , "%.4g" % rhog)  # = 2.666g/cc
y = 291.,472.,497.,511.,862.,        922.,1637.,1805.,1920.,2031.,  2146.,2146.,2200.,2480.,2605.,\
    2780.,2818.,2996.,3015.,3094.,  3293.,3313.,3353.,3353.,3521.,  3702.,3973.,4336.,4608.,4849., \
    5007.,5035.,5389.,5502.,6013.,  6081.,6175.  # depth,ft
    
x = .3363,.3583,.3385,.3314,.3347,  .3133,.2776,.2738,.2875,.2662,  .2887,.2590,.2468,.2519,.2494, \
    .2532,.2288,.2423,.2561,.2489,  .2063,.2124,.1780,.1813,.2060,  .2220,.1855,.1777,.1680,.1420, \
    .1460,.1280,.1303,.1364,.0907,  .0918,.1060  #fractional porosity
y = [.3048*y for y in y]  # convert depth from feet to meters Fehily p242. 

###############################################################################
#Variables for tinkering. These over-ride those imported from 'shale': 
GeothermalGradientMax  = .05             #   .05  Other values used: .084  .046 .05
PaleoMudlineT1         = PaleoMudlineT1  # PaleoMudlineT1
m,n                    = .9, .9          # .9,.9 Trial and error picked exponents for equation 4,5. 
KmSpMy                 = .2 # km solids deposited per My
Tau0_1                 = 1. # Add? (=0,1) remaining geologic time, following section deposition.
#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH    

# Make y increase monotonically, keeping porosity and depth matched :  
ys,xs   = list(map( np.array, list(zip(*sorted( zip(y ,x ) ) )) ))  #Fehily p242.
#______________________________________________________________________________
#print ' ys,xs = ',   ys,  xs
#print 'max(ys), min(ys), Line# = ', '%.4g' % max(ys),'%.4g' % min(ys),  cf.f_lineno  
#______________________________________________________________________________

# Scipy uses y = func(x); I use x = func(y) since porosity == x is plotted vertically :

#print "Line# =", cf.f_lineno, ' porosity =  a+b*ys ' 
a, b, bx = .359, -.000141, 1.  # trial values 
def func(ys,a, b, bx=1. ):
    return a+b*pow(ys,bx)

#print "Line# =", cf.f_lineno, ' porosity = a+b*pow(ys,', repr(bx),') ' 

#Fit for the parameters a, b,.. of the function func: Marsland Machine Learning p376
sig = np.ones( 2 )*.05
popt, pcov = curve_fit(func, ys, xs, sig  )
#popt, pcov = curve_fit(func ys, xs, sig  )
#______________________________________________________________________________ phi0
print(' popt = ',    popt)
#Plot the curve with dy_steps points:
maxys = np.max(ys)
yss = np.linspace(0., maxys,dy_steps ) # == depths,m
xss = np.zeros( dy_steps )             # == porosity (fractional)
for k in range(len(yss)) :
    xss[k] = func(yss[k],*popt)
phi0 = xss[0]
lab = ''
titl1  =  'Porosity = a + b*depth^('+repr(bx)+' ) \n a=%4.2e, b=%4.2e' % tuple(popt)
print(' titl1 = ', titl1)
titl1  =  'Porosity = 0.359 - 1.41*10^-4*depth'

 
ylabl = str(ShaleName1)+' porosity'


# Get std_phi_error = sqrt( sum( ( (phi - phi_fit)**2/( samples-1 ) )  )
s = 0.
est_phi = np.zeros( len(xs) ) 
for q in range( len(xs) ) : 
    est_phi[q] = func(ys[q], *popt) 
    s = s + pow( ( xs[q]- est_phi[q] ), 2 )    
std_phi = np.sqrt( s/( len(xs)-1.) ) 
#fit_std = '%4.3f' % std_phi
print('    porosity experimental fit-std = ','%4.3f' %  std_phi)

######### Create empty arrays for curve fits and data #########################
#vvv_Get_ Sigma,_MatrixMass,_Temperature,Lambda & Arrhenius Profiles_ln(LHS) Eq11 vvv
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




























