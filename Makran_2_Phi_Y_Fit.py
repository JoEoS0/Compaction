

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 19:59:46 2020
Makran_2_Phi_y_Fit.py
@author: ed
"""
from Arrgh import EAandPlotCoords
#from PLOTZ2 import phi_histigram
from PLOTZ2 import Plotz2
#from PLOT_ENTROPY import Entropyx
from PLOT_ENTROPY import Plotz 
from propertys import shale
import numpy as np
from pandas import DataFrame
from scipy.optimize import curve_fit
#from inspect import currentframe, getframeinfo
import pandas as pd

ShaleNames       = [ 'Akita',  'Makran1','Makran2', 'SuluSea', 'Oklahoma', 'Maracaibo', 'NETailand' ]
#def fit( 'name' )  :

#vvvvvvvvvvvvvvvvvv_Makran_Input_Data_from_'def shale'_vvvvvvvvvvvvvvvvvvvvvvvv x1s
#Create some lists to import data into:
AgeRange2            = [ 0., 0. ]
DepthRange2          = [ 0., 0. ]
PorosityDepthParams2 = [ 0., 0., 0. ]
#Import needed data:
ShaleName2, ShaleNumber2, AgeRange2, DepthRange2, PorosityDepthParams2, \
PaleoMudlineT2, GeothermalGradientMax2, TKavg2, TKavgavg,dy_steps,AuthorDate,\
   Low_Temp,Deposition_Rate, AgeName = shale( 'Makran2' )

#print 'ShaleName2, ShaleNumber2, AgeRange2, DepthRange2, PorosityDepthParams2, PaleoMudlineT2, \
# GeothermalGradientMax2, TKavg2, TKavgavg ,dy_steps,AuthorDate,\
#   Low_Temp,Deposition_Rate = ', \
#       ShaleName2, ShaleNumber2, AgeRange2, DepthRange2, PorosityDepthParams2, PaleoMudlineT2, \
#       GeothermalGradientMax2, TKavg2, TKavgavg,dy_steps,AuthorDate

#^^^^^^^^^^^^^^^^^^_Makran_Input_Data_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
# x1 is accretionary prism porosity:
x1 = np.array([.29,.16,.11,.07,.11,.07,.67,.28,.17,.15,.42,.18,.46,\
                       .35,.19,.40,.20,.16,.08,.50,.18,.12,.09])
                   

# y1 is accretionary prism depth in km:
y1   = np.array([0.80,1.64,2.20,3.49,2.35,3.99,0.21,0.84,1.84,1.57,0.55,\
            1.12,0.23,0.45,0.92,0.45,1.11,1.75,3.13,0.36,1.04,1.76,2.81])
###############################################################################          

#Variables for tinkering. These over-ride those imported from 'shale':  
GeothermalGradientMax2   = .03    #.03     #.02  
PaleoMudlineT2           = PaleoMudlineT2  # PaleoMudlineT2
m,n                      = .85, .95        #   .85, .95 
KmSpMy                   = 1.              # Km solid sediment deposited per My 
Tau0_1                   = 1.              # Add? (=0,1) remaining geologic time, following section deposition.    
      
# Here ys increases monotonically. The matching x1s aren't necessarily monotonic:
ys,xs   = list(map( np.array, list(zip(*sorted( zip(y1 ,x1 ) ) )) ))
ys = [w*1000. for w in ys]   #km to m depth
#^^^^^^^^^^^^^^^^^^_MakRan2_Porosity-Depth_Input_Data_^^^^^^^^^^^^^^^^^^^^^^^^^^  
#vvvvvvvvv_Abyssal_Plain_Porosity_vs_Depth_Fit_vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv 

#Compute the  log of experimental porosities, 'map' = Fehily p-242:
phi0  = 0.6    
phi0x = phi0
xp = [np.log( x/phi0x ) for x in xs] 

# Curve fit abyssal plain porosity data :
a = -.00124
b = 1.715*np.exp(-7)    
print( ' ln(porosity/.7) = a*ys +b*pow(ys, 2.) ') 
print()
def test(ys, a, b )  :
    return a*ys +b*pow(ys, 2.)   

#Fit for the parameters d, e of the function func: Marsland p376
sig = np.ones( 2 )*.5
popt, pcov = curve_fit(test, ys, xp, sig  )
print()

print( ' popt = ', popt)
print()

#Plot the fitted curve with dy_steps points:
yss = np.linspace(0., max(ys), dy_steps )
xss = np.linspace(0.,0.,dy_steps )

for q in range( len(xss) ) : 
    xss[q] = phi0x*np.exp( test(yss, *popt)[q] )    #porosity
    
#print ' len(xss), len(yss), xss, yss = ', cf.f_lineno , len(xss), len(yss), xss, yss
print() 
# Get std_phi_error = sqrt( sum( ( (phi - phi_fit)**2/( samples-1 ) )  )
s = 0.
est_phi = np.zeros( len(xs) ) 
for q in range( len(xs) ) : 
    est_phi[q] = phi0x*np.exp( test(ys[q], *popt) )
    s = s + pow( ( xs[q]- est_phi[q] ), 2 )    
std_phi = np.sqrt( s/( len(xs)-1.) ) 
fit_std = '%4.3f' % std_phi

#_ln(xs/.7 = _-.00124*ys+1.71*pow(10.,-7)*pow(ys,2.) 10/17/2020
titl1  =  'ln(porosity/'+repr(phi0x)+') = a*depth + b*depth^2 \n a=%4.2e, b=%4.2e' % tuple(popt)+ ', porosity std='+str(fit_std)+')'
print('titl1 = ', titl1)
titl1 = 'ln(porosity/0.6) = -1.07*10^-3*depth +1.34*10^-7*depth^2\n porosity standard deviation = 0.052'

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
temperature[0] = PaleoMudlineT2
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
temp[0] = PaleoMudlineT2
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
    temperature[i] = temperature[i-1] + GeothermalGradientMax2 *\
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

GeoT_O_DepT = (AgeRange2[1]-AgeRange2[0])/Dtdeposit[-1]      # Has to be > 1.

if GeoT_O_DepT < 1.:  # Mya
    print('THE DEPOSITION TIME IS TOO LONG. MUST KEEP "GeoT_o_DepT" > 1.\
          AND HENCE "KmSpMy" BIGGER.')

for i in range(len(xss)) :
    # Center the deposition time in the geologic time interval:
    Dtdeposit[i] = Dtdeposit[i] + Dtdeposit[-1]/2. +\
        Tau0_1*(AgeRange2[1] + AgeRange2[0] - Dtdeposit[-1])/2.


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
fig = 'Figure'+'  '+repr(ShaleNumber2+1)

ans = 'yes'  # Do you want t deg C  plotted?
lab = 'Fig'+repr(ShaleNumber2+1)
titl = titl1
ylabl = str(ShaleName2)+' porosity'
xlabl = fig+str(ShaleName2)+' well porosity data trend (Velde 1996)'
xDim = ' Depth, m '

Plotz2(np.array([0., .9, .1]), np.array([0., 400., 50.]), ylabl,
       xlabl, lab, titl, xs, ys, xss, yss, 2, temperature, ans)

################ PLOT 2:  Porosity and temperature C  vs Lambda MPa ###########
fig = 'Figure'+' '+repr(ShaleNumber2+7)
anss = 'yes'  # Do you want t deg C  plotted?
labl, lab = '', 'Fig'+str(ShaleNumber2+7),
titl = ' Age interval is'+str('%4.0f' % AgeRange2[0]) + ' -' + str('%4.0f'
                                                                   % AgeRange2[1])+' Ma.'
xlabl = fig+' '+str(ShaleName2)+''
ans = 'yes'
xDim = ' Grain-to-grain stress Lambda, MPa '

Plotz2(np.array([0., .9, .1]), np.array([0., 400., 50.]), ylabl,
       xDim, lab, titl, xs, Lam, xss, Lamda, ShaleName2, temperature, ans)

##### PLOT 3: 'surface-corrected'-Porosity and temperature C  vs_Lambda MPa ###
fig = 'Figure'+' '+repr(ShaleNumber2+13)

labl, lab = ' ', 'Fig'+' '+repr(ShaleNumber2+13)

titl = 'T = a + b*depth;  a,b ='+str('%3.1f' % PaleoMudlineT2) + '\
    ' + str('%4.3f' % GeothermalGradientMax2) +\
    '.  m,n = '+str('%3.2f' % m) + ',  ' + str('%3.2f' % n)

ans = 'yes'  # Do you want t deg C  plotted?
ylabl = 'Surface-corrected'+'  porosity'
xlabl = fig+ShaleName2+'.'
Plotz2(np.array([0., .9, .1]), np.array([0., .400, .05]), ylabl,
       xDim, lab, titl, xsP, Lam, xssP, Lamda, ShaleName2, temperature, ans)


########## PLOT 4:  Age of sediment My since start of deposition vs Depth m ###
fig = 'Figure' + repr(ShaleNumber2+25)
labl, lab = '', 'Fig'+repr(ShaleNumber2+25)
titl = 'Shale age with matrix deposition rate'+str('%4.1f' % KmSpMy)+' km/My'
ans = 'age'
xlabel = 'Depth, km, ' + fig+ShaleName2+'.'
ylabl = str(ShaleName2)+' porosity'
xDim = ' Depth, m '
Plotz2(np.array([0., .9, .1]), np.array([0., 400., 50.]), ylabl,
       xDim, lab, titl, xs, ys, xss, yss, ShaleName2, Dtdeposit, ans)


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
fig = 'Figure '+' '+str(ShaleNumber2+19)
ans = 'no'  # Do you want t deg C  plotted?
lab = 'Fig'+str(ShaleNumber2+19)
titl = ' Arrhenius plot with slope E ='+repr(round(avE, 1)) + ' kJ/mol '
ylabl = 'ln( LHS of Equation 11 )  /Myr'
xlabl = fig+'-1/RT.  '+str(ShaleName2)+', ' + \
    str(AuthorDate[ShaleNumber2])+'    /kJ/mol'
Plotz(axisRT, axislnLS, ylabl,
      xlabl, lab, titl, lnLS11xGen, RTinvx, lnLS11xxGen, RTinvxx, ShaleName2)
##############################################################################

# p-116, W. McKinney "Python for data analysis" Also p-112-133
#https://stackoverflow.com/questions/31983341/using-scientific-notation-in-pandas
pd.set_option('display.float_format', '{:.1E}'.format)
pd.Series(data=[10.0])
print()
print(' ',    str(ShaleName2) + ' Summary :')
print('Known shale age limits, My = ', AgeRange2, ', ', AgeName)
print()
data = {'Parameters': ['Total Depth', 'Solids Thickness',
                       'Geothermal Gradient', 'Surface temperature', 'Surface porosity', 'Minimum porosity',
                       'Minimum Temperature', 'Solids deposition rate', 'm', 'n', 'E', 'A'],

        'Value': [str(round(yss[-1])), str(round(Matrix[-1])),
                  str(GeothermalGradientMax2), repr(
                      PaleoMudlineT2), repr(phi0), repr(phi00),
                  repr(Low_Temp), repr(KmSpMy), repr(round(m, 2)), repr(round(n, 2)), repr(round(avE, 1)), avA],

        'Units': ['m', 'm',
                  'deg C/m', 'deg C', 'fraction', 'fraction', 'deg C', 'km/My',
                  '', '', 'kJ/mole', '/s'],

        'Comments': ['', '', '', '', '', 'Arbitrary limit', 'Arbitrary limit',
                     'Keep > ' + repr(round(Rate, 2))+'km/My', 'Derived', 'Derived', 'Derived', 'Derived']}
frame = DataFrame(data)

#https://www.google.com/search?client=firefox-b-1-lm&q=print+DataFrame+in+Ipython
#export DataFrame to CSV file
nam = str(ShaleName2)
frame.to_csv(r'/home/ed/Desktop/Z_Code0/'+ nam +'Data.csv', index=False)
print(frame)
print( ' THE END  ')
pass




















































#Data Fiits:
#sigma       =  np.array( [ 0.]*len(yss) )  # Fitted porosities vs depth
#temperature =  np.array( [ 0.]*len(yss) )  # Temperature deg C vs depth
#temperature[0] = PaleoMudlineT2
#Lamda       =  np.array( [ 0.]*len(yss) )   # grain_to_grain force per unit contact area
#lnLHS11     =  np.array( [ 0.]*len(yss) )    #Ln(LHS) Eq 11
#xssP        =  np.array( [ 0.]*len(yss) )      #Porosity with 'porosity correction'
#RTinverse   =  np.array( [ 0.]*len(yss) )      # - 1/RT  1/kJ mol
#lnxssP      =  np.array( [ 0.]*len(yss) )      # ln(xssP)
#Matrix      = np.array( [ 0.]*len(yss) )      # sum (1-phi)*dz
#Dedeposit   = np.array( [ 0.]*len(yss) )      # For estimates of activation E 


#Data:
#sig         =  np.array( [ 0.]*len(ys) )    # Fitted porosities vs depth
#temp        =  np.array( [ 0.]*len(ys)  )   #  Temperature deg C vs depth
#temp[0]     =  PaleoMudlineT2
#Lam         =  np.array( [ 0.]*len(ys) )    # grain_to_grain force per unit contact area
#lnLS11      =  np.array( [ 0.]*len(ys) )    #Ln(LHS) Eq 11
#xsP         =  np.array( [ 0.]*len(ys) )
#RTinv       =  np.array( [ 0.]*len(ys) )
#lnxsP       =  np.array( [ 0.]*len(ys) )
#Matrx       = np.array( [ 0.]*len(ys) )      # sum (1-phi)*dz 
#Dtdepo      = np.array( [ 0.]*len(ys) )      # For estimates of activation E 

#for i in range( 1,len( xss ), 1 ) :
#https://www.convertunits.com/from/bar/to/grams+per+(square+meter)   
    #conversion = [gf/cm^2*(10^4cm^2/m^2)]*[9.80665bar/(gf/m^2)] # multiplier = 9.80665*10^4 
#    sigma[i] = sigma[i-1] + 1.67*(1. - xss[i] )*( yss[i] - yss[i-1] )*\
#            .00980665  # g/cc*m to MPa
#    temperature[i] = temperature[i-1] + GeothermalGradientMax2*( yss[i] - yss[i-1] )       #deg C 
#    Lamda[i]       = Lamda[i-1]       +sigma[i]/(1. - xss[i])                              #kilobar force
#    Matrix[i]      = Matrix[i-1]      +(1. - xss[i] )*( yss[i] - yss[i-1] )                #meters of solids    
    
#    for k in range( len(ys) )  :    #Interpolate for plotting
#        if  (yss[i-1] <= ys[k]) and (yss[i] >= ys[k] )  :
#            sig[k] = sigma[i]
#            temp[k]= temperature[i]
#            Lam[k] = Lamda[i]   
#            Matrx[k]  = Matrix[i]   #meters of solids
            
##Quantities for getting meters of matrix per million years == Rate for this well:            
#TD           =  yss[ len(xss)-1 ]
#Matrix_to_TD = Matrix[ len(xss)-1 ]/ TD
#Rate         =  Matrix[ len(xss)-1 ]/ ( AgeRange2[1] - AgeRange2[0] )                 
#dz_dt        = ys[-1]*2./(AgeRange2[0]+AgeRange2[1]) # For estimates of activation E
#Average_Age  = ( AgeRange2[0]+AgeRange2[1])/2.      
#print 'dz_dt, Average_Age, AgeRange2 = ', 177,dz_dt,  Average_Age, AgeRange2[0], AgeRange2[1]
#print()
      
############# Firstly__Correct_Porosity_for_Surface_Areas_##################### AverageAge

#PhiAv = np.mean(xss)      #Normalization constant
#PhiNorm = pow( PhiAv,-1.33*m)* pow( (1.-PhiAv) ,-1.33*n )  #Normalization constant
#print()
#print('m, n, PhiAv, PhiNorm = ', m, n,  '% 5.4f % 5.4f' % ( PhiAv, PhiNorm ))
#print()
#for i in range( len(xss) )   :
 #   xssP[i]        = xss[i]* pow( xss[i],-1.33*m)* pow( (1.-xss[i]) ,-1.33*n )/PhiNorm 
    #lnxssP[i]      = np.log( xssP[i] )      
    #dPhi_dz[i]     = popt[0]+2.*yss[i]*popt[1]            # For computing E 

#for i in range( len(xs) )   : 
#    xsP[i]        = xs[i]* pow( xs[i],-1.33*m)* pow( (1.-xs[i]) ,-1.33*n )/PhiNorm
    #lnxsP[i]      = np.log( xsP[i] )
    #dPh_dz[i]     = popt[0]+2.*ys[i]*popt[1]            # For computing E 
    
#print()
#print('max(xssP), min( xssP) = \
#', '% 6.3e %6.3g ' % (np.max(xssP), np.min( xssP)   ))    
#print('max(xsP), min( xsP) = ', '% 6.3e %6.3g ' % ( np.max(xsP), np.min( xsP) ))              
#print()

########## Secondly,_Correct_ for Porosity and Temperature #################### New

#Tbar =  np.mean(temperature)+273.2
#for i in range( len(xss) )   :
#    RTinverse[i]  = -1./( 273.2+temperature[i])/(8.314*.001 )  #kJ/mol   
#avRTinverse = np.mean(RTinverse) 

#for i in range( len(xss) )   :    
#    lnLHS11[i]    = np.log( pow(xss[i],-4.*m/3.)*pow( (1.-xss[i]), -4.*n/3.)*np.log( (1.-xss[i])/(1.- .81))/Average_Age )     

#Tbr = np.mean( temp ) +273.2    
#for i in range( len(xs) )   :     
#    RTinv[i]      = -1./( 273.2 + temp[i])/(8.314*.001)  #kJ/mol
    
#avRTinv = np.mean(RTinv)
#for i in range( len(xs) )   :      
#    lnLS11[i]     = np.log( pow(xs[i],-4.*m/3.)*pow( (1.-xs[i]), -4.*n/3.)*np.log( (1.-xs[i])/(1.- .81))/Average_Age )  

#print(' max(RTinv), min(RTinv) = ', ' %6.3g %6.3g ' %  ( np.max(RTinv), np.min(RTinv) ))  
      

 ############## PLOT - 1 : porosity vs depth ################################## New
#phi0 = .7
#fig =  'Figure '+repr(ShaleNumber2+1)+' '

#print ' Average_Age = ',  int(Average_Age+.5)  #53.4
#ans = 'yes'  #Do you want t deg C  plotted?
#lab = 'Fig'+repr(ShaleNumber2+1)
#titl  = titl1
#ylabl = str(ShaleName2)+' porosity'
#xlabl =fig+str(ShaleName2)+' well porosity data trend (Fowler 1985) '
#xDim = ' Depth, m '

#Plotz2( np.array([0., .9, .1]), np.array([0., 400., 50.]), ylabl,\
#      xDim,lab,titl,xs, ys, xss, yss, ShaleName2,temperature, ans  )    
    
################ PLOT 2:  Porosity vs_Lambda_################################### New
#fig = 'Figure '+str(ShaleNumber2+7)+' '
#anss = 'yes'  #Do you want t deg C  plotted?
#labl, lab = '','Fig'+str(ShaleNumber2+7)
#titl = ' Average of age interval is'+str('%4.0f' % Average_Age)+' Ma'
#ylabl = 'Porosity '
#xlabl = fig+str(ShaleName2)+''
#ans = 'yes'
#xDim = ' Grain-to-grain stress Lambda, MPa '

#Plotz2( np.array([0., .9, .1]), np.array([0., 400., 50.]), ylabl,\
#      xDim,lab,titl,xs, Lam, xss, Lamda, ShaleName2,temperature,ans  )    
    
################ PLOT 3:  Porosity vs_Lambda, 'surface-corrected' ############ New

#fig = 'Figure '+repr(ShaleNumber2+13)+'     '
#print 'len(xss), len(yss), len(xs), len(ys) = ', cf.f_lineno ,    len(xss), len(yss), len(xs), len(ys) 
#print 'len(sigma), len(sig) = ', cf.f_lineno ,  len(sigma), len(sig)
#print ' len(temperature), len(Lamda) = ',  len(temperature), len(Lamda)  
#print ' len(Lam) len(Lamda) = ', cf.f_lineno , len(Lam), len(Lamda)
#labl, lab = '', 'Fig'+repr(ShaleNumber2+13)

#titl =  'T deg C = a + b*depth;  a,b ='+str('%4.0f' % PaleoMudlineT2)+ ',  ' +str('%4.3f' % GeothermalGradientMax2) 
#titl =  'T = a + b*depth;  a,b ='+str('%3.1f' % PaleoMudlineT2)+ ',  ' +str('%4.3f' % GeothermalGradientMax2)+\
#    '.  m,n = '+str('%3.2f' % m)+ ',  ' +str('%3.2f' % n)

#ans = 'yes' #Do you want t deg C  plotted?
#ylabl = 'Surface-corrected'+'  porosity'
#xlabl = fig+ShaleName2+'.'

#Plotz2( np.array([0., .9, .1]), np.array([0., 400, .05]), ylabl,\
#      xDim,lab,titl,xsP, Lam, xssP, Lamda, ShaleName2,temperature,ans  )


################ PLOT 4:  Porosity vs_Lambda, T and surface-corrected ######### New

#fig = 'Figure '+repr(ShaleNumber2+19)+'     '
#ans = 'yes' #Do you want t deg C  plotted?
#titl =   'Exponents m and n are '+repr(m)+','+repr(n)
#labl, lab =  '', 'Fig'+repr(ShaleNumber2+19)
#ylabl, xlabl = ''Corrected''+' porosity', fig+ShaleName2+'.'    

#Plotz2( popt, np.array([0., .9, .1]), np.array([0., 400., 50.]), ylabl,\
#      xlabl,lab,titl,xsTP, Lam, xssP, Lamda, ShaleName2,temperature,ans  ) 

############### Plot 5: Transformed  Porosity Histogram ####################### New

#phi_histigram( xssP, ShaleName2 )
#______________________________________________________________________________
#phi00 = 0.5
#name, entropyx, entropxx, eta_ratio = Entropyx( ShaleName2,xss, xssP  )     
#______________________________________________________________________________ New
#____________________________________________________ New


 ############## PLOT - 0 : Arrhenius plot ##################################### New
#fig =  'Figure '+' '+str(ShaleNumber2+19)

#print(' Average_Age = ', int(Average_Age+.5))
#print()
#ans = 'no' #Do you want t deg C  plotted?
#lab = 'Fig'+str(ShaleNumber2+19)
#titl  =  ' Arrhenius plot with slope E kj/mol '
#ylabl =  'ln( LHS of Equation 11)  /Myr' 
#xlabl =fig+'   -1/RT.  '+str(ShaleName2)+', '+str(AuthorDate[ShaleNumber2])+''
#xDim = ' -1/RT,   1/(kJ/mol) '
#### Estimate E from the SLOPE of the data/trend 'corrected' points alone: #### New
#### Estimate E from the SLOPE of the data/trend 'corrected' points alone: #### New


#Eslopx, lnTAUx, Aslopx, axisRT, axislnLS, lnLS11xGen, RTinvx = \
#EAandPlotCoords( temp,       Low_Temp,     RTinv,  xs, m, n, phi00,Dtdepo )  

#print("This is ", filename, ", code line = ", cf.f_lineno) 
#print()
#print(' Eslopx,Aslopx = ', ' %5.2e, %5.3e ' %  ( Eslopx,Aslopx ))
#print()    
#print(' len(axisRT), len(axislnLS), len(lnLS11xGen), len(RTinvx) = ', \
#    '%5.2e, %5.3e, %5.3e, %5.3e ' % \
#    ( len(axisRT), len(axislnLS), len(lnLS11xGen), len(RTinvx) ))    
#print()  

#### Estimate E from the SLOPE of the fitted curve 'corrected' points alone: ## New    
#### Estimate E from the SLOPE of the fitted curve 'corrected' points alone: ## New 
            
#Eslopexx, lnTAUxx, Aslopexx, axisRTx, axislnLHSx, lnLS11xxGen, RTinvxx = \
#EAandPlotCoords( temperature, Low_Temp, RTinverse, xss ,m, n,phi00, Dtdeposit )

#print("This is ", filename, ", code line = ", cf.f_lineno) 
#print()
#print(' Eslopexx,Aslopexx = ', ' %5.2e, %5.3e ' %  ( Eslopexx,Aslopexx ))
#print()    
#print(' len(axisRTx), len(axislnLHSx), len(lnLS11xxGen), len(RTinvxx) = ', \
#    '%5.2e, %5.3e, %5.3e, %5.3e ' % \
#    ( len(axisRT), len(axislnLS), len(lnLS11xGen), len(RTinvx) ))    
#print()  

#ratex =  Eslopx*Aslopx 
#ratexx = Eslopexx*Aslopexx
#avrate = (ratex+ratexx)/2.
#avlnTAU    = (lnTAUx+lnTAUxx)/2.

#avE, avA = (Eslopx+ Eslopexx)/2., (Aslopx+ Aslopexx)/2.

#avAsec     = avA/(1000000.*31556952.)
#Aslopxsec  = Aslopx/(1000000.*31556952.)
#Aslopxxsec  = Aslopexx/(1000000.*31556952.)

#print("This is ", filename, ", code line = ", cf.f_lineno) 
#print()
#print('avE,avAsec  = ',cf.f_lineno, ' %5.3e, %5.3e' % ( avE, avAsec ))
#print()
#print('Eslopx, Eslopexx = ',cf.f_lineno, ' %5.3e, %5.3e' % ( Eslopx, Eslopexx ))
#print()
 
#print(' Aslopxsec, Aslopxxsec = ',cf.f_lineno, '  %5.3e, %5.3e'  % ( Aslopxsec, Aslopxxsec )) 
#print()

#pint('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#print('________________________________________________________________________')
#print("This is ", filename, ", code line = ", cf.f_lineno) 
#print(" See lines 42-46. ")
#print()
#print(' Thickness of solid matrix, total depth, and ratio==c  m/m = ', Matrix[ len(xss) -1], TD, Matrix_to_TD)    
#print(' Thickness of solid matrix/ Geologic age span m/My == Av Rate for this well = ', Rate)  
#print(' Method # = code that evaluates dln(1-phi)/dt  =   ', Method)   
#print()
#print('Input-  Low_Temp,  m, n, phi0, phi00, GeothermalGradientMax2,AgeRange2 = ',\
#               Low_Temp,  m, n, phi0, phi00, GeothermalGradientMax2,AgeRange2)         
#print()
#print('Derived:  Eslopx, lnTAUx  Aslopxsec,   ratex     =   ', ' %6.3e, %6.3e,  %5.3e,  %5.3e ' \
#             % ( Eslopx, lnTAUx, Aslopxsec,   ratex ))    
#print('Derived:  Eslopexx, lnTAUxx, Aslopxxsec, ratexx =  ',  ' %6.3e, %6.3e,  %5.3e, %5.3e ' \
#             % ( Eslopexx, lnTAUxx, Aslopxxsec, ratexx))    
#print('Derived:  avE,  avlnTAU,    avAsec,      avrate  =  ',  ' %6.3e, %6.3e,  %5.3e, %5.3e ' \
#             % ( avE,  avlnTAU,    avAsec,      avrate  ))
#print()
#print('Derived:  name,                             entropyx, entropxx, eta_ratio  = ',\
#                 name, ' %6.3e  %6.3e  %6.3e'  % ( entropyx, entropxx, eta_ratio ))  
#print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx') #phi0

#print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
#print('________________________________________________________________________')
#print("This is ", filename, ", code line = ", cf.f_lineno) 
#print(" See lines 51-55. ")
#print 'Input:   m, n, phi0, GeothermalGradientMax2 = ', m,'  ',n, phi0, GeothermalGradientMax2
#print('Input-  Low_Temp,  m, n, phi0, phi00, GeothermalGradientMax2,AgeRange2 = ',\
 #              Low_Temp,  m, n, phi0, phi00, GeothermalGradientMax2,AgeRange2)         
#print()
#print(' Method # = code that evaluates dln(1-phi)/dt  =   ', Method) 
#print(' Thickness of solid matrix, total depth, and ratio==c  m/m = ', Matrix[ len(xss) -1], TD, Matrix_to_TD)    
#print(' Thickness of solid matrix/ Geologic age span m/My == Av Rate for this well = ', Rate)  
#print()
#print()
#print('Derived:  Eslopx, lnTAUx  Aslopxsec,   ratex     =   ', ' %6.3e, %6.3e,  %5.3e,  %5.3e ' \
#             % ( Eslopx, lnTAUx, Aslopxsec,   ratex ))    
#print('Derived:  Eslopexx, lnTAUxx, Aslopxxsec, ratexx =  ',  ' %6.3e, %6.3e,  %5.3e, %5.3e ' \
#             % ( Eslopexx, lnTAUxx, Aslopxxsec, ratexx))    
#print('Derived:  avE,  avlnTAU,    avAsec,      avrate  =  ',  ' %6.3e, %6.3e,  %5.3e, %5.3e ' \
#             % ( avE,  avlnTAU,    avAsec,      avrate  ))
#print()
#print('Derived:  name,                             entropyx, entropxx, eta_ratio  = ',\
#                 name, ' %6.3e  %6.3e  %6.3e'  % ( entropyx, entropxx, eta_ratio ))  
#
#print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx') #phi0

#Plotz( popt, axisRT, axislnLS, ylabl,\
#      xDim,lab,titl,lnLS11xGen , RTinvx,lnLS11xxGen , RTinvxx, ShaleName2  )      
#pass

































































                   
