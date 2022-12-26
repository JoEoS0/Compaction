# -*- coding: utf-8 -*-

"""
 F:\PyThings\VLpack\VLplots.py
 Get 'Best' stock gains as estimated & shelved by GetWeights.py or GetWgtS.py. Plot comparisons
 with the S&P500 or the equally-weighted Value Line average. Frac_Good
 
Created on Tue Nov 05 18:40:06 2013
@author: Edward
"""

import numpy as np
import matplotlib.pyplot as pl
import shelve
import time
import datetime
from Read_Excel import macro_data_input
from copy import deepcopy
from coherent import Herd_dir
from inspect import currentframe, getframeinfo
from pandas import  DataFrame


#%%%%%  Latest week for which results have been computed in VLstart.py. %%%%%%% 
LatestWeek =  macro_data_input( 2, 1 ) #The number of the latest VL data set available and the current file name.
frameinfo = getframeinfo(currentframe())            
print frameinfo.filename, frameinfo.lineno
print  ' Computational results were stored through week = ', LatestWeek  #The maximum 'Final' 

    
#%%%%%%%%%%%%%%%%%%% Date and Time Stamp for plots %%%%%%%%%%%%%%%%%%%%%%%%%%%% 
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

# %%%%%%%%%Parameters used in the most recent run of VLstart.py %%%%%%%%%%%%%%%

parameters = shelve.open('F:\\Pythings\\params.obj') #Fehily 323.
#.....................Computational options....................................
Re_Calc                   = parameters['Re_Calc']      # 1=recalculate rule weights. 
size                      = parameters['size']         #Number of weeks near top & bottom to average (GetWgtS.py).
Back                      = parameters['Back' ]          #Time increments,(Final-Initial) ; (VLstart.py).
Pct_type                  = parameters[ 'Pct_type' ]   #Report gains as 1 == 100.*ln(p1/p0) or 0 == 100.(p1/p0-1).
#.....................Run start and end options................................
Start                     = parameters['Start' ]       #First week number in calculations initiated by VLstrt.py.
TheEnd                    = parameters['TheEnd']       #Final week number in calculations initiated by VLstrt.py.
#NoData                    = parameters['NoData']       #Missing data weeks.
#.....................Print-out options........................................
Ahead                     = parameters['Ahead' ]       #In historic studies, 'weeks ahead' beyond any 'current' weekly estimate.VLresult1
LastFile                  =parameters['LastFile']      #'Final' to compute actual gains (GetWeights.py).
parameters.close()

Backs =  str(Back)  #Weeks before (back from) 'Center'.
NoDat  =  range(350,370 )
NoData =  NoDat                                     #Missing data weeks
NoData.extend( [ 97,98, 109,110, 183,184, 199,200 ] )     #Ocassional missing data early on
NoData.sort()
# List all possible solution weeks not including 'NoData' weeks, weeks 0-6, and weeks 371-5. 
# Later, leave out the final 'Ahead' weeks when computing median, mean and standard deviations of gains: 
good_soln_weeks = np.r_[ np.zeros( 6 ), np.ones( LastFile + 1 - 6 ) ]     #This puts week 1 into position one and week 'LastFile'
#into position 'LastFile+1'.Now throw out weeks without solutions for this Back[-2] and Ahead : 
for i in range( len(NoData) )  :
    good_soln_weeks[ NoData[i]                     ] = 0
    good_soln_weeks[ NoData[i] - Ahead             ] = 0
    good_soln_weeks[ NoData[i] + Back[-2]          ] = 0

good_soln_weeks =  ( good_soln_weeks ).tolist()    #This puts week-one data into list position '1'.
#print'65VLplot: LastFile, len(good_soln_weeks), good_soln_weeks[:7] =',LastFile, len(good_soln_weeks), good_soln_weeks[:7] 
#print '66VLplot: len(good_soln_weeks), sum(good_soln_weeks), LastFile, difference = ', len(good_soln_weeks), sum(good_soln_weeks), LastFile, LastFile-np.sum(good_soln_weeks)
#__________________________________________________________________________________________________
print 'VLplot68: Count & locate the weekly rule-weight solutions which were missing(0), convergent(1), or extrapolated(2) :'
fwn               = 'F:\\Pythings\\GainWgtsn.obj'   #No-div GainWgtsn
WgtSaven          = shelve.open( fwn, 'r' )         #Fehily 323.
fwd               = 'F:\\Pythings\\GainWgtsd.obj'   #Div GainWgtsd
WgtSaved          = shelve.open( fwd, 'r' )         #Fehily 323.

#wgt_statusN,TwosN, countN0, countN1, countN2 = [],[],0,0,0
#wgt_statusD,TwosD, countD0, countD1, countD2 = [],[],0,0,0
#for i in range( 1, LastFile+1 )  :
#    realN         = 'RN'+ str( i ) + Backs
#    if WgtSaven.has_key( realN )  :
#        GotWgtN   = WgtSaven[ realN ]          # = 0,1,2 if weights were not calculated, were new, were extrapolated for this week. 
#        wgt_statusN.append(  [i,int(GotWgtN) ]  )
#        countN0   += (int(GotWgtN) == 0)
#        countN1   += (int(GotWgtN) == 1)
#        countN2   += (int(GotWgtN) == 2)
#        if int(GotWgtN) == 2  :
#            TwosN.append( i )                   #At these weeks, extrapolated rule weights were used.
#            
#    realD         = 'RD'+ str( i ) + Backs
#    if WgtSaved.has_key( realD )  :
#        GotWgtD   = WgtSaved[ realD ]          # = 0,1,2 if weights were not calculated, were new, were extrapolated for this week.  
#        wgt_statusD.append( [i,int(GotWgtD)] )
#        countD0   += (int(GotWgtD) == 0)
#        countD1   += (int(GotWgtD) == 1)
#        countD2   += (int(GotWgtD) == 2)
#        if int(GotWgtD) == 2  :
#            TwosD.append( i )                   #At these weeks, extrapolated rule weights were used.            
WgtSaven.close()
WgtSaved.close()
#print '93VLplot: countN0, countN1, countN2, TwosN =', countN0, countN1, countN2, TwosN
#print '94VLplot: countD0, countD1, countD2, TwosD =', countD0, countD1, countD2, TwosD
#print '95VLplot: len( wgt_statusN ), wgt_statusN[95:111] = ', len( wgt_statusN ), wgt_statusN[91:111]
#print '96VLplot: len( wgt_statusD ), wgt_statusD[95:111] = ', len( wgt_statusD ), wgt_statusD[91:111]
#ba2N, ba2D        = set( TwosN ), set( TwosD )
#Extrapolate2       = ba2N & ba2D
#NoDat_set         = set( NoData )
#NoData_Extrapolate2 = NoDat_set & ba2N | NoDat_set & ba2D
#print '105plot sets: Extrapolate2, NoData_Extrapolate2 = ', Extrapolate2, NoData_Extrapolate2
#___________________________________________________________________________________________________
#------------------------------------------------------------------------------
LW, LWp = LatestWeek, LatestWeek+1

pl.close() 

SaP1     = [0. ]*LatestWeek  #S&P500 index. 
Week, VLfileName, SaP500, VIX, Tbill3m, SaP1,LastWeek = macro_data_input( LatestWeek,0 )

#------------------------------------------------------------------------------    
frameinfo = getframeinfo(currentframe())            
print frameinfo.filename, frameinfo.lineno
print ' size = ', size
print ' For latest VLstart.py run: Calculation Start & End; LatestWeek = ', Start,'  ', TheEnd,';  ', LatestWeek

#%%%%%%%%% INPUT NEEDED TO SPECIFY PLOT: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print '                                                                       '
print 'REQUESTED INPUT FROM THE "VLplot.py" USER --->   '
frameinfo = getframeinfo(currentframe())            
print frameinfo.filename, frameinfo.lineno
TS1            = 5 + input(' Enter 0 (no-div) or 1 (div) to plot gains-vs-time.|||Else to plot gains vs net_positive_weights, enter 2(no-div) ,3(div) : ' )
NewWeights = 1  # input(" To include all weeks enter 0, only weeks with fully-data-supported weights, 1 : " )
Timing_Use    = 0   #Default value for NOT using market-timing to get out of the market.
#if TS1 <= 6  :
#    Timing_Use = input( ' Use (=1) or do NOT use (=0) market-timing in gains-vs-time plots : ' )
    
Usual           = input( " The usual type of plot = 1 ? Else select plot variables = 0 ? " )
if TS1 > 6  :
    frameinfo  = getframeinfo(currentframe())            
    print frameinfo.filename, frameinfo.lineno    
    Gain_Group = input("  Enter  1,2, or 3 to plot 'All', 'Best' or 'Worst' median gains vs Psychiatric Score: " ) 
TS             = TS1*(TS1<=6) + 5*(TS1==7) + 6*(TS1==8)
    
if Usual       ==  1 :     
    num1       = 6
    numf       = min(LatestWeek, 3000 ) #The final week of data for plot inclusion.
    PlotType, CumPlot = 1,1  # =1,1 for median gains and cumulative gains.
else           :     
    frameinfo = getframeinfo(currentframe())            
    print frameinfo.filename, frameinfo.lineno    
    num1       = input(" Enter the first data week to average and plot : " )    
    numf       = input(" Enter the last data week number to plot: " )
    numf       = min(LatestWeek, numf )   #The last week of data for inclusion.       
    PlotType   = input(' Enter "0"=to plot average %gains,"1"=to plot median %gains : ')    
    CumPlot    = input(" Enter 0 to plot WEEKLY %gains, 1 to plot CUMULATIVE %gains: " )
    NewWeights = 1  #input(" To include all weeks enter 0, only weeks with CVX-converged weights, 1 : " )
    
#%%%%% Unravel the shelved averages needed in the plots%%%%%%%%%%%%%%%%%%%%%%%%
#Below, G & S == greatest and smallest gain estimates.
numfp                       = numf + 1 
GnG,  GnS, GnVL             = [0.]*numfp, [0.]*numfp, [0.]*numfp              #Average gains%         
FracAdvBest, FracAdvWorst   = [0.]*numfp, [0.]*numfp                          #Frac advances in 'size' stocks. 
GnGM, GnSM, GnVLM,  FracAdv = [0.]*numfp, [0.]*numfp, [0.]*numfp, [0.]*numfp  #Medians; fraction of all VL stocks advancing.
GnGsd,GnSsd,GnVLsd, Tic     = [0.]*numfp, [0.]*numfp, [0.]*numfp, [0.]*numfp  #Standard Deviations of average gains%
OKG, GROCERY_Idx            = [0.]*numfp, [0.]*numfp

#Retrieve shelved average gain data, 'vector', produced near the end of GetWgtS.py to plot here:    
fin            = 'F:\\Pythings\\VLresult1.obj' #<<The '1' here is the replacable default.
result         = shelve.open( fin )   # == 'vector' in GetWgts. Fehily 323.
Gnum           = 0  
for  i in range(num1,numfp ) :  #Get weekly averages.
    kfin   = str( str(i).zfill(5) ) + Backs
    if result.has_key(kfin)  :      # "Python" C. Fehily 323.
        vector = result[ kfin ]
        if len( vector )>=7 and isinstance(vector[5], list) and isinstance( vector[6], list)  :  

        #vector = [FileNum, VLfileName, SandP500, VIX, 'blank' ]
            #       0        1              2       3     4
        #________________________________________________________________________ 
            #vec5,6=[ va1G, gnsG, va1S, gnsS, gnsAv, FracAdvBest, FracAdvWorst, \
            #           0     1     2      3     4        5            6        
            #         PriceG, PriceS, gnGM, gnSM, gnM, FracAdv, \
            #             7     8      9     10   11    12                       
            #         gnsGsd, gnsSsd, gnsAvsd, GotWgtD GROCERY_Idx ] 
            #           13      14       15      16       17     
            #________________________________________________________________________
            #Leave out weeks where the weight estimates didn't converge, if so specified in 'input':
            
            try  :
                OKG[i] = NewWeights*( good_soln_weeks[i] ) + (1-NewWeights)                
            except  :
                frameinfo = getframeinfo(currentframe())            
                print frameinfo.filename, frameinfo.lineno                
                print( 'TROUBLE! : i, NewWeights, num1, numf = ', i, NewWeights, num1, numf )

            GG  = OKG[i]                                       #Includes all the existing/specified weekly data for both plotting and statistics.            

            GnG[i]            = vector[TS][1]*GG               #Best weekly stock <gains %>
            GnS[i]            = vector[TS][3]*GG               #Worst weekly stock <gains %>
            GnVL[i]           = vector[TS][4]*GG               #All(no-div or div) Value Line stock <gains%>
            FracAdvBest[i]    = vector[TS][5]*GG               #Fraction of advances for best  stock group.
            FracAdvWorst[i]   = vector[TS][6]*GG               #Fraction of advances for worst stock group.
            GnGM[i]           = vector[TS][9]*GG               #Best weekly stock % gains median.             
            GnSM[i]           = vector[TS][10]*GG              #Worst weekly stock % gains median.        
            GnVLM[i]          = vector[TS][11]*GG              #All(no-div or div) Value Line stock % gains median.
            FracAdv[i]        = vector[TS][12]*GG              #Fraction of all(no-div or div) Value Line stocks advancing. 
            GnGsd[i]          = vector[TS][13]*GG              #Standard Deviations of Best  stock gains %.
            GnSsd[i]          = vector[TS][14]*GG              #Standard Deviations of Worst stock gains %.       
            GnVLsd[i]         = vector[TS][15]*GG              #Standard Deviations of All(no-div or div) Value Line stock gains %.
            GROCERY_Idx[i]    = vector[TS][17]*GG              #Grocery Index == sum of market caps, weekly.
            Gnum += GG*( num1 <= i <= ( LastFile-Ahead )  )    #Excludes missing beginning weeks, and ending weeks where gain periods are short.
result.close             

# Get fraction 'rr' of available weeks for statistics for the rule weights and gains computed:
denominator= LastFile - (num1-1) - Ahead    #Denominator, span of weeks possibly containing weight & computed gains solutions
if TS1    <= 6 or TS <= 6 :
#'NoData' weeks limit the number of weeks with proper solutions that are included.   
    rr     = float( Gnum )/float(denominator)      #Fraction of weeks with proper solutions out of the available time span.
    print 'denominator, Gnum = ', denominator, Gnum

frameinfo = getframeinfo(currentframe())            
print frameinfo.filename, frameinfo.lineno
print '  Fraction of total accessable weeks with gains data included, no-dividend or dividend = ', '%5.3f' %rr 
 
SaP = [ SaP1[u] - SaP1[ num1 ] for u in range( len(SaP1) ) ]  #For comparitative plots only.
GROCERY_Id  = [0.]*len(GnG)
#GROCERY_Id = [ GROCERY_Idx[u]*1000./GROCERY_Idx[ num1 ] for u in range( len(GROCERY_Idx) ) ]      
#__________________________________Series to plot______________________________
# Cumulative 'x' gains for making plots, -not for statistical calculations.
if CumPlot == 1  :
    GnGx,  GnSx,  GnVLx                = [0.]*numfp, [0.]*numfp, [0.]*numfp
    GnGMx, GnSMx, GnVLMx, FracAdvCum   = [0.]*numfp, [0.]*numfp, [0.]*numfp, [0.]*numfp
    for  k in range( num1, numfp ) :  #Get cumulative sums.
                   
        GnGx[k]            =   sum(GnG[:k])*OKG[k]       #Best  stock <gains %> prediction results
        GnSx[k]            =   sum(GnS[:k])*OKG[k]       #Worst stock <gains %> prediction results
        GnVLx[k]           =   sum(GnVL[:k])*OKG[k]      #All no-div or div VL stock <gains %>
        GnGMx[k]           =   sum(GnGM[:k])*OKG[k]      #Best  stock %gains median prediction results             
        GnSMx[k]           =   sum(GnSM[:k])*OKG[k]      #Worst stock %gains median prediction results
        GnVLMx[k]          =   sum(GnVLM[:k])*OKG[k]     #All no-div or div  median prediction results         
        FracAdvCum[k]      =   sum(FracAdv[:k])*OKG[k]   #All no-div or div  fraction of advancing stocks    

#____________________________Statistics________________________________________        
#   
#Gnum=Number of weeks with proper solutions computed between num1 and min(numf, LastFile-Ahead).
def stat1(y,decimals, Gnum ):
    yav = sum(y)/Gnum
    ysqdif = ( (np.array(y))*( np.array(y)) ).sum()/float(Gnum) -yav*yav
    ysd    = round( abs(ysqdif)**0.5, decimals ) 
    yav    = round( yav, decimals         )
    return yav, ysd
u = len(GnVL)-1 - Ahead    
GnGav, GnGsd                  = stat1( GnG[:u] , 1,Gnum )        # Time Average & Standard Deviation of gain% averages 
GnSav, GnSsd                  = stat1( GnS[:u] , 1,Gnum )        # Time Average & Standard Deviation of gain% averages
GnVLav, GnVLsd                = stat1( GnVL[:u], 1,Gnum )        # Time Average & Standard Deviation of gain% averages 
FracAdvBestav, FracAdvBestsd  = stat1( FracAdvBest[:u], 2, Gnum) # Time Average & Standard Deviation of Frac advances
FracAdvWorstav, FracAdvWorstsd= stat1( FracAdvWorst[:u], 2,Gnum )# Time Average & Standard Deviation of Frac advances
GnGMav, GnGMsd                = stat1( GnGM[:u], 1,Gnum )        # Time Average & Standard Deviation of gain medians
GnSMav, GnSMsd                = stat1( GnSM[:u], 1,Gnum )        # Time Average & Standard Deviation of gain medians
GnVLMav, GnVLMsd              = stat1( GnVLM[:u], 1,Gnum )       # Time Average & Standard Deviation of gain medians
FracAdvav, FracAdvsd          = stat1( FracAdv[:u], 2,Gnum )     # Time Average & Standard Deviation of gain medians

#_________________________________Statistics to View___________________________
frameinfo = getframeinfo(currentframe())            
print frameinfo.filename, frameinfo.lineno
print ' <<Best  gain % estimate>> /stock & <<fraction advancing>> =           ',GnGav, '  ',FracAdvBestav
print ' <<Worst gain % estimate>> /stock & <<fraction advancing>> =           ',GnSav, '  ',FracAdvWorstav 
good_weeks = sum( good_soln_weeks )
print ' LatestWeek,starting and stopping weeks for plot, good_weeks, data_weeks  = ', LatestWeek, num1, numf, good_weeks, Gnum
print ' Fraction of all (no-dividend or dividend) stocks advancing: recent, average, standard deviation, and cumulative = ',\
    round(FracAdv[-Ahead-1],3),'  ', FracAdvav,'  ', FracAdvsd,'  ', round(FracAdvCum[-1],0)
#_____________________________Plots____________________________________________

iis = [ i for i in range( num1, numf) ]  # x-axis week numbers.
#__________________________Plot Heading________________________________________
Bak = -Back[-2 ]
params = 'Parameters: Holding time(weeks), #Stocks, Weeks Back, Initial & Final weeks, Last data set, Fraction of weeks used, Plot date = \n '+str(Ahead)+', '+\
str(size)+', '+str( Bak )+', '+str(num1)+' & '+str(TheEnd)+', '+str(VLfileName)+', '+str( '%5.3f' %rr )+', '+str(st)    
    
TSDict       = { 5:'NON-DIVIDEND PAYING',     6:'DIVIDEND-PAYING' }
TSDict2      = { 5:'ALL NON-DIVIDEND PAYING', 6:'ALL DIVIDEND-PAYING' }
TSDict1      = { 5:'NO-DIV ',                 6:'DIV ' }
PlotTypeDict = { 1: 'MEDIAN',                 0: 'AVERAGE' }
CumDict      = { 0:" WEEKLY Gains ",  1:" CUMULATIVE Gains ", 2:"Gains after ''Ahead'' Weeks" }
Pct_typeDict = { 1:' ln(P/P0)*100 ',          0:' (P/P0-1)*100 '}
RuleWgtDict  = { 1:' Best ', 2:' Best ' ,     3:' Worst ' }
Weight_Dict  = { 1:' computed only',           0:' computed and extrapolated' }
Timing_Dict  = { 1:'',                    0:'not ' }
#TITLE:
Titl =   'Year \n'+ PlotTypeDict[PlotType]+ Pct_typeDict[Pct_type]+ CumDict[CumPlot]+' Are Plotted For '\
+TSDict[TS]+' Stocks. 8/25/1996+' 
   
print '                                                                        '
pl.close()
if TS1 <= 6  :
    #__________________________PLOT Average GAIN% vs TIME______________________ Parameters
#          http://jakevdp.github.io/mpl_tutorial/tutorial_pages/tut1.html
    if PlotType == 0 :    #Plot the gain% averages.
        ID9 = TSDict[TS]+' Value Line stocks gains%: median, average, std, avg/std, and fraction positive = '\
        +str(GnVLMav)+'  '+str( GnVLav )+'  '+str(GnVLsd)+'  '+str( round(GnVLav/GnVLsd,2) )+'  '+str(FracAdvav ) 
        pl.plot(iis, GnVLx[num1:numf],    'red'   ,  label = ID9 )
    
        ID  = 'Best '+TSDict1[TS]+'stocks gain%: median, average, std, avg/std, and fraction of these stocks advancing      = '\
          +str(GnGMav)+'  '+str( GnGav )+'  '+str( GnGsd )+'  '+str(round( GnGav/GnGsd,2) )+'  '+ str(FracAdvBestav )
        pl.plot(iis, GnGx[num1:numf] ,   'green'   ,  label = ID  )
        
        ID1 = 'Worst '+TSDict1[TS]+'stocks gain%: median, average, std, avg/std, and fraction of these stocks advancing    = '\
          +str(GnSMav)+'  '+str( GnSav  )+'  '+str( GnSsd )+'  '+str(round(GnSav/GnSsd,2) )+'  '+str(FracAdvWorstav )
        pl.plot(iis, GnSx[num1:numf],    'purple'   ,  label = ID1 )
        
        if CumPlot == 1 : 
            pl.ylim( min( min(GnSx),min(GnVLx) , -1000.), max(max(GnVLx),max(GnGx)+1000., 7000.) )
            
    #____________________________PLOT MEDIAN GAIN% vs TIME_____________________
    if PlotType == 1 : #Plot the median% gains.  
    
        IDM9 = TSDict[TS]+' Value Line stocks gains%: median, average, std, avg/std, and fraction positive = '\
        +str(GnVLMav)+'  '+str( GnVLav )+'  '+str(GnVLsd)+'  '+str( round(GnVLav/GnVLsd,2) )+'  '+str(FracAdvav )
        pl.plot(iis, GnVLMx[num1:numf],    'red'   ,  label = IDM9 )
        pl.plot( range(350, 371 ), [400]*21,'black' )
    
        IDM  = 'Best '+TSDict1[TS]+'stocks gain%: median, average, std, avg/std, and fraction of these stocks advancing      = '\
        +str(GnGMav)+'  '+str( GnGav )+'  '+str( GnGsd )+'  '+str( round(GnGav/GnGsd,2) )+'  '+ str( FracAdvBestav )
        pl.plot(iis, GnGMx[num1:numf] ,   'green'   ,  label = IDM  )
        
        IDMf = '(Showing cumulative gains for stocks with convergent solutions only)'
        
        
        IDM1 = 'Worst '+TSDict1[TS]+'stocks gain%: median, average, std, avg/std, and fraction of these stocks advancing    = '\
        +str(GnSMav)+'  '+str( GnSav  )+'  '+str( GnSsd )+'  '+str(round(GnSav/GnSsd,2) )+'  '+str( FracAdvWorstav )+' \n' \
        'Weeks with' +Weight_Dict[NewWeights]+ ' solutions are included.' + ' Market timing is ' +Timing_Dict[Timing_Use]+ 'included.'

        
        pl.plot(iis, GnSMx[num1:numf],    'purple'   ,  label = IDM1 )
    
        if CumPlot == 1 : 
            pl.ylim( min(min(GnSMx),min(GnVLMx) , -1000.), max(max(GnVLMx),max(GnGMx)+4000., 7000.) )
            
    #______________________________________________________________________________

        #__________________Print 'year' on x-axis instead of week number_______________
        #https://matplotlib.org/examples/ticks_and_spines/ticklabels_demo_rotation.html
        #https://matplotlib.org/examples/ticks_and_spines/tick_labels_from_values.html
        pl.xticks([-31,21,72,124,176,228,280,333,384,437,489,541,593,646,698,751,803,855,907,959,1011,1063],\
        [r'96',r'97',r'98',r'99',r'00',r'01',r'02',r'03',r'04',r'05',r'06',r'07',r'08',\
        r'09',r'10',r'11',r'12',r'13',r'14',r'15',r'16',r'17'])    
        # You can specify a rotation for the tick labels in degrees or with keywords.
        #pl.xticks(x, labels, rotation='vertical')
        # Pad margins so that markers don't get clipped by the axes
        pl.margins(0.2)
        # Tweak spacing to prevent clipping of tick-labels
        pl.subplots_adjust(bottom=0.15)
        pl.show()    
    #______________________________________________________________________________
#    if PlotType < 2 :
#        ID_GROC = ' GROCERY            '
#        pl.plot(iis, GROCERY_Id[num1:numf], 'brown', label = ID_GROC )        
# 
#        IDTSaP = ' S&P500            '
#        pl.plot(iis,     SaP[num1:numf], 'brown', label = IDTSaP )                    
        #&&&&&&&& Plot Annotations &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        
        pl.annotate('Dot-com peak', xy= (184,-20), xytext=(195,-800), \
        arrowprops=dict(facecolor = 'black', shrink = 0.05), )
        
        pl.annotate('(no-data weeks)', xy= (372,400), xytext=(403,600), \
        arrowprops=dict(facecolor = 'black', shrink = 0.05 ), )
        
        pl.annotate('Housing peak', xy= (515,-20), xytext=(460,-800), \
        arrowprops=dict(facecolor = 'black', shrink = 0.05), )
        
        pl.annotate('QE1',          xy= (638,-20), xytext=(600,-800), \
        arrowprops=dict(facecolor = 'black', shrink = 0.05), )  #Nov 2008
        
        pl.annotate('QE2',          xy= (741,-30), xytext=(738,-1000), \
        arrowprops=dict(facecolor = 'black', shrink = 0.05), )  #Nov 2010
        
        pl.annotate('QE3',          xy= (838,-30), xytext=(815,-1600), \
        arrowprops=dict(facecolor = 'black', shrink = 0.05), )  #Sept13,2012
        
        pl.annotate('(fewer weeks)', xy=(numf- Ahead,-60), xytext= (numf- Ahead - 115,-1600), \
        arrowprops=dict(facecolor = 'black', shrink = 0.05), )
        
        #_________________ Plot labels ________________________________________
        
        pl.title(params)
        pl.xlabel(Titl )
        pl.ylabel('Best & Worst Stocks:Green & Purple; '+TSDict2[TS]+' VL Stocks:Red')
        pl.axhline(y = 0)
        pl.axvline(x = LatestWeek - Ahead )
        pl.legend(loc = 'upper left')
         
        pl.show()
       
#______________________________________________________________________________
        
#______________________________________________________________________________
#______________PLOT MEDIAN GAINS vs 'PSYCHIATRIC' WEIGHT SCORE___________________

#______________________________________________________________________________ 
#______________________________________________________________________________        
        #http://jakevdp.github.io/mpl_tutorial/tutorial_pages/tut1.html           
    
if TS1 > 6 :

    pl.close()  # = Close a figure window. cla() = clear one axis and clf() = clear entire figure & axes & keep window open   
    num1          = 6
    TSDict        = { 5:'NON-DIVIDEND-PAYING', 6:'DIVIDEND-PAYING' }    
    RuleWgtDict   = { 1:' All ', 2:' Best ' , 3:' Worst ' }   # Here, Gain_Group = 1,2, or 3. 
                 
    OK11   = np.array( OKG )                       #OKG is initially mostly [1]*numf at this point. 
    Index   = np.where(  OK11 != 1 )               # Indices where there are zeros == no data. Marsland 379 
    #http://stackoverflow.com/questions/10996140/how-to-remove-specific-elements-in-a-numpy-array

    netN_p, netD_p = [0]*LW, [0]*LW
    #These  items are returned for all LW weeks:
    #print 'VLplots412: type(FracAdv), type(GROCERY_Id), type( Gain_Group ) = ', type(FracAdv), type(GROCERY_Id), type( Gain_Group )
    netN_p, netD_p = Herd_dir(  LastFile,Ahead, LastFile, 2, FracAdv, GROCERY_Id,Gain_Group  ) 

#Here we drop the significance of weekly position and go to relative position:   CVXopt
    GnVLMy = deepcopy( np.delete( np.array( GnVLM      ), Index ) )        
    GnGMy  = deepcopy( np.delete( np.array( GnGM       ), Index ) )
    GnSMy  = deepcopy( np.delete( np.array( GnSM       ), Index ) )
    xxN    = deepcopy( np.delete( np.array( netN_p     ), Index ) )
    xxD    = deepcopy( np.delete( np.array( netD_p     ), Index ) )
    #GROCERY= deepcopy( np.delete( np.array( GROCERY_Id ), Index ) )
    FracA  = deepcopy( np.delete( np.array( FracAdv    ), Index ) )     
    #No significant difference observed if data < week 370 is left out.
    
    avxN   = np.arange( -9, 10 )
    avxD   = np.arange( -10,11 )  

    #Dividend Stocks :
    
    if TS1 == 6 or TS1 == 8  :
        Markov0= np.zeros( (21,21) )                    #E Parzen, Ch 3 Modern Probability Theory, or W. Feller.
        Markov1= np.zeros( (21,21) )    
        Markov3= np.ones(  (21,21) )        
        for m in range( len( GnVLMy ) - 1 )  :
            ro =  xxD[m]   + 10
            cl =  xxD[m+1] + 10
            Markov0[ro][cl] += 1                       #Markov 'transition probabilities(sums, actually)'
            Markov1[ro][cl] += GnVLMy[ m+1 ]           #The corresponding summed gains 
        Markov2 = Markov1/np.fmax( Markov0, Markov3 )  # Average the gains. Don't divide by zero. Wes McKinney p 96
        Markov2 = np.rint( Markov2 )                   #Round averages to nearest integer  
        framM  = DataFrame( Markov0, columns = avxD, index = avxD   )  #Wes McKinney, Python for Data Analysis, Ch 5.
        frameinfo = getframeinfo(currentframe())            
        print frameinfo.filename, frameinfo.lineno       
        print '______________________________________________________________________'
        print '~Markov: Counts of Successive Dividend Stock States :                    '
        print  framM  
        print '______________________________________________________________________'
        frameinfo = getframeinfo(currentframe())            
        print frameinfo.filename, frameinfo.lineno        
        print '~Markov: Average of Gains in Each Dividend Stock State :                 '            
        framM1 = DataFrame( Markov2, columns = avxD, index = avxD   )
        print  framM1 

#Non-Dividend Stocks :    
    if TS1 == 5 or TS1 == 7  :
        Markov0= np.zeros((19,19))                    #E Parzen, Ch 3 Modern Probability Theory, or W. Feller.
        Markov1= np.zeros((19,19))    
        Markov3= np.ones( (19,19))        
        for m in range( len( GnVLMy ) - 1 )  :
            ro =  xxN[m]   + 9
            cl =  xxN[m+1] + 9
            Markov0[ro][cl] += 1                       #Markov 'transition probabilities(sums, actually)'
            Markov1[ro][cl] += GnVLMy[ m+1 ]           #The corresponding summed gains 
        Markov2 = Markov1/np.fmax( Markov0, Markov3 )  # Average the gains. Don't divide by zero. Wes McKinney p 96
        Markov2 = np.rint( Markov2 )                   #Round averages to nearest integer  
        framM  = DataFrame( Markov0, columns = avxN, index = avxN   )  #Wes McKinney, Python for Data Analysis, Ch 5.
        frameinfo = getframeinfo(currentframe())            
        print frameinfo.filename, frameinfo.lineno        
        print '______________________________________________________________________'
        print '~Markov: Counts of Successive Non-Dividend States :                    '
        print  framM  
        frameinfo = getframeinfo(currentframe())            
        print frameinfo.filename, frameinfo.lineno        
        print '______________________________________________________________________'
        print '~Markov: Average of Gains in Each Non-Dividend Stock State :                 '            
        framM1 = DataFrame( Markov2, columns = avxN, index = avxN   )
        print  framM1         

#///////////////////////////////////////////////////////////////////////////////        
    #print '402: len GnVLMy, GnGMy, GnSMy, xxN, xxD = ', len(GnVLMy), len(GnGMy), len(GnSMy), len(xxN), len(xxD)
    keep        = 'F:\\Pythings\\Mkt_Timer.obj'   # Fehily 323. Mkt_Timer
    SaveAvgs    = shelve.open( keep, "c" )
    ky          = 'Dir' + str( Ahead ).zfill(3) + str( size ).zfill(4)     
    if not SaveAvgs.has_key( ky )  :
        collect = [0]*6
    else  :
        collect = SaveAvgs[ ky ]
#_______________________________________________________________________________________________    
    if TS1 == 7  :
        
        IDM7 = RuleWgtDict[Gain_Group]+'NON-DIVIDEND stocks'
        
        xx = deepcopy(        xxN[ num1: ]   ) 
        if Gain_Group == 1  :
            yy = deepcopy( GnVLMy[ num1: ]   )  #All
        if Gain_Group == 2  :
            yy = deepcopy( GnGMy[  num1: ]   )  #Best
        if Gain_Group == 3  :
            yy = deepcopy( GnSMy[  num1: ]   )  #Worst
        #https://www.google.com/search?q=add+to+python+plot+with+time+delay&ie=utf-8&oe=utf-8
        #http://stackoverflow.com/questions/10944621/dynamically-updating-plot-in-matplotlib
        #https://stackoverflow.com/questions/8213522/when-to-use-cla-clf-or-close-for-clearing-a-plot-in-matplotlib            
        #https://www.packtpub.com/mapt/book/big_data_and_business_intelligence/9781849513265/3/ch03lvl1sec48/controlling-tick-spacing
        pl.clf()  # = Close a figure window. cla() = clear one axis and clf() = clear entire figure & axes & keep window open
                                                                
        Titl  =  ' Market Direction ( unrefined method ) '
        pl.title(params)
        pl.xlabel(Titl )
        pl.ylabel('Median percent gains: '+TSDict[TS]+ ' VL Stocks' )
        pl.legend(loc = 'upper left')         
        pl.axes = pl.gca()      #GCA = get current axes
        pl.axes.set_ylim([-25.,+20.])         
        pl.axes.set_xlim([-10,+10])
        pl.grid(True)
           
        pl.plot( xx, yy, '^b' , label = IDM7 )         
        pl.plot( xx[ -2: ], yy[ -2: ], '-r' )
           
        VN  = [0]*len(yy)   
        avN = [0]*19
        avxN   = np.arange( -9, 10 )
        #pl.plot( avxN, avN, '-k'  )        
        for k in range( -9, 10 )  :
            VN       = np.where(  np.array(xx) == k, 1, 0 )
            avN[k+9] = (  VN*np.array(yy) ).sum()/max(1., VN.sum() ) 
        pl.plot( avxN, avN, '-k' , label = IDM7  )            
        #https://www.google.com/search?q=Eliminate+NaN+in+python+list+comprehension&ie=utf-8&oe=utf-8    
        rounded_list = [int(round(n,0)) for n in avN  ]
        #print(RuleWgtDict[Gain_Group] + 'Non-dividend stocks <gains%> for states = -9 to +9 :', rounded_list)        
        pl.plot( avxN[-3:], avN[-3:], '-k'  )

        collect[ Gain_Group -1 ] = rounded_list
   
#______________________________________________________________________________________________    
    if TS1 == 8 :
        IDM8 =  RuleWgtDict[Gain_Group]+ 'DIVIDEND stocks'
        pl.clf()  # = Close a figure window. cla() = clear one axis and clf() = clear entire figure & axes & keep window open        
        xx = deepcopy(       xxD[ num1: ] )
        if Gain_Group == 1  :
            yy = deepcopy( GnVLMy[num1: ] )  #All
        if Gain_Group == 2  :
            yy = deepcopy( GnGMy[num1:  ] )  #Best
        if Gain_Group == 3  :
            yy = deepcopy( GnSMy[num1:  ] )  #Worst

        Titl  =  ' Market Direction ( unrefined method ) '
        pl.title(params)
        pl.xlabel(Titl )
        pl.ylabel('Median percent gains: '+TSDict[TS]+ ' VL Stocks' )
        pl.legend(loc = 'upper left')         
        pl.axes = pl.gca()      #GCA = get current axes
        pl.axes.set_ylim([-25.,+20.])         
        pl.axes.set_xlim([-10,+10])
        pl.grid(True)
           
        pl.plot( xx, yy, 'go' , label = IDM8 )
        pl.plot( xx[ -2: ], yy[ -2: ], '-r', label = IDM8 )                    

        VD  = [0]*len(yy)   
        avD = [0]*21
        for k in range( -10, 11)  :
            VD       = np.where(  np.array(xx) == k, 1, 0 )
            avD[k+10] = (  VD*np.array(yy) ).sum()/max(1., VD.sum() )               
        rounded_list = [int(round(n,0)) for n in avD ]
        pl.plot( avxD, avD, '-k'  )        
                
        print(RuleWgtDict[Gain_Group] + 'Dividend stocks <gains%> for states = -10 to +10 :', rounded_list)
#https://matplotlib.org/devdocs/gallery/api/date_index_formatter.html#sphx-glr-gallery-api-date-index-formatter-py        
#https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.subplots.html
        
        collect[ Gain_Group +2 ] = rounded_list
        
    SaveAvgs[ ky ] = collect
#____________________________________________________________________________________________________________________    
    # Put out 'collect' as a table, using Pandas -Wes McKinney, "Python for Data Structures" 112-132
    
    if collect.count(0) <= 3  :
    #No-dividend stocks table :
        c1    = deepcopy( np.array( collect[ :3 ] ) )
        rangN = np.arange( -9, 10 )
        
        #Delete the all-zero rows, which spoil the statistical calculations:
        none  = np.where( c1[0] == 0. )  #and c1[1].all() == 0. and c1[ 2 ].all() == 0. )
        c2    = deepcopy( np.delete( c1,    none, axis = 1 ) ) 
        rangeN = deepcopy( np.delete( rangN, none           ) )
         
        c3 = np.transpose( c2 )
        frame = DataFrame(  c3 , columns = [ '      All Non-Div' , ' Best '+str(size) , 'Worst '+str(size) ], index=rangeN  )
        #frame[ frame[m] !=  np.zeros(3) for m in index ]
        frame.index.name = 'Score'        
        frameinfo = getframeinfo(currentframe())            
        print frameinfo.filename, frameinfo.lineno        
        print'------------------------------------------------------'        
        print 'Averaged median non-dividend stocks gains held for '+str(Ahead)+' weeks:'
        print 'Data interval is weeks '+str(num1)+' - '+str(numf)+'.'        
        print frame
        print frame.describe()        
        print'------------------------------------------------------'
        #https://www.google.com/search?q=python+pandas+data+frame+to+excel&ie=utf-8&oe=utf-8
        #http://xlsxwriter.readthedocs.io/example_pandas_simple.html
        #http://pbpython.com/excel-pandas-comp.html
        #http://pbpython.com/improve-pandas-excel-output.html

    #Dividend stocks table:
    if collect.count(0) == 0  :
        c1     = deepcopy( np.array( collect[ 3: ] ) )
        rangD = np.arange( -10,11 ) 
        
        #Delete the all-zero rows, which spoil the statistical calculations:
        noned  = np.where( c1[ 0 ] == 0 ) #and c1[ 1 ].any() == 0. and c1[ 2 ].any() == 0. ]
        c2    = deepcopy( np.delete( c1,    noned, axis = 1 ) ) 
        rangeD = deepcopy( np.delete( rangD, noned           ) )        
        print ' noned, rangeD = ', noned, rangeD
        c3 = np.transpose( c1 )
        fram = DataFrame(  c3 ,  columns = [ '       All Div' , ' Best '+str(size) , 'Worst '+str(size) ], index = rangD )
        fram.index.name = 'Score'
        frameinfo = getframeinfo(currentframe())            
        print frameinfo.filename, frameinfo.lineno        
        print'------------------------------------------------------'        
        print 'Averaged median dividend stocks gains held for '+str(Ahead)+' weeks:'
        print 'Data interval is weeks '+str(num1)+' - '+str(numf)+'.'
        print fram
        print fram.describe()
        print'------------------------------------------------------'     
        
    SaveAvgs[ ky ] = collect    
    SaveAvgs.close()   
    
    pl.axhline(y = 0)
    pl.legend(loc = 'upper left')
#    
    pl.show()
    
None
