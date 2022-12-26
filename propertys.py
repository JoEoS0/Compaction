#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:28:37 2020

  Individual shale properties, indicated by an author name, are RETURNED 
  starting with (current porosity, current depth, current age,  grain density, water density.) 
  Also inferred or guessed (surface paleo-temperature maximum, paleo geothermal gradient, 
  overburden, water pressure, and resultant sigma==S-p.) 
  
  UNITS: porosity:fractional, depth:km, age:million years, temperatures:Kelvin or centigrade,
      geothermal gradients: deg K/m==deg C/m, densities: kg/m**3,
      overburden and pressure: kgf/m**2
Ref 00  Simple quantitative evaluation of porosity of argillaceous sediments at various depths of burial
        Rashid D.Dzevanshir1Leonid A.Buryakovskiy1George V.Chilingarian2   \
        Sedimentary Geology Volume 46, Issues 3–4, February 1986, Pages 169-175 \
        Relationship of porosity (φ) to depth of burial (D, in m), geologic age (A, in millions of years),\
        and lithology (ratio of thickness of shales/total thickness of terrigenous deposits = R)\
        has been developed by the writers, which gives satisfactory results: where φ0 = initial\
        porosity of the argillaceous sediment. A nomogram has been prepared for an easy 
        solution of the above equation.
Ref 0        #https://sci-hub.se/   https://arxiv.org/    <<free preprints !
Ref 1 -  #https://www.researchgate.net/publication/341496386_AN_EASILY_USED_MATHEMATICAL\
    #_   MODEL_OF_POROSITY_CHANGE_WITH_DEPTH_AND_GEOLOGIC_TIME_IN_DEEP_SHALE_COMPACTION
    #    May 2020 International Journal of GEOMATE 19(73):108-115
    #    Avirut Puttiwongrak, P. H. Giao, Sakanann Vann
Ref 2    #https://en.wikipedia.org/wiki/Geologic_temperature_record#Overall_view
Ref 3    https://www.britannica.com/science/seawater/Density-of-seawater-and-pressure
Ref 4    #http://dandebat.dk/eng-klima5.htm  History of theEarth's Climate. T-pleistocene = 15 C.
Ref 5    KoichiAoyagiTadashiAsakawa
         Primary migration theory of petroleum and its application to petroleum exploration
         Organic Geochemistry Volume 2, Issue 1, January 1980, Pages 33-43
Ref 6    Geothermal gradient, Wikipedia
Ref 7    Cory H. Christie; Seiichi Nagihara, Geothermal gradients of the northern
         continental shelf of the Gulf of Mexico, Geosphere (2016) 12 (1): 26–34.
Ref 8    James W. Schmoker, Donald L. Gautier, Compaction of basin sediments:
         Modeling based on time‐temperature history, J Geophysical Research Solid Earth 
         Volume 94, Issue B6 10 June 198
Ref 9    L. F. Athy, Density, Porosity, and Compaction of Sedimentary Rocks1,
         AAPG Bulletin (1930) 14 (1): 1–24.
Ref 10   B.Velde, Compaction trends of clay-rich deep sea sediments. Marine Geology,
         Volume 133, Issues 3–4, August 1996, Pages 193-201
Ref 11   DAVID BLACKWELL and MARIA RICHARDS,Calibration of the AAPG Geothermal 
         Survey of North America BHT Data Base, AAPG04 Blackwell_and_Richards.pdf
         10 June 1989 Pages 7379-7386
Ref 12   Stephen Ehrenberg,P H Nadeau, Øyvind Steen, Petroleum reservoir porosity
         versus depth: Influence of geological age, Article in AAPG Bulletin · 
         October 2009DOI: 10.1306/06120908163
Ref 13   https://ui.adsabs.harvard.edu/abs/1990GeoRL..17.2073B/abstract
Ref 14   Berner, Ulrich; Bertrand, Philipp,Evaluation of the paleo-geothermal
         gradient at Site 768 (Sulu Sea), Geophysical Research Letters, Volume 17,
         Issue 11, p. 2073-2075, Pub Date: 1990. DOI: 10.1029/GL017i011p02073 
         Bibcode: 1990GeoRL..17.2073B  (250 cen/ km )
Ref15    Jacek A.Majorowicz, Ashton F.Embry,   Present heat flow and paleo-geothermal
         regime in the Canadian Arctic margin: analysis of industrial thermal data
         and coalification gradients, Tectonophysics Volume 291, Issues 1–4, 15
         June 1998, Pages 141-159.  (.028+-9 C/km)  
Ref 16   Sulu Sea bottom 55 deg F. https://www.history.noaa.gov/stories_tales/lukensphil.html 
Ref 17   70-90C ~ 2-2.5 km. Compaction and rock properties of Mesozoic and 
         Cenozoic mudstones and shales, northern North Sea
         M. K. Zadeh, N H Mondol,J Jahren Marine and Petroleum Geology
         Volume 76, September 2016, Pages 344-361
    Geologic ages / million years, for reference :
     Continental drift: https://www.youtube.com/watch?v=uLahVJNnoZ4   
    GeologicAges = { 'holocene':[0.,.01], 'pleistocene':[.01,1.8], 'pliocene':[1.8,5.3], \
            'miocene':[5.3,23.], 'neogene':[2.58,23.03], 'eocene':[33.7,54.8], \
            'paleogene':[65,105.], 'cretaceous':[65.5,145.5], \
            'triassic':[201.,252.], 'permian':[252.,299.], \
            'pennsylvanian':[299,323], 'carboniferous':[299.,359.] }  
@author: ed

"""
#import copy
import numpy as np

def shale( shale_name ):
    #https://sci-hub.se/   https://arxiv.org/    <<free preprints !
    #https://arxiv.org/    <<free preprints  
    print(shale_name)
    
    # Names of shales in Puttiwongrak order:    
    ShaleNames       = [ 'Akita',  'Makran1','Makran2', '\"SuluSea\"', 'Oklahoma', 'Maracaibo', 'ODP-DSDP' ]
    AuthorDate = [ 'Aoyagi etal. 1980', 'Fowler etal 1985', 'Fowler etal 1985','Velde 1996','Athy 1930','Hedberg 1936', 'Velde 1996'] 
    #print 'ShaleNames = ', ShaleNames    
    ShaleNumber = ShaleNames.index( shale_name )   
    #RETURN #1
    #print 'shale_name, ShaleNumber = ', shale_name, ShaleNumber
    
    # Depth in kilometers, in Puttiwongrak order. 1ft = .0003048km:
    DepthRanges = [ [.2, 3.], [0.30, 4.47], [ 0.36, 3.99 ], [.05, 4.], [.425, 2.], [.0887, 1.882], [.0,.5] ]
    DepthRanges[ 3] = [.05,5.]
    #RETURN #2
    #
    # Geologic ages are millions of years BP, in  Puttiwongrak order :
    AgeRanges = [ [ 5., 145.5 ], [ 2.58, 66. ], [2.58, 66. ], [ .01, 23. ], [ 254., 323. ],\
                  [ 2.58, 66. ], [ .1, 6. ]   ]
    #RETURN #3
    AgeName  = ['Miocene-Cretaceous', 'Paleogene-Neogene', 'Paleogene-Neogene',\
                 'Holocene-Miocene', 'Pennsylvanian-Permian','Paleogene-Neogene',\
                 'Northeastern Thailand']
        
    # Units are fractional porosity and km**{-1}, in Puttiwongrak order:     
    PorosityDepthParams = { 'Akita':[ .72, .656,0. ] , 'Makran1':[.7, -.796, .065 ] ,'Makran2':[.7,-1.239, .171 ], \
       '\"SuluSea\"':[ .56, .486, 0. ] , 'Oklahoma':[ .48, 1.457, 0. ]  ,          \
       'Maracaibo':[ .41, .698, 0.] , 'ODP-DSDP':[ .0875, .071, 0. ] }
    #RETURN # 4
    #
    #Eocene excess = 15 deg C,Miocene= +10, Pliocene = +4. Above present 1960-1990 average = 14 deg C.
    #Mud line deg C using global maximum paleo-temperatures. Pleistocene= 14 C maximum:
    #Mindanao av annual temperature = 31.5 deg C.
    #https://www.sciencedirect.com/science/article/abs/pii/S0967064506003006#:~\
    #:text=Thus%2C%20the%20Sulu%20Sea%20basin%20is%20unique%2C%20with,depths%20be\
    #low%201000%20m%20%28Rathburn%20et%20al.%2C%201996%29.
    #Ref 16   Sulu Sea bottom 55 deg F. https://www.history.noaa.gov/stories_tales/lukensphil.html
    # Continental drift video: https://www.youtube.com/watch?v=uLahVJNnoZ4      
    Cmud0, Cmud1, Cmud2, Cmud3, Cmud4, Cmud5 = (3.+14. +13.)/2. ,3. ,3.,  (13.+10.+14.)/2., (13.+14.+7.)/2., (3.+ 14.+13.)/2.
    Cmud3 = 12.7  #(12.7+31.5)/2.   #12.7 
    Cmud6 = 4.  #14+13.
    #Cmud7 = (3.+14.+13.)/2.
    PaleoMudlineT = np.array([  Cmud0, Cmud1, Cmud2, Cmud3, Cmud4, Cmud5, Cmud6 ])
    #RETURN #5
    #
    # Assumed paleo geothermal gradient. See Ref 5.
    #Oil window = 50-150 C,  Gas window = 150-200 C; see:
    # https://en.wikipedia.org/wiki/Kerogen5, .05, .02 ] #deg C/m
    GeothermalGradientMax = [   .05, .03,  .03, .03, .06,.05, .03  ]
    
    #Better m ?:                1.1   1.0  1.0  1.0   .85  1.0   
    #Better n ?:                1.0   0.7  0.6  0.5   1.0  0.9 
    #RETURN #6
    #Average temperature for all wells, Deg K::
    TKavg = np.zeros( len( ShaleNames ) )

    for i in range( len( ShaleNames ) ) :
        TKavg[i] = ( PaleoMudlineT[i] + GeothermalGradientMax[i]*DepthRanges[i][1]*1000. )/2. +273.2
    TKavgavg = np.mean(TKavg)
    #RETURN #7,8
    dy_steps = 500  # number of divisions for fitting porosity vs depth
    #RETURN # 1-8 :
    Low_Temp = 40.  #Don't use data at temperatures lower than this
    Deposition_Rate = 1.  # m/1000yr= m/3.171*pow( 10.-8) sec
    return ShaleNames[ShaleNumber], ShaleNumber, AgeRanges[ ShaleNumber ], DepthRanges[ShaleNumber], \
        PorosityDepthParams[ ShaleNames[ShaleNumber] ], PaleoMudlineT[ ShaleNumber ], \
            GeothermalGradientMax[ ShaleNumber ], TKavg[ShaleNumber], TKavgavg, dy_steps,\
            AuthorDate, Low_Temp,  Deposition_Rate, AgeName[ShaleNumber]
    #
#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH_Hedberg_Maracaibo_HHHHHHHHHHHHHHHHHHHHHHH    
    #Hedberg, H.D., 1936. Gravitational compaction of clays and shales.
    #American Journal of Science 31 (184), 241–287. p 279 for rho_g and p-254 for porosity-depth Maricaibo:
    #y =Depth ft, x=fractional porosity. rhog = 2.666, rhow = 1. 
    y = 291.,472.,497.,511.,862.,        922.,1637.,1805.,1920.,2031.,  2146.,2146.,2200.,2480.,2605.,\
        2780.,2818.,2996.,3015.,3094.,  3293.,3313.,3353.,3353.,3521.,  3702.,3973.,4336.,4608.,4849., \
        5007.,5035.,5389.,5502.,6013.,  6081.,6175.
        
    x = .3363,.3583,.3385,.3314,.3347,  .3133,.2776,.2738,.2875,.2662,  .2887,.2590,.2468,.2519,.2494, \
        .2532,.2288,.2423,.2561,.2489,  .2063,.2124,.1780,.1813,.2060,  .2220,.1855,.1777,.1680,.1420, \
        .1460,.1280,.1303,.1364,.0907,  .0918,.1060 
#
#        
#AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA_Aoyagi__Akita_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
    #shale( 'Akita' )                              
#         Y_values = np.array([222, 284, 308.5, 333, 358, 411, 477, 518, 880, 1080, 1259])
#         X_values = np.array([0.1282, 0.2308, 0.2650, 0.3120 , 0.3547, 0.4530, 0.5556,\
#                    0.6154, 0.8932, 0.9103, 0.9316])
    
#
#^^^^^^^^^^^^^^^^^^_Makran_Input_Data_^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  
    
    # x1 is abyssal plain porosity:
    x1 = np.array( [ .46,.31,.11,.46,.31,.29,.21,.15,.12,.35,.20,.14,.07,\
      .38,.16,.53,.36,.22,.11,.46,.24,.08,.07,.12,.07,.69,.41,.23,.13 ] )
    # y1 is abyssal plain depth in km:
    y1 = np.array( [ 0.30,0.97,1.86,0.30,0.97,1.53,1.71,2.37,3.23,0.92,1.84,\
      2.52,4.47,0.50,3.08,0.28,1.66,2.03,3.61,0.44,1.45,3.97,4.21,2.55,3.78,\
      0.05,0.31,1.72,3.04 ] )
    # x2 is accretionary prism porosity:
    x2 = np.array([.29,.16,.11,.07,.11,.07,.67,.28,.17,.15,.42,.18,.46,\
                           .35,.19,.40,.20,.16,.08,.50,.18,.12,.09 ] )
    # y2 is accretionary prism depth in km:
    y2   = np.array([0.80,1.64,2.20,3.49,2.35,3.99,0.21,0.84,1.84,1.57,0.55,\
                1.12,0.23,0.45,0.92,0.45,1.11,1.75,3.13,0.36,1.04,1.76,2.81  ])
    #
    # Here y1s increases monotonically. The matching x1s aren't monotonic:
    #y1s,x1s   = map( np.array, zip(*sorted( zip(y1 ,x1 ) ) ) )
    #print ' y1s,x1s = ', y1s, x1s  
    #
    #
    #
    #xxxxxxxxxxxxxxxxxxxxxxxxxxx_Dutta_Gulf_of_Mexico_xxxxxxxxxxxxxxxxxxxxxxxxx
    #TANIMA DUTTA , GARY MAVKO , and TAPAN MUKERJI , Stanford Rock Physics Laboratory
    #T IM L ANE , BP Exploration and Production Technology Group
    #Compaction trends for shale and clean sandstone iFor 0-6200 ft. No geologic age.
    #Coefficients of exponential fits using a general exponential equation:
    #Z = depth below mud-line (DBML) in ft.
    # phi = a*e^(b*z) = c*e^(d*z) a,b,c,d = .2875,-.007784,.4384,-.0001761 
    #GreenCanyon, Gulf of Mexico << may give T and age.
            
    #xxxxxxxxxxxxxxxxxxxxxxxxxRevil_South Eugene Islandxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #
    #Andre Revil, L. M. Cathles The Porosity-Depth Pattern defined by 40 wells in South Eugene Island
    #Block 330 area, and its relation to pore pressure, fluid leakage, and seal
    #migration 900m to 1950m Block 314,330. rhog, rhow = 2.65, 1.1
    #The minibasin is well described in Ph.D.
    #dissertations and in the published literature (cf., Holland et al. 1990; Alexander, 1995;
    #Alexander and Flemings, 1995; Coelho, 1997; Alexander and Handschy, 1998).
    #Holland, D. S., J. B. Leedy, and D. R. Lammelin, 1990, Eugene Island Block 330 field -
    #USA, offshore Louisiana, in Beaumont, E.A., and Foster, N.H., eds., Structural
    #Traps; III, Tectonic fold and fault traps,                                               
    #Alexander, L. L., and P. B. Flemings, 1995, Geologic evolution of a Pliocene-Pleistocene
    #salt withdrawal mini-basin: Eugene Island Block 330, offshore Louisiana, AAPG
    #Bull., v. 79, p. 1737-1756.
    #Alexander, L. L., and J. W. Handschy, 1998, Fluid flow in a faulted reservoir system:
    #fault trap analysis for the Block 330 field in Eugene Island, South Addition,
    #offshore Louisiana, AAPG Bull., v. 82, p. 387-411.     
    #Alexander, L. L., 1995, Geologic evolution and stratigraphic controls on the fluid flow of
    #10the Eugene Island Block                                          
    #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx_Athy_Oklahoma_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    
    #L. F. Athy, DENSITY, POROSITY, AND COMPACTION OFSEDIMENTARY ROCKS,BULLETIN
    #of the AMERICAN ASSOCIATION OF PETROLEUM GEOLOGISTS v14, Number 1, pp 1-24.
    #Figure 2:   
    Density = 1.40, 2.08, 2.245, 2.38, 2.480,  2.56, 2.60,  2.615, 2.63
    Depth   =    0., 1750.,2000.,  3200., 4000., 5200., 6000., 6400.                                                                                             
    # p = p { e - bx ), p = .48.   Figure 3, depth ft. :
    Phi   = .48, .26,   .20,  .126,   .075,  .05,   .03,   .022  
    Depth = 0., 1400., 2000., 3000., 4000., 5200., 6000., 7000. 
        
    #    In Figure 2, densities of samples from wells in the Mervine, South
    #Ponca, Thomas, Garber, and Blackwell fields are plotted. The sed-
    #iments range from the Enid Red-beds of Permian age to the Cherokee
    #shales in the base of the .Pennsylvanian, a stratigraphic range of about
    #4,000 feet.
    #If it is granted that the compaction depth relations in north-central
    #Oklahoma are approximately correct as given in this paper, then abou    
    #1,400 feet of Permian and younger sediments once covered the area
    #around Garber, Oklahoma, nearly 2,000 feet were present at Thomas
    #and Tonkawa, and 2,400-2,500 feet at Ponca City.
    #By applying the compaction-depth data derived from the areas just
    #named to the Nowata-Chelsea district southeast of Bartlesville, Okla-
    #homa, where densities range from 2.47 to 2.51 at depths ranging from
    #150 to 400 feet in the lower Pennsylvanian, it is estimated that the eroded
    #overburden is 4,000-4,500 feet. These figures indicate that the total
    #stratigraphic thickness deposited above the Oswego limestone in the
    #lower part of the Pennsylvanian was 1,000-1,500 feet less at Chelsea
    #than at Garber.   
    #The compaction noted by Hedberg in Hamilton County, Kansas, is
    #much less per thousand feet of burial than that observed by the writer
    #in Kay, Noble, and Garfield counties, in Oklahoma. Either there were
    #inherent differences in the character of the sediments in the two areas
    #which cause their compressibilities to differ, or some other factors, such
    #as the age of the formations or the nearness to centers of folding, cause
    #the difference.   
    
    #88888888888888888_Aoyagi_Akita_8888888888888888888888888888888888888888888
    
    DepthA = 0.,  440.,  790., 1270.,1850., 2620. 
    PhiA   = .75,  .50,  .40,  .30,   .20,   .10

    #88888888888888888_Aoyagi_Akita_8888888888888888888888888888888888888888888
                                    