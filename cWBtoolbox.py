#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:43:21 2020

@author: stuver

FUNCTION LISTING (in the order that they appear in this file):
    
    01 - rdLiveROOT:
        Read cWB livetime ROOT files into Pandas data frames and (optional) write to CSV files (requires: GWpy).  **Assumes H1L1 cWB run.**
    02 - plotBkgnd: 
        Plot the backgound false alarm rate and false alarm probability WRT cWB rho or SNR (per IFO).  **Assumes H1L1 cWB run.**
    03 - rdEVENTS: 
           Read cWB EVENTS text file into Pandas data frame.
    04 - igwncolor: 
           Retreive the hex color code for a desired IFO using the IGWN color scheme.
    05 - avPSD: 
           Calculate the average PSD over a time range or an array of times (requires: GWpy).  ** Usage for H1 or L1 only. **
    06 - det2ifo: 
           Assigns a string code for a detector based on the defult detector numbers used by cWB (cwb_parameters.C). 
    07 - getUnique: 
           Get the unique start and stop segments from backround information (live) for a given detector (det).
    08 - getFlags: 
           Get DQ observing flags for a LIGO detector and LIGO double coincidence.
    09 - getSNR: 
           Get the SNR at a specified background false alarm rate. This is calulated over a 1 hour period.
    10 - getPSD: 
           Returns 10 minute PSDs covering 1 hour.
    11 - select_f: 
           Select only T-F values corresponsing to a given frequency range: [flo, fhi).
    12 - brange: 
           Returns the spectrum of burst range given an SNR and the energy in solar masses converted into GW.
    13 - plot_brange: 
           Create plots of the glitch-adjusted burst range. 3 plots are produced: the range spectrum, the averaged rage, and the BNS range (for comparison).
    14 - query_idq:
           Queries iDQ local frame files from start to end for given IFO.
    15 - veto_events:
           Manually veto cWB events against a DQ flag.
    16 - which_run:
           Returns a string identifying the run label during the specified time. If time is not during a LVK run, the run label is 'unknown'.
    17 - cwb_files:
           Determine the location of the cWB EVENTS file and background (live) file based on the run label and if the results are ONLINE or OFFLINE.  ** THIS ASSUMES CIT LDG EXECUTION LOCATION! **
    18 - load_files:
           Load cWB background files (EVENTS, live) and unique background segments (unique).
    19 - brangeALIVE:
           Determines if this run has existing data signifying that a previous run crashed. If true, gathers the existing data and determines the point to restart the run.
    20 - veto_perdiem:
           Determines the percentage of vetoed background events for each day in the range of veto segments.
    21 - txt2segs:
           Converts a text file of segments to a gwpy.SegmentList.
    22 - segs2txt:
           Converts a gwpy.SegmentList to a text file of segments.
    23 - txt2flag:
           Converts a text file of segments to a gwpy.DataQualityflag.
    24 - mkfname:
           Format the path and file name to hold burst range results (final and intermediate).
    25 - integrate_exp_fit:
           Definite integral of exp(a*x)*exp(b) between a finite x-value and infinity. N.B. This solution is for all real values of a < 0.
    26 - getSNR2: 
           Copy of getSNR. Get the SNR at a specified background false alarm rate. This is calulated over a 1 hour period.Returns alert if SNR 
           is greater than input reference SNR

"""

# 01 =======================================================
# --- START rdLiveROOT()
def rdLiveROOT(root_file, path='.', out_file='', SLAG=-1, verbose=True):
    """
    Read cWB livetime ROOT files into Pandas data frames and (optional) write to CSV files (requires: GWpy).  **Assumes H1L1 cWB run.**
    
INPUTS:
    root_file  = (str) path to livetime ROOT file \n 
    path       = (str) path to save results \n
    out_file   = (str) file name for optional output CSV file (default is an empty string meaning DO NOT write CSV file) \n 
    SLAG       = (int) slag2 (unique slag id) to select (default is -1 meaning return all slags in ROOT file) \n
    verbose    = (bool) switch to display status updates \n

OUTPUTS:
    live       = (pd.DataFrame) Contents of the livetime ROOT file \n
    Columns: \n      
        run \n
        gps \n
        live \n
        lag<0, 1, 2> \n
        slag<0, 1, 2> \n
        start<0, 1, 2> \n
        stop<0, 1, 2> \n

Required libraries: gwpy, numpy, pandas, pickle
    """
    
    import numpy as np
    import numpy.random as rng
    import pandas as pd
    from gwpy.table import EventTable
    import pickle
    
    if verbose:
        print('-- Loading ROOT file...')
    # Read the ROOT file into a gwpy EventTable:
    table = EventTable.read(root_file, format='root', treename='liveTime') 
    
    # Make each of the col varaibles (viewed with table.keys()) an array:
    run = table.get_column('run')
    gps = table.get_column('gps')
    live = table.get_column('live')
    lag = table.get_column('lag')
    slag = table.get_column('slag')
    start = table.get_column('start')
    stop = table.get_column('stop')
    
    '''
    Notes for lag, slag, start, & stop:
    
    Each row is it's own array.  To access the whole array that corresponds to the columns (variables), use (example):
        lag[col]
    To access the nth element in the lag[col] array (example):
        lag[col][n]
    '''
    
    if verbose:
        print('-- Slect SLAG...')
    # Select SLAG:
    if SLAG >= 0:
        slag2 = [slag[k][2] for k in range(len(run))]
        # Select only slag2 = SLAG events:
        ndx = np.where(np.array(slag2) == SLAG)
        ndx = ndx[0]
    else:
        ndx = range(len(run))
    
    slag0 = [slag[k][0] for k in ndx]
    slag1 = [slag[k][1] for k in ndx]
    
    # Separate elements of (s)lag, start, stop
    # LAG:
    lag0 = [lag[k][0] for k in ndx]
    lag1 = [lag[k][1] for k in ndx]
    lag2 = [lag[k][2] for k in ndx]
    # SLAG:
    slag0 = [slag[k][0] for k in ndx]
    slag1 = [slag[k][1] for k in ndx]
    slag2 = [slag[k][2] for k in ndx]
    # START:
    start0 = [start[k][0] for k in ndx]
    start1 = [start[k][1] for k in ndx]
    start2 = [start[k][2] for k in ndx]
    # STOP:
    stop0 = [stop[k][0] for k in ndx]
    stop1 = [stop[k][1] for k in ndx]
    stop2 = [stop[k][2] for k in ndx]
    
    # Write ROOT info to Pandas fame (and CSV file)
    if verbose:
        print('-- Write selected SLAG to pd.DataFrame...')
    # Make dict:
    data = {'slag0': slag0, 'slag1': slag1, 'slag2': slag2, 'lag0': lag0, 'lag1': lag1, 'lag2': lag2, 'start0': start0, 'start1': start1, 'start2': start2, 'stop0': stop0, 'stop1': stop1, 'stop2': stop2, 'live': live[ndx], 'gps': gps[ndx], 'run': run[ndx]}
    
    # Convert dict to data frame:
    live = pd.DataFrame(data)
    
    if out_file != '':
        if verbose:
            print(f'-- Write livetime data to {path}/{out_file}')
        if out_file.lower().endswith('.p'):
            # Write to pickle file:
            pickle.dump(data, open(f'{path}/{out_file}', 'wb'))
        else:
            # Write to a csv file:
            live.to_csv(f'{path}/{out_file}')
    #if verbose:
    #    print('-> Done processing livetime file!')
        
    return live
# --- END rdLiveROOT()
# 02 =======================================================
# --- START plotBkgnd()
def plotBkgnd(data, wrt, bkg_time, zerolag, nbins=1000, ifo='', save='', plot=False):
    """
    Plot the backgound false alarm rate and false alarm probability WRT cWB rho or SNR (per IFO).  **Assumes H1L1 cWB run.**

INPUTS:
    data     = (array) \n
    wrt      = (str) quantity on horizontal axis; options: 'rho', 'snr' \n
    bkg_time = (float) \n
    zerolag  = (float) \n
    nbins    = (int) \n
    ifo      = (str) \n
    save     = (str) \n
    plot     = (bool) \n

OUTPUTS:
    prob     = (np.array) \n
    bins     = (np.array) \n

Required libraries: cWBtoolbox, matplotlib, numpy, scipy
    """
    
    import numpy as np
    from scipy.special import erf
    import matplotlib.pyplot as plt
    from cWBtoolbox import igwncolor
    
    # Convert bkg_time and zerolag from seconds to years:
    yr = 365*24*3600 # conversion factor (# seconds/year)
    bkg_time = bkg_time/yr
    zerolag = zerolag/yr
    
    # Make ifo lowercase to ensure proper comparisons:
    ifo = ifo.lower()
    
    # Determine size of data set:
    n = len(data)
    
    # Check to make sure wrt is 'rho' or 'snr':
    if wrt != 'snr' and wrt != 'rho':
        print("wrt must be 'rho' or 'snr'!")
    
    if plot:
        # Open plot object:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,6))
    
    if wrt == 'rho':
        # Bins for rho:
        maxrho = max(data)
        minrho = min(data)
        bins = np.linspace(minrho,maxrho+1,nbins)
        # Appropriate x-axis label:
        xlabel = r'zero-lag $\rho$'
    else:
        # Bins for snr:
        bins = np.linspace(0,1000,nbins+1)
        # Appropriate x-axis label:
        xlabel = r'zero-lag SNR'
        # Define color schemes for L1 and H1:
        livingston = igwncolor('L1')
        hanford = igwncolor('H1')
    
    # False Alarm Rate
    # Counts per bin:
    counts, _ = np.histogram(data,bins)
    # Poisson error per bin:
    error = np.sqrt(counts)
    
    # Rate:
    counts = np.vstack([counts-error, counts, counts+error])
    rate = np.flip( np.cumsum( np.flip(counts, axis=1), axis=1 ), axis=1)/bkg_time
    
    # Format for step plot:
    rate = np.insert(rate, 0, rate[:,0], axis=1)
    
    # Set legend switch to False by default:
    leg = False
    
    if plot:
        # Plot false alarm rate:
        if wrt == 'rho':
            ax1.step(bins, rate[1,:], 'm')
            ax1.fill_between(bins, rate[0,:], rate[2,:], step='pre', color='k', alpha=0.15)
        elif ifo == 'l1' or ifo == 'livingston' or ifo == 'llo':
            ax1.step(bins, rate[1,:], livingston, label='L1')
            ax1.fill_between(bins, rate[0,:], rate[2,:], step='pre', color=livingston, alpha=0.15)
            leg = True
        elif ifo == 'h1' or ifo == 'hanford' or ifo == 'lho':
            ax1.step(bins, rate[1,:], hanford, label='H1')
            ax1.fill_between(bins, rate[0,:], rate[2,:], step='pre', color=hanford, alpha=0.15)
            leg = True
        else:
            ax1.step(bins, rate[1,:], 'k')
            ax1.fill_between(bins, rate[0,:], rate[2,:], step='pre', color='k', alpha=0.15)
        ax1.set_yscale('log')
        ax1.set_ylabel('estimated rate  [$yr^{-1}$]')
        ax1.set_xlabel(xlabel)
        if wrt == 'snr' and leg:
            ax1.legend(loc='upper right')
        ax1.set_title(f'False Alarm Rate, n = {n}')
        ax1.grid(True)
    
    # False Alarm Probability:
    prob = 1 - np.exp(-rate*zerolag)
     
    # Define confidences wrt sigma:
    sigma1 = 1-erf(1/np.sqrt(2))
    sigma2 = 1-erf(2/np.sqrt(2))
    sigma3 = 1-erf(3/np.sqrt(2))
    sigma4 = 1-erf(4/np.sqrt(2))
    #sigma5 = 1-erf(5/np.sqrt(2))
    
    if plot:
        # Plot false alarm probability:
        if wrt == 'rho':
            ax2.step(bins, prob[1,:], 'm')
            ax2.fill_between(bins, prob[0,:], prob[2,:], step='pre', color='k', alpha=0.15)
        elif ifo == 'l1' or ifo == 'livingston' or ifo == 'llo':
            ax2.step(bins, prob[1,:], livingston)
            ax2.fill_between(bins, prob[0,:], prob[2,:], step='pre', color=livingston, alpha=0.15)
        elif ifo == 'h1' or ifo == 'hanford' or ifo == 'lho':
            ax2.step(bins, prob[1,:], hanford)
            ax2.fill_between(bins, prob[0,:], prob[2,:], step='pre', color=hanford, alpha=0.15)
        else:
             print('ifo input error')   
        ax2.axhline(y=sigma1, label=r'top to bottom: $1\sigma$, $2\sigma$, $3\sigma$, & $4\sigma$', c='k', linestyle='dashed', linewidth=1)
        ax2.axhline(y=sigma2, c='k', linestyle='dashed', linewidth=1)
        ax2.axhline(y=sigma3, c='k', linestyle='dashed', linewidth=1)
        ax2.axhline(y=sigma4, c='k', linestyle='dashed', linewidth=1)
        ax2.legend(loc='upper right')
        ax2.set_yscale('log')
        ax2.set_ylabel('probability')
        ax2.set_xlabel(xlabel)
        ax2.set_title(f'False Alarm Probability, n = {n}')
        ax2.grid(True)
        fig.tight_layout()
        
        # Save figure:
        if len(save) != 0:
            fig.savefig(f'{save}.png') 
    
    return rate, prob, bins
# --- END plotBkgnd()
# 03 =======================================================
# --- START rdEVENTS()
def rdEVENTS(filename, rows_to_skip=28, verbose=False):
    """
    Read cWB EVENTS text file into Pandas data frame.

INPUTS:
    filename     = (str) \n
    rows_to_skip = (int) \n

OUTPUTS:
    data         = (pd.DataFrame) contents of the cWB EVENTS file \n
    
    ** Alternative: 
    from gwpy.table import EventTable
    trigger_table = EventTable.read(filename, format='ascii.cwb')

Required libraries: pandas
    """
    
    import pandas as pd
    
    if filename.lower() == 'empty':
        # Return an empty DataFrame with the cWB EVENTS columns:
        dict = {'plus':[], 'minus':[], 'rho':[], 'CC_0':[], 'CC_2':[], 'CC_3':[], 'SNR_net':[], 'lag':[], 'slag':[], 'likelihood':[], 'pen_fac':[], 'energy_dis':[], 'freq':[], 'bw':[], 'dur':[], 'size':[], 'freq_res':[], 'run':[], 'GPS_L1':[], 'GPS_H1':[], 'SNR_L1':[], 'SNR_H1':[], 'hrss_L1':[], 'hrss_H1':[], 'PHI':[], 'THETA':[], 'PSI':[]}
        data = pd.DataFrame(dict)
    else:
        # Read in data for cWB EVENTS.txt file:
        data = pd.read_csv(filename, skiprows=rows_to_skip, delim_whitespace=True, header=None)
        data.columns = ['plus','minus','rho','CC_0','CC_2','CC_3','SNR_net','lag','slag','likelihood','pen_fac','energy_dis','freq','bw','dur','size','freq_res','run','GPS_L1','GPS_H1','SNR_L1','SNR_H1','hrss_L1','hrss_H1','PHI','THETA','PSI']
    
    return data
# --- END rdEVENTS()
# 04 =======================================================
# --- START igwncolor()
def igwncolor(ifo):
    """
    Retreive the hex color code for a desired IFO using the IGWN color scheme (https://gwpy.github.io/docs/stable/plot/colors.html).

INPUTS:
    ifo   = (str) interferometer identifier ('L1', 'Livingston', & 'llo' are all acceptable formats, but cWB det number is not) \n
    
OUTPUTS:
    color = (str) HTML hex color code \n 

Required libraries: NONE
    """
    
    # Make ifo lowercase to ensure proper comparisons:
    ifo = ifo.lower()
    
    # Select the hex color code for the desired IFO:
    if ifo == 'l1' or ifo == 'livingston' or ifo == 'llo':
        color = '#4ba6ff' # L1
    elif ifo == 'h1' or ifo == 'hanford' or ifo == 'lho':
        color = '#ee0000' # H1
    elif ifo == 'g1' or ifo == 'geo' or ifo == 'geo600':
        color = '#222222' # G1
    elif ifo == 'k1' or ifo == 'kagra':
        color = 'ffb200' # K1
    elif ifo == 'i1' or ifo == 'ligo-india' or ifo == 'india' or ifo == 'lio':
        color = 'b0dd8b' # I1
    elif ifo == 'v1' or ifo == 'virgo':
        color = '9b59b6' # V1
    else:
        print("ifo must be 'L1', 'H1', 'V1', 'K1', 'G1', or 'I1'!")
    
    return color
# --- END igwncolor()
# 05 =======================================================
# --- START avPSD()    
def avPSD(time, ifo, FFTlen=8, OverlapFrac=0.5, freqRange=[-1], checkFlag=False, verbose=False, plot=False):
    """
    Calculate the average PSD over a time range or an array of times (requires: GWpy).  ** Usage for H1 or L1 only. **

INPUTS:
    time        = \n
    ifo         = (str) \n
    FFTlen      = (int) \n
    OverlapFrac = (float) \n
    freqRange   = (arraylike) \n
    checkFlag   = (bool) \n
    verbose     = (bool) switch to display status updates \n
    plot        = (bool) \n
    
OUTPUTS:
    meanPSD     = (arraylike) \n
    f           = (arraylike) \n

Required libraries: gwpy, matplotlib, numpy
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from gwpy.timeseries import TimeSeries
    from gwpy.segments import DataQualityFlag
    
    # Check format of freqRange:
    if len(freqRange) <= 2 and len(freqRange) > 0:
        if len(freqRange) == 2:
            if freqRange[0] < 0 or freqRange[1] < 0:
                print(f'ERROR: a 1x2 freqRange must be postitive!  freqRange = {freqRange}')
                # !!! Change to ValueError
            if freqRange[0] >= freqRange[1]:
                print(f'ERROR: freqRange[1] must be <= freqRange[1]!  freqRange = {freqRange}')
                # !!! Change to ValueError
        else: # len(freqRange) = 1 (use full PSD range)
            # Define the sample frequency of CALIB_STRAIN:
            fs = 16384
            # Define the Nyquist frequency:
            nyquist = fs/2
            # Determine the full frequency range of the PSD:
            freqRange = [0, nyquist - (nyquist%(1/FFTlen))]
    else:
        print(f'ERROR: freqRange must be a scalar or a 1x2 list/array!  freqRange = {freqRange}')
        # !!! Replace this with ValueError
    
    # Time can be a list of 2 times, but must be an array of arrays in the code below.
    # If an array of 2 times, set it as an array of arrays:
    if np.size(time) == 2:
        time = np.array([time])
    
    # Caluclate the overlap:
    Overlap = FFTlen*OverlapFrac # seconds
    
    # Determine the length of resulting PSD: 
    PSDlen = int((16384*(FFTlen/2))+1)
    
    # Allocate variables to hold PSD information (per time segment):
    meanPSD = np.zeros([np.shape(time)[0],PSDlen])
    # Initiate variable for total PSD averaged:
    totAve = 0
    # Initiate variable for total averaged time:
    totTime = 0
    
    # Loop over time segments:
    for m in range(np.shape(time)[0]):
        start = time[m,0]
        stop = time[m,1]
        
        if verbose:
            print(f'Processing time segment {m+1} of {np.shape(time)[0]}:   [{int(time[m][0])} {int(time[m][1])})')
        
        if checkFlag:
            # Determine when each IFO was in observing mode:
            flagL = DataQualityFlag.query('L1:DMT-ANALYSIS_READY:1', start, stop)
            flagH = DataQualityFlag.query('H1:DMT-ANALYSIS_READY:1', start, stop)
            # Determine coincident observing time:
            both = flagH.active & flagL.active 
            # !!! Replace this stuff with coincFlag?
            # !!! Add vetoFlag as input and inclue in line above for vetos
        else:
            #both = SegmentList([Segment(start, stop)])
            both = np.array([[start, stop]])
        
        # Determine the number of conincident obs segments:
        nSeg = len(both)
        
        # Allocate variables to hold PSD information (per obs segment):
        psdAve = np.zeros([nSeg,PSDlen])
        
        # Initiate variable for total observing time:
        dt = 0
        
        # Loop over segments:
        for k in range(nSeg):
            if verbose:
                print(f'-> Processing obs. segment {k+1} of {nSeg}:   [{int(both[k][0])} {int(both[k][1])})')
            # Calculate the duration of the segment:
            dur = both[k][1]-both[k][0]
            
            if checkFlag:
                # Add dur to running total of observing time:
                #dt += dur.ns()*1e-9
                dt += dur.gpsSeconds
            else:
                dt += stop-start
            
            # Determine the # of averages making up PSD:
            nFFTave = (2*dur/FFTlen)-1
            if nFFTave%OverlapFrac >= OverlapFrac:
                nFFTave += 1
            nAve = np.floor(nFFTave)
            totAve += nAve
            
            # Fetch observing data for IFO:
            data = TimeSeries.get(f'{ifo}:GDS-CALIB_STRAIN', both[k][0], both[k][-1])
            # Calculate PSD
            psd = data.psd(fftlength=FFTlen, overlap=Overlap)
            # Weight PSD by the number of averages:
            psdAve[k,:] = psd.to_value()*nAve
        
        # Add this time segment to the total averaged time:
        totTime += dt
        # Calculate the sum of PSD over all segments:
        meanPSD[m,:] = np.sum(psdAve, axis=0)
    
    # Calculate the final mean PSD:
    meanPSD = np.sum(meanPSD, axis=0)/totAve
    
    # Calculate frequency bins:
    df = psd.df.to_value()
    f = np.arange(0, (fs/2)+df, df)
    
    # Only return a select range frequency bins:
    if len(freqRange) == 2:
        ndxStart = np.where(f >= freqRange[0])
        ndxStop = np.where(f < freqRange[1])
        ndx = np.intersect1d(ndxStart, ndxStop)
        meanPSD = meanPSD[ndx]
        f = f[ndx]
    
    # Plot results:
    if plot:
        from cWBtoolbox import igwncolor
        ifoColor = igwncolor(ifo)
        plt.loglog(f, meanPSD, ifoColor)
        plt.xlim([freqRange[0], freqRange[1]])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Time [Hz$^{-1}$]')
        plt.title(f'Average PSD for {ifo}\n{int(totTime)} sec of Obs. time')
    
    return meanPSD, f
# --- END avPSD()
# 06 =======================================================
# --- START det2ifo()
def det2ifo(det):
    """
    Assigns a string code for a detector based on the defult detector numbers used by cWB (cwb_parameters.C).
    
INPUTS:
    det = (int) number representing a detector \n

OUTPUTS:
    ifo = (string) label of detector \n

Required libraries: NONE
    """

    # Assign ifo string based on det code:
    if det == 0:
        ifo = 'L1'
    elif det == 1:
        ifo = 'H1'
    elif det == 2:
        ifo = 'V1'
    elif det == 3:
        ifo = 'I1'
    elif det == 4:
        ifo = 'K1' # !!! cWB uses 'J1'
    elif det == 5:
        ifo = 'G1'
    else:
        print(f'ERROR: det must be int >0 and <= 6(entered {det})!')
        
    return ifo
# --- END det2ifo()
# 07 =======================================================
# --- START getUnique()
def getUnique(live, det, path, outFile=False, file='', on_off='online', verbose=True):
    """
    Get the unique start and stop segments from backround information (live) for a given detector (det).
    
INPUTS:
    live    = (pd.DataFrame) contents of the livetime ROOT file \n
    ifo     = (int) detector code: 0 == L1, 1 == H1 \n
    path    = (str) file name to save results to.  If empty, will use default file name: unique-<ifo>.txt.\n
    outFile = (bool) switch to write results to a csv file (True) or not (False) \n
    file    = (str) file name to save unique segments. \n
    on_off  = (str) switch to process 'online' or 'offline' results \n
    verbose = (bool) switch to display status updates \n 
    
OUTPUTS:
    unique  = (pd.DataFrame) unique start and stop segments with the same columns as the live input \n      

Required libraries: cWBtoolbox, numpy, pandas
    """
    
    import numpy as np
    import pandas as pd
    import cWBtoolbox as cwb
    
    ifo = cwb.det2ifo(det)
    
    # Determine unique start and stop times:
    start = []
    stop = []
    for k in live.start1.unique():
        query = f'start{det} == {k}'
        if ifo == 'L1':
            end = live.query(query).stop0.unique()
        elif ifo =='H1':    
            end = live.query(query).stop1.unique()
        else:
            print('det must be 0 or 1')
        for m in end:
            start = np.append(start, k)
            stop = np.append(stop, m)
    dict = {'start': start, 'stop': stop}
    unique = pd.DataFrame(dict)
    
    # Save unique segments:
    if outFile:
        if not file:
            if on_off == 'online':
                from gwpy.time import from_gps
                date = from_gps(np.mean(start))
                file = f'{path}/{ifo}/unique-{ifo}-{date.year}-{date.month}-{date.day}.txt'
            else:
                run, _ = cwb.which_run(np.mean(start))
                file = f'{path}/{ifo}/unique-{ifo}-{run}.txt'  
        if verbose:
            print(f'-- Write unique background segments to {file}')
        unique.to_csv(file)
    
    return unique
# --- END getUnique()
# 08 =======================================================
# --- START getFlags()
def getFlags(ifo, start, stop):
    """
    Get DQ observing flags for a LIGO detector and LIGO double coincidence.
    
INPUTS:
    ifo   = (string) code for LIGO detector of interest; 'L1' or 'H1' \n
    start = (float) GPS start time \n
    stop  = (float) GPS stop time \n

OUTPUTS:
    flag1 = observing DQ flag for detector specified in ifo input \n
    coinc = double coincidence DQ flag for the 2 LIGO detectors \n

Required libraries: gwpy
    """
    
    from gwpy.segments import DataQualityDict
    
    # Get flags L1 and H1 flags in a dictionary:
    flags = DataQualityDict.query(['L1:DMT-ANALYSIS_READY:1', 'H1:DMT-ANALYSIS_READY:1'], start, stop)
    
    # Observing flag for ifo:
    flag1 = flags[f'{ifo}:DMT-ANALYSIS_READY:1']
    # Determine what the "other" LIGO detecor is:
    if ifo == 'L1':
        flag2 = flags['H1:DMT-ANALYSIS_READY:1']
    else:
        flag2 = flags['L1:DMT-ANALYSIS_READY:1']
    
    # Determine the double coincidence:
    coinc = flag1 & flag2
    
    # Rename the flags:
    flag1.name = "Observing"
    coinc.name = "Coincidence"
    coinc.description = "H1 & L1 BOTH Ready for Analysis (Science & Calibrated) from h(t) DQ flags"
    
    return flag1, coinc
# --- END getFlags()
# 09 =======================================================
# --- START getSNR
def getSNR(hstart, hstop, thresh, det, live, events, unique, thresh_on='rate', plot=False, study=False, verbose=True):
    """
    Get the SNR at a specified background false alarm rate.  This is calulated over a 1 hour period.

INPUTS:
    hstart     = \n
    hstop      = \n
    thresh     = \n
    det        = (int) \n
    live       = (pd.DataFrame) \n
    events     = (pd.DataFrame) \n
    unique     = \n
    thresh_on  = (str) 'prob' or 'rate' : apply threshold to false alarm probability or false alarm rate \n
    plot       = (bool) \n
    study      = (bool) switch to save intermediate data to thresh_temp_bkgnd.p for threshold studies \n
    verbose    = (bool) switch to display status updates \n

OUTPUTS:
    snr        = \n

Required libraries: cwBtoolbox, numpy, pandas, pickle
    """
    
    import pdb # pdb.set_trace()  <- to insert breakpooint
    import cWBtoolbox as cwb
    import numpy as np
    import pandas as pd
    import pickle
    
    # Get background live time:
    bkgndtime = live.query("start" + str(det) + " >= " + str(hstart) + " and stop" + str(det) + " <= " + str(hstop)).live.sum()
    if verbose:
        print(f'-> background time = {bkgndtime}')
    # Find unique segments in background:
    hold = unique.query('start >= ' + str(hstart) + ' and stop <= ' + str(hstop))
    # Find zero-lag live time:
    zerolag = (hold.stop-hold.start).sum() # should I be using flags to query zero-lag time? -No! Seg length was 600 or bust.
    if verbose:
        print(f'-> zero-lag time = {zerolag}')
    
    if type(events) is not dict:
        trials = 1
    else:
        trials = len(events.keys())
    
    snr = np.zeros(trials) # preallocate snr array
    for q in range(trials):
        if type(events) is dict:
            EVENTS = events[str(q)]
        else:
            EVENTS = events
        # Get events in this hour:
        data = pd.DataFrame()
        for m in range(len(hold)): # !!! MIKE:  I think the issue may be btwn here and line 762.
            # Get events in this hour:
            if det == 0:
                temp = EVENTS.query('GPS_L1 >= ' + str(hold.iloc[m].start) + " and GPS_L1 <" + str(hold.iloc[m].stop))
            elif det == 1:
                temp = EVENTS.query('GPS_H1 >= ' + str(hold.iloc[m].start) + " and GPS_H1 <" + str(hold.iloc[m].stop))
            else:
                print('ERROR: det must be 0 or 1')
            data = pd.concat([data, temp], axis=0).reset_index(drop=True)
        if data.empty:
            if verbose and q == trials-1:
                print('-> no events')
            snr[q] = float('nan') # return snr = NaN if there are no events in this hour
        else:
            if det == 0:
                rate, prob, bins = cwb.plotBkgnd(data.SNR_L1, 'snr', bkgndtime, zerolag, nbins=1000, ifo='L1', plot=plot)
            elif det == 1:
                rate, prob, bins = cwb.plotBkgnd(data.SNR_H1, 'snr', bkgndtime, zerolag, nbins=1000, ifo='H1', plot=plot)
            else:
                print('ERROR: det must be 0 or 1')
            
            if study:
                # This is needed for threshold studies:
                background = {'rate':rate, 'prob':prob, 'bins':bins, 'zerolag':zerolag, 'bkgndtime':bkgndtime}
                pickle.dump(background, open('thresh_temp_bkgnd.p', 'wb'))

            # Find indices where rate/prob is greater than the threshold:
            if thresh_on.lower() == 'rate':
                # Determine where rate >= rateThresh:
                ndx = np.where(rate[1,:] >= thresh)
            elif thresh_on.lower() == 'prob': 
                # Determine where prob >= thresh:
                ndx = np.where(prob[1,:] >= thresh)
            else:
                print('getSNR(): thresh_on must == "prob" or "rate"!')

            # Did thresh intersect with the background distribution?
            if len(ndx[0]) > 0: # The distribution crosses the threshold value:
                # Determine smallest SNR with rate >= rateThresh:
                snr[int(q)] = bins[ndx[0][-1]+1]
                # if snr[int(q)] > refSNR: # Red flag condition!
                #    pdb.set_trace() # this means pause and let me poke around the memory
                # !!!  MIKE: Poke around, may want to run something like this:
                # rate, prob, bins = cwb.plotBkgnd(data.SNR_H1, 'snr', bkgndtime, zerolag, nbins=1000, ifo='H1', plot=True)
                if verbose and q == trials-1:
                    print(f'-> SNR = {snr}')
            else: # Thresh doesn't cross distribution, don't calculate:
                snr[int(q)] = float('nan')
                if verbose: #and q == trials-1:
                    print(f'-> No SNR: thresh didn\'t intersect with background distribution!')
        
    return snr
# --- END getSNR()
# 10 =======================================================
# -- START getPSD()
def getPSD(data, ifo, flag, minTime=600, FFTlen=8, overlap=4, flo=32, fhi=2048, plot=False, verbose=True):
    """
    Returns 10 minute PSDs covering 1 hour.
    
INPUTS:
    data     = \n
    ifo      = (str) \n
    flag     = \n
    minTime  = (float) \n
    FFTlen   = (float) \n
    overlap  = (float) \n
    flo      = (int) \n
    fhi      = (int) \n
    verbose  = (bool) switch to display status updates \n

OUTPUTS:
    PSD      = \n
    f        = \n
    flagStat = \n

Required libraries: cWBtoolbox, gwpy, numpy, matplotlib (if plot == True)
    """
    
    import cWBtoolbox as cwb
    import numpy as np
    from gwpy.segments import Segment, SegmentList
    from gwpy.time import to_gps, from_gps
    import gwpy.astro as astro
    
    # Determine start and stop time of data:
    hstart = data.span[0]
    hstop = data.span[1]
    # Determine the number of 10 minute segments:
    nten = int((hstop-hstart)/600)
    
    # Determine the length of resulting PSD:
    PSDlen = int((16384*(FFTlen/2))+1)
    
    # Allocate variables to hold PSD information (per time segment):
    PSD = np.zeros([nten,PSDlen])
    
    flagStat = [False, False, False, False, False, False]
    # Loop through 10 min segments:
    for tmin in range(nten):
        mstart = hstart+(tmin*600)
        mstop = mstart+600
        mseg = SegmentList([Segment(mstart, mstop)]) & flag.active 
            
        if len(mseg) > 0:
            if verbose:
                print(f'--- PSD: Calculating for 10 min starting {from_gps(mstart)}')
            
            mseg = mseg.to_table()
            # Determine if there is enough data to work with:
            mseg.remove_rows(np.where(mseg['duration']<minTime))
                
            for seg in range(len(mseg)):
                # Record if there is coincident data during this 10 min:
                flagStat[tmin] = True
                
                # select subsegment of data:
                hold = data.crop(mseg[seg][1], mseg[seg][2])
                
                # Calculate PSD
                psd = hold.psd(fftlength=FFTlen, overlap=overlap, method='median')
                                
                # Make PSD array and the corresponding frequency array:
                S = psd.to_value()
                df = psd.df.to_value()
                f = np.arange(0, ((hold.sample_rate.to_value())/2)+df, df)
                PSD[tmin,:] = S
                
                # Plot PSD if desired:
                if plot:
                    import matplotlib.pyplot as plt
                    ax = psd.plot()
                    p = ax.gca()
                    p.set_title(f'Power Spectral Desity for {ifo} between {mseg[seg][1]} and {mseg[seg][2]}')
                    p.set_xlim(flo, fhi)
                    p.set_ylim([1e-47, 1e-37])
        else:
            if verbose:
                print(f'--- PSD: * No data * for 10 min starting {from_gps(mstart)}')
            f = np.arange(0, (16384/2)+(FFTlen**-1), (FFTlen**-1))
    
    # Only return date within specified bandwidth [flo, fhi):
    PSD, f = cwb.select_f(PSD, f, flo, fhi)
    
    return PSD, f, flagStat
# --- END getPSD()
# 11 =======================================================
# --- START select_f()
def select_f(PSD, f, flo=16, fhi=2048):
    """
    Select only T-F values corresponsing to a given frequency range: [flo, fhi).
    
INPUTS:
    PSD = (array) T-F data, like a PSD[time, frequency] \n
    f   = (vector) frequencies corresponding to the data \n
    flo = (scalar) minimum frequency value to keep \n
    fhi = (scalar) select frequencies up to but not including this \n

OUTPUTS:
    PSD = (array) selected T-F data \n
    f   = (vector) selected frequencies \n

Required libraries: numpy
    """
    
    import numpy as np
    
    # Find indices where f > flo:
    nlo = np.where(f >= flo)
    # find indices where f >= fhi:
    nhi = np.where(f < fhi)
    
    # Find indecies where f > flo AND f >= fhi:
    ndx = np.intersect1d(nlo,nhi)

    # Return only selected indices:
    f = f[ndx]
    PSD = PSD[:, ndx]
    
    return PSD, f
# --- END select_f()
# 12 =======================================================
# --- START brange()
def brange(psd, f, snr, Egw):
    """
    Returns the spectrum of burst range given an SNR and the energy in solar masses converted into GW.
Ref:\n
Sutton, Patrick, "Rule of Thumb for the Detectability of Gravitational-Wave Bursts", arXiv:1304.0210v1, LIGO DCC# P1000041-v3
    
INPUTS:
    psd = \n
    f   = \n
    snr = (float) \n
    Egw = (float) \n

OUTPUTS:
    R   =  \n

Required libraries: astropy, numpy
    """
    
    import numpy as np
    import astropy.constants as const
    
    # Determine size of snr:
    wholeHR = True # apply SNR to whole hour by default
    if len(snr) == 6:
        wholeHR = False # SNR calculated for each 10 min
    
    # Convert Egw from solar masses to J:
    Egw = Egw*const.M_sun*const.c**2
    
    # Variable to hold range results:
    if psd.ndim == 1:
        R = np.zeros(len(psd))
        rows = 1
    else:
        R = np.zeros(np.shape(psd))
        rows = np.shape(psd)[0]
    
    for k in range(rows):
        if psd.ndim == 1:
            if f[0] == 0: # exclude freq = zero to avoid divide by zero
                R[1:] = np.sqrt( (const.G*Egw) / (2*np.pi**2*const.c**3*psd[1:]*f[1:]**2*snr**2) ) / (const.pc*1e+3)
            else:
                 R = np.sqrt( (const.G*Egw) / (2*np.pi**2*const.c**3*psd*f**2*snr**2) ) / (const.pc*1e+3)
        else:
            if wholeHR: # snr is a scalar
                if f[0] == 0: # exclude freq = zero to avoid divive by zero
                    R[k,1:] = np.sqrt( (const.G*Egw) / (2*np.pi**2*const.c**3*psd[k,1:]*f[1:]**2*snr**2) ) / (const.pc*1e+3)
                else: 
                    R[k,:] = np.sqrt( (const.G*Egw) / (2*np.pi**2*const.c**3*psd[k,:]*f**2*snr**2) ) / (const.pc*1e+3)
            else: # snr is a vector
                if f[0] == 0: # exclude freq = zero to avoid divide by zero
                    R[k,1:] = np.sqrt( (const.G*Egw) / (2*np.pi**2*const.c**3*psd[k,1:]*f[1:]**2*snr[k]**2) ) / (const.pc*1e+3)
                else:
                    R[k,:] = np.sqrt( (const.G*Egw) / (2*np.pi**2*const.c**3*psd[k,:]*f**2*snr[k]**2) ) / (const.pc*1e+3)
    
    return R
# --- END brange()
# 13 =======================================================
# --- START plot_brange()
def plot_brange(R, f, gps_start, ifo, flag, coincFlag, path, DQflag='', save=True, flo=32, fhi=2048, vmin=0.01, vmax=300, cmap='viridis', norm='log', shading='auto'):
    """
    Create plots of the glitch-adjusted burst range.  3 plots are produced: the range spectrum, the averaged rage, and the BNS range (for comparison).
    
INPUTS:
    R         = \n
    f         = \n
    gps_start = \n
    ifo       = \n
    flag      = \n
    coincFlag = \n
    path      = \n
    DQflag    = \n
    save      = \n
    flo       = \n
    fhi       = \n
    vmin      = \n
    vmax      = \n
    cmap      = \n
    norm      = \n
    shading   = \n
    
OUTPUTS:
    T         = \n
    F         = \n
    t         = \n
    meanRange = \n
    BNS       = \n

Required libraries: gwpy, numpy, matplotlib
    """
    
    import numpy as np
    from gwpy.plot import Plot
    from gwpy.time import from_gps
    from gwpy.timeseries import TimeSeries
    from matplotlib.colors import LogNorm
    
    # Is R a dictionary?
    if type(R) is dict:
        isdict = True
    else:
        isdict = False
    
    # Determine the number of trials (none if not a dictionary):
    if isdict:
        trials = len(R.keys())
    else:
        trials = 1
    
    # Set a threshold to limit the amound plots produced:
    plotThresh = 11 # only disply plot(s) if less than this
    
    # Preserve the original R varibale passed to function to draw trials from:
    hold_R = R
    
    # Loop over trials:
    for trial in range(trials):
        if isdict:
            R = hold_R[str(trial)] # select R for this trial
            if trial == 0: # Only allocate meanRange when processing the first trial
                points = np.shape(R)[0]  # !!! START HERE
                meanRange = np.zeros([trials,points+1])
        else:
            points = np.shape(R)[0]
            meanRange = np.zeros([trials,points+1])
        # Make array with correct units of time for R:
        start_date = from_gps(gps_start)
        hours = int(np.shape(R)[0]/6) # determine the number of hours from number of rows in range array
        decimals = np.array([0, 1, 2, 3, 4, 5])*1/6 # make base array of decimal hours corresponging to 10 min segs
        t = [] # create time variable
        for k in range(hours):
            t = np.append(t, k+decimals) # append hours+deicimal hours to time variable
        #t = np.append(t, t[-1]+(1/6)) # add one more 10 min segment to use flat shading in pcolormesh (flat depreiciated!)
        t = t*3600+gps_start # convert to GPS

        # Make meshgrid time and frequency array for pcolormesh:
        T, F = np.meshgrid(t,f)
        
        if trials < plotThresh:
            # Plot range spectrum:
            p = Plot(figsize=(12, 6)) # define handle to the plot, returned at end of function
            ax = p.gca()
            ax.pcolormesh(T, F, np.transpose(R), vmin=vmin, vmax=vmax, cmap=cmap, norm=LogNorm(), shading=shading)
            ax.set_xscale('auto-gps', epoch=from_gps(gps_start))
            ax.colorbar(cmap=cmap, norm=norm, clim=(vmin, vmax), label=r'Range [kpc]')
            ax.set_ylim(flo, fhi)
            ax.set_yscale('log')
            ax.set_ylabel('Frequency [Hz]')
            ax.set_title(f'{ifo} Narrowband Burst Range')
            if not DQflag: # if no DQflag segs are provided...
                p.add_segments_bar(flag)
                p.add_segments_bar(coincFlag)
            else: # if provided, show DQflag segs...
                p.add_segments_bar(coincFlag)
                p.add_segments_bar(DQflag)
            p.show()
            if save:
                if not isdict:
                    p.savefig('{:s}/{:s}/brange-spec-{:s}-{:04.0f}-{:02.0f}-{:02.0f}.png'.format(path, ifo, ifo, start_date.year, start_date.month, start_date.day))
                else:
                    p.savefig('{:s}/{:s}/brange-spec-{:s}-{:04.0f}-{:02.0f}-{:02.0f}-{:f}.png'.format(path, ifo, ifo, start_date.year, start_date.month, start_date.day, trial))
        
        # If R is a dictionary, return T as a dictionary:
        if not isdict:
            Tdict = T
        else:
            if trial == 0:
                Tdict = {str(trial): T}
            else:
                Tdict[str(trial)] = T
            
        # Calculate the mean range between flo and fhi (assume R already selected for these frequencies):
        for k in range(points):
            test = np.mean(R[k])
            if not np.isnan(test) and not np.isinf(test):  # if finite
                if k == 0:
                    meanRange[trial,k] = test # format meanRange for stem plots
                meanRange[trial,k+1] = test
        t = np.append(t, t[-1]+(1/6))
        
        if trials < plotThresh:
            # Plot the mean range:
            if ifo == 'L1':
                color = 'gwpy:ligo-livingston'
            else:
                color = 'gwpy:ligo-hanford'
                
            nozeros_meanRange = meanRange[meanRange != 0]
            mean = np.mean(nozeros_meanRange)
            std = np.std(nozeros_meanRange)
            N = len(nozeros_meanRange)
            error = std/np.sqrt(N)    
            
            
            pp = Plot(figsize=(12, 6))  
            ax = pp.gca()
            ax.step(t, meanRange[trial,:], color=color)
            #ax.step(t, meanRange[trial,:], color=color,label=f'Mean Range = {mean:.3f}kpc $\pm$ {error:.3f}kpc')
            ax.set_title(f'{ifo} Mean Narrowband Burst Range Between {flo}-{fhi} Hz')
            ax.set_ylabel('Range [kpc]')
            ax.legend([f'Mean Range = {mean:.3f}kpc $\pm$ {error:.3f}kpc'], loc='upper right')
            #ax.legend(loc='upper right')
            ax.set_xscale('auto-gps', epoch=from_gps(gps_start))
            if not DQflag: # if no DQflag segs are provided...
                pp.add_segments_bar(flag)
                pp.add_segments_bar(coincFlag)
            else: # if provided, show DQflag segs...
                pp.add_segments_bar(coincFlag)
                pp.add_segments_bar(DQflag)
            pp.show()
            if save:
                if not isdict:
                    pp.savefig('{:s}/{:s}/brange-mean-{:s}-{:04.0f}-{:02.0f}-{:02.0f}.png'.format(path, ifo, ifo, start_date.year, start_date.month, start_date.day))
                else:
                    pp.savefig('{:s}/{:s}/brange-mean-{:s}-{:04.0f}-{:02.0f}-{:02.0f}-{:f}.png'.format(path, ifo, ifo, start_date.year, start_date.month, start_date.day, trial))
        
        # If R is a dictionary, return t as a dictionary:
        if not isdict:
            tdict = t
        else:
            if trial == 0:
                tdict = {str(trial): t}
            else:
                tdict[str(trial)] = t
            
        '''
        # Code to calculate the mean range in the "bucket":
        bucket = [30, 1000]
        Rb, fb = select_f(R, f, flo=bucket[0], fhi=bucket[1])

        pointsb = np.shape(Rb)[0]
        meanRangeb = np.zeros(pointsb)
        for k in range(pointsb):
            test = np.mean(Rb[k])
            if not np.isnan(test):
                if not np.isinf(test):
                    meanRangeb[k] = test
        meanRangeb = np.hstack([meanRangeb[0], meanRangeb])

        ppb = Plot(figsize=(20, 10))  
        ax = ppb.gca()
        ax.step(t, meanRangeb, color=color)
        ax.set_title(f'{ifo} Integrated Narrowband Burst Range (BUCKET: [{flo}, {fhi})])')
        ax.set_ylabel('Range [kpc]')
        ax.set_xscale('auto-gps', epoch=from_gps(gps_start))
        ppb.add_segments_bar(flag)
        ppb.add_segments_bar(coincFlag)
        ppb.show()
        '''
    # END loop over trials
    
    '''
    # Plot BNS range:
    BNS = TimeSeries.get(f'{ifo}:DMT-SNSH_EFFECTIVE_RANGE_MPC.mean', gps_start, gps_start+(24*3600), pad=0, verbose=False)
    
    if trials < plotThresh:
        bns = Plot(figsize=(12, 6))  
        ax = bns.gca()
        ax.plot(BNS, color=color)
        ax.set_title(f'{ifo} BNS Range')
        ax.set_ylabel('Range [Mpc]')
        ax.set_xscale('auto-gps', epoch=from_gps(gps_start))
        ax.set_xlim([gps_start, gps_start+(24*3600)])
        bns.add_segments_bar(flag)
        bns.add_segments_bar(coincFlag)
        bns.show()
        if save:
            bns.savefig('{:s}/{:s}/bnsrange-{:s}-{:04.0f}-{:02.0f}-{:02.0f}.png'.format(path, ifo, ifo, start_date.year, start_date.month, start_date.day))
    '''
    return Tdict, F, tdict, meanRange
    #return Tdict, F, tdict, meanRange, BNS
# --- END plot_brange()
# 14 =======================================================
# --- START querry_idq()
def query_idq(source, start, end, ifo, prob=False, like=True):
    """
    Queries iDQ local frame files from start to end for given IFO.

INPUTS:
        source = (str) path to local frame files \n
        start  = (float) GPS start time to query \n
        end    = (float) GPS end time of query \n
        ifo    = (str) which ifo to query (e.g. 'L1', 'H1') \n
        prob   = (bool) return p(glitch|aux), gets channel name: {ifo}:IDQ-PGLITCH_OVL_16_2048 \n
        like   = (bool) return log-likelihood, gets channel name: {ifo}:IDQ-LOGLIKE_OVL_16_2048 \n
            
OUTPUTS:
        data  = (gwpy.TimeSeries) \n
        
Required libraries: glob, gwpy, numpy, os
    -> Originally by Ethan Marx (ethan.marx@ligo.org).  This is a modified version of his work.
    """
    
    from gwpy.timeseries import TimeSeries, TimeSeriesDict
    import numpy as np
    import glob
    import os
    
    if start > end:
        raise ValueError('start is after end, invalid!')
    
    # Code to speed up query process:    
    gps_day_start = int(str(start)[0:5])
    gps_day_end = int(str(end)[0:5])

    if gps_day_start == gps_day_end:
        paths = sorted(glob.glob(os.path.join(source, f'{ifo}/START-{gps_day_start}/{ifo.strip("1")}-OVLtimeseries-*.gwf')))
            
    else:
        days = np.arange(gps_day_start, gps_day_end+1, 1)
        paths = [sorted(glob.glob(os.path.join(source, f'{ifo}/START-{day}/{ifo.strip("1")}-OVLtimeseries-*.gwf'))) for day in days]
        paths = list(np.concatenate(paths))
                       
    cut_paths = []
    
    # Eliminate unnecessary paths: 
    for path in paths:
        
        begin = float(path.split('/')[-1].split('-')[2])
        length = float(path.split('/')[-1].split('-')[3].split('.')[0])

        # Frame does not contain requested time:
        if begin > end or (begin + length) < start:
            continue

        # If any part of frame is in requested time:
        elif begin <= end and (begin + length) >= start:
            cut_paths.append(path)
            
    # If data is not there, return nan list:
    if len(cut_paths) == 0:
        return None
    
    # Assemble channel list:
    if prob and like: # get both p(glitch|aux) and log-likelihood
        channels = [f'{ifo}:IDQ-PGLITCH_OVL_16_2048', f'{ifo}:IDQ-LOGLIKE_OVL_16_2048']
    elif not prob: # get just log-likelihood
        channels = f'{ifo}:IDQ-LOGLIKE_OVL_16_2048'
    else: # get just p(glitch|aux)
        channels = f'{ifo}:IDQ-PGLITCH_OVL_16_2048'
    
    # Read in data and crop from start to end: 
    if prob and like:
        data = TimeSeriesDict.read(cut_paths, channels=channels, start=start, end=end, pad=np.nan)
    else:
        data = TimeSeries.read(cut_paths, channels, pad=np.nan)
    data = data.crop(start, end)
                      
    return data
# --- END querry_idq()
# 15 =======================================================
# --- START veto_events()
def veto_events(events, veto_segs_path, ifo, random=False, coincFlag=None, trials=1, flag=True, save=False, verbose=True):
    """
    Manually veto cWB events against a DQ flag for a given day.
    
INPUTS:
    events         = (pd.frame) cWB events data \n
    veto_segs_path = (str) path to segwizard formatted text file \n
    ifo            = (str) 'L1' or 'H1' only
    random         = (bool) switch to generate random vetoes based on real veto segs \n
    coincFlag      = (gypy.SegmentList) coincidence flag (only required if random = True) \n
    trials         = (int) number of MC trails, only if random = True \n
    flag           = (bool) switch to return DQ segs of vetoed events \n
    save           = (bool) switch to save random vetoes in segwizard text file \n
    verbose        = (bool) switch to display status updates \n
    
OUTPUTS:
    EVENTS        = (pd.frame or dict of pd.frame) veoted cWB events data \n
    DQflag        = (gwpy.DataQualityFlag) 1 second segments centered on vetoed events \n
                     N.B. If flag=False, then DQflag returns an empty list. \n
    time          = (array) veto segment durations \n
    
Required libraries: numpy, datetime, gwpy
    """
    
    import numpy as np
    import numpy.random as rng
    import datetime
    from gwpy.time import to_gps, from_gps
    
    # SANITY CHECK - If this is not a random trials run, set the number of trials to one:
    if not random:
        trials = 1
    
    # Import veto segments:
    _, vstart, vstop, dur = np.loadtxt(veto_segs_path, unpack=True)
    
    for q in range(trials):
        
        # Generate random veto coincidence times matching segment durations in true vetoes:
        if random:
            # Determine the day be analyzed:
            start = from_gps(coincFlag.active[0].start)
            gps_start = to_gps(datetime.datetime(start.year,start.month,start.day,0,0,0))

            times = np.array([]) # preallocate array to hold coincidence times
            for k in coincFlag.active: 
                hold = k.start+np.cumsum(np.ones(int(k.end-k.start)))-1 # integer coinc GPS times
                times = np.hstack([times, hold]) # collect coinc times
            breakpoint()
            # Determine what veto segments occured within the day in question:
            less = np.where(vstop < gps_start+(24*3600))[0] # time less than veto end time indices 
            more = np.where(vstart >= gps_start)[0] # time more than veto start time indices 
            results = np.intersect1d(less, more) # vetoed indices = intersection
            results = np.sort(np.unique(results)).astype(int) # make int, sort, remove duplicates
            dur = dur[results] # select corresponding veto segment durations

            # Preallocate arrays to hold random veto segments
            vstart = np.zeros(len(dur))
            vstop = np.zeros(len(dur))
            
            k = 0 # initialize counter
            while k < len(dur): # while less than the number of segments
                N = len(times)-(dur[k]+1) # measure the coincidence time available
                ndx = rng.randint(0, high=N) # pick a random index of coinc time
                rand_veto = times[ndx:ndx+int(dur[k])+1] # select this time to have same duration
                if not sum(np.diff(rand_veto)) > dur[k]: # if the times in seg are continuous
                    vstart[k] = rand_veto[0] # set veto start time
                    vstop[k] = rand_veto[-1] # set veto stop time
                    indices = ndx + np.cumsum(np.ones(int(dur[k])))-1 # select indices of segment
                    times = np.delete(times,indices.astype('int')) # remove this segment from coinc time
                    k += 1 # increment counter
            
            if save: # save random vetoes as a segwizard file...
                from pathlib import Path
                
                # Assemble save path:
                parts = veto_segs_path.split('/') # split at folders
                path = '.' # empty to append to in loop
                # Loop over each directory:
                for z in range(1,len(parts)-1):
                    path = path + '/' + parts[z] # assemble path
                    # need to append to path ??
                rootName = veto_segs_path.split('/')[-1].split('.')[0] # extract file name of veto file w/o ext
                print(path)
                file = f'{path}/rveto/rveto-{q}-{ifo}-{start.year}-{start.month}-{start.day}-{rootName}.txt' # create output file name
                segNum = np.arange(0, len(dur), 1) # create array of segment numbers to write to file
                np.savetxt(file, np.transpose([segNum, vstart, vstop, dur]), fmt='%0.0f', header='seg\tstart\tend\tduration', delimiter='\t') # write random vetoes to file

        # Convert background event times to numpy:
        test = events[f'GPS_{ifo}'].to_numpy()

        # Collect indices of vetoed events:
        results = np.array([]) # initalize collection variable
        #ndx = np.array([])
        time = np.array([])
        for k in range(len(vstart)):
            less = np.where(test < vstop[k])[0] # indices time less than veto end time
            more = np.where(test >= vstart[k])[0] # indices time more than veto start time
            results = np.append(results,np.intersect1d(less, more)) # veto indices = intersection
            if len(np.intersect1d(less, more)) > 0: # there was an event vetoed
                #ndx = np.append(ndx, k)
                time = np.append(time, dur[k])
                # NATHANIEL: you could record which vetos segs are used here
        results = np.sort(np.unique(results)).astype(int) # make int, sort, remove duplicates

        # THE CODE BELOW GENERATES RANDOM VETOS TO MATCH THE *NUMBER OF VETOED EVENTS*, NOT RANDOM COINC TIMES...
        '''
        # If this is a random veto, select the same number of events as if this were from true veto segments:
        if random: # this should replace random_veto_events()
            # Randomly select events to veto:
            ndx = np.cumsum(np.ones(len(events)))-1
            results = np.random.choice(ndx, len(results), replace=False)
            results = np.sort(np.unique(results)).astype(int)
        '''

        # Remove vetoed events:
        keep = events.drop(events.index[results])
        # Assemble dict of frames (if trials > 1) or just a data frame:
        if trials > 1: # more than one trial
            if q == 0: # establish dictionary
                if verbose:
                    print(f'-- Simulating random vetoes: {trials} trials')
                EVENTS = {str(q):keep}
            else: # add key to dict
                EVENTS[str(q)] = keep
        else: # if only a single trial, return data frame
            EVENTS = keep
                    
        # Display effect of vetoes:
        if verbose:
            orig_len = len(events)
            veto_len = len(keep)
            if int(np.floor(np.log10(trials))+1) == 1:
                print(f'-> Trial {q:01.0f} done: {(orig_len-veto_len)*100/orig_len:0.2f}% of events vetoed (N = {orig_len-veto_len}).')
            elif int(np.floor(np.log10(trials))+1) == 2:
                print(f'-> Trial {q:02.0f} done: {(orig_len-veto_len)*100/orig_len:0.2f}% of events vetoed (N = {orig_len-veto_len}).')
            elif int(np.floor(np.log10(trials))+1) == 3:
                print(f'-> Trial {q:03.0f} done: {(orig_len-veto_len)*100/orig_len:0.2f}% of events vetoed (N = {orig_len-veto_len}).')
            elif int(np.floor(np.log10(trials))+1) == 4:
                print(f'-> Trial {q:04.0f} done: {(orig_len-veto_len)*100/orig_len:0.2f}% of events vetoed (N = {orig_len-veto_len}).')
            else:
                print(f'-> Trial {q} done: {(orig_len-veto_len)*100/orig_len:0.2f}% of events vetoed (N = {orig_len-veto_len}).')

        # Create segments of this DQ flag:
        if flag:
            from gwpy.segments import DataQualityFlag

            vetoed = events.iloc[results]
            times = vetoed[f'GPS_{ifo}']

            start = from_gps(np.mean(test)) # what day is it?
            gps_start = to_gps(datetime.datetime(start.year, start.month, start.day, 0, 0, 0))
            gps_stop = gps_start + (24*3600)
            dq = DataQualityFlag('Vetoes', active=zip(times-0.5,times+0.5), known=[[gps_start,gps_stop]])
            # Assemble dict of flags (if trials > 1) or just a flag:
            if trials > 1: # more than one trial
                if q == 0: # establish data frame
                    DQflag = {str(q):dq}
                else: # add key to frame
                    DQflag[str(q)] = dq
            else: # if only a single trial, return flag
                DQflag = dq
        else:
            DQflag = None
        
    return EVENTS, DQflag, time
# --- END veto_events()
# 16 =======================================================
# --- START which_run()
def which_run(gps):
    """
    Returns a string identifying the observing run label during the specified time.  If time is not during a LVK run, the run label is 'unknown'.
    
INPUTS:
    gps     = (float) GPS time to evaluate \n
    
OUTPUTS:
    run     = (string) run label or 'unknown' \n
    run_gps = (dict) dictionary of GPS times for start and end of runs 
              keys are '<run>_start' or '<run>_stop' where run can be O1, O2, O3a, or O3b

Required libraries: datetime, gwpy
    """
    
    import datetime
    from gwpy.time import from_gps, to_gps
    
    # Convert GPS time to datetime format:
    date = from_gps(gps)
    
    # Define O1:
    O1_start = datetime.datetime(2015, 9, 12, 0, 0, 0)
    O1_stop = datetime.datetime(2016, 1, 19, 16, 0, 0)
    
    # Define O2:
    O2_start = datetime.datetime(2016, 11, 30, 16, 0, 0)
    O2_stop = datetime.datetime(2017, 8, 25, 22, 0, 0)
    
    # Define O3a:
    O3a_start = datetime.datetime(2019, 4, 1, 15, 0, 0)
    O3a_stop  = datetime.datetime(2019, 10, 1, 15, 0, 0)
    # Define O3b:
    O3b_start = datetime.datetime(2019, 11, 1, 15, 0, 0)
    O3b_stop  = datetime.datetime(2020, 3, 27, 17, 0, 0)
    
    # Define O4a:
    O4a_start = datetime.datetime(2023, 5, 24, 3, 0, 0)
    O4a_stop = datetime.datetime(2024, 1, 16, 15, 45, 0) # real end date
      
    # Determine appropriate run label for the time given:
    if date <= O1_stop and date >= O1_start:
        run = 'O1'
    elif date <= O2_stop and date >= O2_start:
        run = 'O2'
    elif date <= O3a_stop and date >= O3a_start:
        run = 'O3a'
    elif date <= O3b_stop and date >= O3b_start:
        run = 'O3b'
    elif date <= O4a_stop and date >= O4a_start:
        run = 'O4a'
    else:
        run = 'unknown' 
    
    run_GPS = {'O1_start': to_gps(O1_start), 'O1_stop': to_gps(O1_stop), 'O2_start': to_gps(O2_start), 'O2_stop': to_gps(O2_stop), 'O3a_start': to_gps(O3a_start), 'O3a_stop': to_gps(O3a_stop), 'O3b_start': to_gps(O3b_start), 'O3b_stop': to_gps(O3b_stop), 'O4a_start': to_gps(O4a_start), 'O4a_stop': to_gps(O4a_stop)}
    
    return run, run_GPS
# --- END which_run()
# 17 =======================================================
# --- START cwb_files()
def cwb_files(start, stop, on_off='online'):
    """
    Determine the location of the cWB EVENTS file and background (live) file based on the run label and if the results are ONLINE or OFFLINE.  ** THIS ASSUMES CIT EXECUTION LOCATION! **
    
INPUTS:
    start  = (float) GPS start time \n
    stop   = (float) GPS stop time \n 
    on_off = (str) label for 'online' or 'offline' result \n
    
OUTPUTS:
    run        = (str) run label; see help for cWBtoolbox.which_run() \n
    eventsfile = (str) location of the background EVENTS file \n
    livefile   = (str) location of the background (live) ROOT file \n

Required libraries: cWBtoolbox
    """
    
    import cWBtoolbox as cwb
    from gwpy.time import from_gps
    
    # Determine the run label for this time:
    run, _ = cwb.which_run(start)
    # !!! Error check that the start and stop times have the same label.
    print(f'Run: {run}')
    
    # Get ONLINE cWB file locations based on run label:
    if on_off.lower() == 'online':
        # Load (daily) background EVENTS file:
        if run == 'O3a':
            eventsfile = f'/home/waveburst/public_html/online/O3_LH_BurstLF_ONLINE/POSTPRODUCTION/FOM_daily_{start}-{stop}/plotbin2_cut/data/EVENTS.txt' # background EVENTS file
            livefile = f'/home/waveburst/online/O3_LH_BurstLF_ONLINE/TIME_SHIFTS/POSTPRODUCTION/FOM_daily_{start}-{stop}/merge/live_cwb_bkg.M1.C_Rho.C_bin2_cut.root' # background (live) ROOT file
        elif run == 'O3b':
            eventsfile = f'/home/waveburst/public_html/online/O3b_LH_BurstLF_ONLINE/POSTPRODUCTION/FOM_daily_{start}-{stop}/plotbin2_cut/data/EVENTS.txt' # background EVENTS file
            livefile = f'/home/waveburst/online/O3b_LH_BurstLF_ONLINE/TIME_SHIFTS/POSTPRODUCTION/FOM_daily_{start}-{stop}/merge/live_cwb_bkg.M1.C_Rho.C_bin2_cut.root' # background (live) ROOT file
        elif run == 'O4a':
            date = from_gps(start) # convert to date from gps to use in paths
            eventsfile = f'/home/waveburst.online/public_html/online/O4_L1H1_AllSky_ONLINE/POSTPRODUCTION/FOM_daily_{date.year}-{date.month:02d}-{date.day:02d}/plotxgb/data/EVENTS.txt' # background EVENTS file
            livefile = f'/home/waveburst.online/online/O4_L1H1_AllSky_ONLINE/TIME_SHIFTS/POSTPRODUCTION/FOM_daily_{date.year}-{date.month:02d}-{date.day:02d}/merge/live_cwb_bkg.M1.C_all.XGB_rthr0_all.root' # background (live) ROOT file
        else:
            raise ValueError('Time is not within O3a, O3b, or O4a')

    # Get OFFLINE cWB file locations based on run label:
    elif on_off.lower() == 'offline':
        # Load offline (bulk) results:
        if run == 'O3a':
            eventsfile = f'/home/waveburst/public_html/online/O3_LH_BurstLF_ONLINE/POSTPRODUCTION/FOM_10000_0_0/plotbin2_cut/data/EVENTS.txt' # background EVENTS file
            livefile = f'/home/waveburst/online/O3_LH_BurstLF_ONLINE/TIME_SHIFTS/POSTPRODUCTION/FOM_10000_0_0/merge/live_cwb_bkg.M1.C_Rho.C_bin2_cut.root' # background (live) ROOT file
        elif run == 'O3b':
            eventsfile = f'/home/waveburst/public_html/online/O3b_LH_BurstLF_ONLINE/POSTPRODUCTION/FOM_10000_0_0/plotbin2_cut/data/EVENTS.txt' # background EVENTS file
            livefile = f'/home/waveburst/online/O3b_LH_BurstLF_ONLINE/TIME_SHIFTS/POSTPRODUCTION/FOM_10000_0_0/merge/live_cwb_bkg.M1.C_Rho.C_bin2_cut.root' # background (live) ROOT file
        elif run == 'O4a':
            date = from_gps(start) # convert to date from gps to use in paths
            eventsfile = f'/home/waveburst.online/public_html/online/O4_L1H1_AllSky_ONLINE/POSTPRODUCTION/FOM_daily_{date.year}-{date.month:02d}-{date.day:02d}/plotxgb/data/EVENTS.txt' # background EVENTS file
            livefile = f'/home/waveburst.online/online/O4_L1H1_AllSky_ONLINE/TIME_SHIFTS/POSTPRODUCTION/FOM_daily_{date.year}-{date.month:02d}-{date.day:02d}/merge/live_cwb_bkg.M1.C_all.XGB_rthr0_all.root' # background (live) ROOT file
        else: # O4 offline not implemented yet
            raise ValueError('Time is not within O3a or O3b; O4a can only be used online')
    else:
        raise ValueError("on_off must be 'online' or 'offline'.")
    
    return run, eventsfile, livefile
# --- END cwb_files()
# 18 =======================================================
# --- START load_files()
def load_files(start, stop, det, select, path, on_off='online', verbose=True):
    """
    Load cWB background files (EVENTS, live) and unique background segments (unique).

INPUTS:
    start   = (float) GPS start time \n
    stop    = (float) GPS stop time \n
    det     = (int) numerical code of detector, usually: 0 = L1, 1 = H1, 2 = V1 \n
    select  = (str) pd.DataFrame.query string to select a desired SLAG; e.g. 'slag == 0' 
    path    = (str)
    on_off  = (str) label for 'online' or 'offline' result \n
    verbose = (bool) switch to display status updates \n
    
OUTPUTS:
    events  = (pd.DataFrame) contents of the cWB EVENTS file \n
    live    = (pd.DataFrame) contents of the cWB background livetime ROOT file \n\n
    unique  = (pd.DataFrame) unique start and stop segments with the same columns as the livetime ROOT input \n

Required libraries: cWBtoolbox, datetime, gwpy, os, pandas, pickle
    """
    
    import cWBtoolbox as cwb
    import datetime
    from gwpy.time import from_gps
    import os
    import pickle
    import pandas as pd
    
    # Determine run start date:
    date = from_gps(start)
    
    # Assign ifo string based on det code:
    ifo = cwb.det2ifo(det)
    
    if on_off.lower() == 'online':

        if verbose:
            print(f'*****  Processing date: {from_gps(start)}  *****')

        # Determine location of EVENTS file and background (live) file for this time:
        _, eventsfile, livefile = cwb.cwb_files(start, stop, on_off='online')

        if verbose:
            print('|| Get cWB daily events...')
        if os.path.exists(eventsfile):
            events = cwb.rdEVENTS(eventsfile)
            # Select desired SLAG:
            events = events.query(select)
        else:
            events = cwb.rdEVENTS('empty')
        if verbose:
            print('-> Done (EVENTS)')
            

        # Get background information from live file:
        if verbose:
            print('|| Get cWB background details (livetime ROOT file)...')
        # !!! Use format string and pad single digits with 0 (see below)
        live_out = f'live-{ifo}-{date.year}-{date.month}-{date.day}.csv'
        # !!! Check if live_out exists; import that instead
        live = cwb.rdLiveROOT(livefile, out_file=live_out, SLAG=0)
        if verbose:
            print('-> Done (livetime)')

        # Determine unique segments:
        if verbose:
            print('|| Determining unique segments...')
        # !!! Use format string and pad single digits with 0 (see below)
        uniqueFile = f'unique-{ifo}-{date.year}-{date.month}-{date.day}.txt'
        # !!! Check if uniqueFile exists; import that instead
        unique = cwb.getUnique(live, det, path, outFile=True, file=uniqueFile)
        
        # Don't bother resurrecting failed daily jobs... (i.e. don't use brangeALIVE)
        startHR = 0
        brange = None

        if verbose:
            print('-> Done (unique segments)')
            
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    elif on_off.lower() == 'offline':
        
        # Get needed file locations and names:
        run, eventsfile, livefile = cwb.cwb_files(start, stop, on_off=on_off)
        
        # Determine if this is a resurrection: 
        rerun = False # by default
        startHR = 0 # by default
        brange = None # by default

        # !!! brangeALIVE goes here; get rid of 2 lines above?

        if not startHR == 24:
            # Load cWB livetime tree and EVENTS data:    
            if verbose:
                print(f'|| Load EVENTS file...')
            cachedFile = f'/{path}/EVENTS-{run}.p'
            filename = os.path.basename(cachedFile).split('.')[0]
            
            # Check to if pickle file exists:
            if os.path.exists(filename + '.p'): # pre-processed file exists
                if verbose:
                    print('Read from pickle...')
                events = pd.read_pickle(filename + '.p')
            else: # must read from raw data
                if verbose:
                    print(f'Read EVENTS file from:\n{eventsfile}...')
                # Load background event data:
                events = cwb.rdEVENTS(eventsfile)
                # Select desired SLAG:
                events = events.query(select)
                events.to_pickle(f'/{path}/EVENTS-{run}.p') # save locally to pickle
            if verbose:
                print('-> Done (EVENTS)')

            # Load livetime tree data:
            if verbose:
                print(f'|| Load livetime file...')
            filename = f'/{path}/livetime5_slag0-{run}'
            if os.path.exists(filename + '.p') or os.path.exists(filename + '.csv'): # pre-processed file exists
                # Check to if pickle file exists:
                if os.path.exists(filename + '.p'): # it's a pickle file
                    if verbose:
                        print('--> File containing pre-determined livetime data found.  Reading from that file...')
                        print('Read from pickle...')
                    live = pd.read_pickle(filename + '.p')
                else: # it's a csv file
                    if verbose:
                        print('--> File containing pre-determined livetime data found.  Reading from that file...')
                        print('Read from csv...')
                    live = pd.read_csv(livefile, index_col=0)
                    live.to_pickle(f'{filename}.p') # save to locally to pickle
                if verbose:
                    print('-> Done (livetime)')
            else: # must process from raw data
                if verbose:
                    print('--> File containing pre-determined livetime data NOT found!')
                    print(f'Process livetime file from:\n{livefile}...')
                live = cwb.rdLiveROOT(livefile, path, SLAG=0)
                live.to_pickle(f'{filename}.p') # save to locally to pickle
                          
            # Get unique segments:
            if verbose:
                print('|| Get unique segments...')
            filename = f'{path}/{ifo}/unique-{ifo}-{run}'
            if os.path.exists(filename + '.p') or os.path.exists(filename + '.csv'): # pre-processed data exists
                if verbose:
                    print(f'--> File containing pre-determined unique segments for {ifo} found.  Reading from that file...')
                # Check to if pickle file exists:
                if os.path.exists(f'{filename}.p'): # it's a pickle file
                    if verbose:
                        print('Read from pickle...')
                    unique = pd.read_pickle(f'{filename}.p')
                else: # it's a csv file
                    if verbose:
                        print('Read from csv...')
                    unique = pd.read_csv(uniqueFile, index_col=0)
                    unique.to_pickle(f'{filename}.p') # save to locally to pickle
            else: # must process from raw data
                if verbose:
                    print(f'--> Determining unique segments for {ifo}...')
                unique = cwb.getUnique(live, det, path, outFile=True)
                if verbose:
                    print(f'Saving to {filename}.p')
                unique.to_pickle(f'{filename}.p') # save to locally to pickle
            if verbose:
                print('-> Done (unique segments)')
        else:
            print('-> All data previously processed. Move to plotting...')
                              
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    else:
        raise ValueError("on_off must be 'online' or 'offline'.")
    
    if verbose:
        print('---> Done loading ALL files!')                              
                              
    return events, live, unique, startHR, brange
# --- END load_files()
# 19 =======================================================
# --- START brangeALIVE
def brangeALIVE(ifo, path, date, on_off='offline'):
    """
    Determines if this run has existing data signifying that a previous run crashed.  If true, gathers the existing data and determines the point to restart the run.

INPUTS:
    ifo     = (str) output of cWBtoolbox.det2ifo() \n
    path    = (str) \n
    date    = (datetime) \n
    on_off  = (str) label for 'online' or 'offline' result \n

OUTPUTS:
    startHR = (float) \n
    brange  = (dict or None) \n
        .R \n
        .flag \n
        .coincFlag \n
        .f \n

Required libraries: numpy, os, pickle
    """
    import numpy as np
    import os
    import pickle
    
    # Determine if this is a resurrection: 
    rerun = False # by default
                              
    if on_off.lower() == 'online': # ONLINE code
        # Determine file name:
        fname = '{:s}/{:s}/brange-ON-{:s}-{:04.0f}-{:02.0f}-{:02.0f}.p'.format(path, ifo, ifo, date.year, date.month, date.day) 
        # If this file name exists, then this is a rerun:
        if os.path.exists(fname):
                rerun = True
            
    elif on_off.lower() == 'offline': # OFFLINE code
        # Determine file name:
        fname = '{:s}/{:s}/brange-OFF-{:s}-{:04.0f}-{:02.0f}-{:02.0f}.p'.format(path, ifo, ifo, date.year, date.month, date.day) # !!! Add on/offline info to file name
        # If this file name exists, then this is a rerun:
        if os.path.exists(fname):
                rerun = True
        
    # Resurrect run if needed:
    if rerun:
        # Determine backup file name:  
        if on_off.lower() == 'online': # ONLINE bup file name
            bupname = '{:s}/{:s}/brange-ON-{:s}-{:s}-{:04.0f}-{:02.0f}-{:02.0f}-bup.p'.format(path, ifo, ifo, on_off.lower(), date.year, date.month, date.day) 
        elif on_off.lower() == 'offline': # OFFLINE bup file name
            bupname = '{:s}/{:s}/brange-OFF-{:s}-{:s}-{:04.0f}-{:02.0f}-{:02.0f}-bup.p'.format(path, ifo, ifo, on_off.lower(), date.year, date.month, date.day)
                              
        if verbose:
            print(f'|| Restart run from {fname}...')
            print(f'-- Backup file to {bupname}...')
        # Backup original file (just in case!):
        cmd = f'cp {fname} {bupname}'
        os.system(cmd)
        if verbose:
            print(f'-> Backup complete.')
            print(f'-- Load data...')
        # Load pickle file:
        burstRange = pickle.load(open(fname, 'rb'))
        # Assign variables out of burstRange dict:
        R = burstRange['R']
        flag = burstRange['flag']
        coincFlag = burstRange['coincFlag']
        f = burstRange['f']
        # Calculate what hour to restart calculation:
        startHR = int(np.shape(R)[0]/6) # determine the number of hours from number of rows in range array
        if verbose:
            print(f'-> Data loaded.  Restart at hour {startHR}.')
    else:
        # If this is not a rerun, then start the calculation at hour 0:
        startHR = 0
        burstrange = None

    return startHR, burstrange

# --- END brangeALIVE
# 20 =======================================================
# --- START veto_perdiem
def veto_perdiem(ifo, select, veto_segs_path):
    """
    Determines the percentage of vetoed background events for each day in the range of veto segments.
    
INPUTS:
    ifo            = (str) output of cWBtoolbox.det2ifo() \n
    select         = (str) pd.DataFrame.query string to select a desired SLAG; e.g. 'slag == 0' 
    veto_segs_path = (str) path to segwizard formatted text file \n
    
OUTPUTS:
    NO FUNCTION OUTPUTS (results are displayed on screen)
    
Required libraries: cWBtoolbox, datetime, gwpy, numpy, os
    """

    import cWBtoolbox as cwb
    import datetime
    from gwpy.time import to_gps, from_gps
    import numpy as np
    import os
    
    # Import veto segments:
    _, vstart, vstop, _ = np.loadtxt(veto_segs_path, unpack=True)

    # Determine range of days veto is applied over:
    start = from_gps(min(vstart))
    stop = from_gps(max(vstop))
    dur = stop-start
    days = int(np.ceil(dur.total_seconds()/(24*3600)))
    
    # Determine first day to start:
    gps_start = to_gps(datetime.datetime(start.year,start.month,start.day,0,0,0))

    # Test each day:
    for k in range(days):
        gps_start = gps_start+(24*3600) # !!! Am I measureing the first day?
        # !!!  Do I need the next 4 lines?
        less = np.where(vstop < gps_start+(24*3600))[0] # time less than veto end time indices 
        more = np.where(vstart >= gps_start)[0] # time more than veto start time indices 
        results = np.intersect1d(less, more) # vetoed indices = intersection
        results = np.sort(np.unique(results)).astype(int) # make int, sort, remove duplicates

        # Get daily EVENTS file path:
        run, eventsfile, _ = cwb.cwb_files(gps_start, gps_start+(24*3600), on_off='online')
        if os.path.exists(eventsfile): # if no path, there wasn't data that day = skip!
            events = cwb.rdEVENTS(eventsfile) # get events
            events = events.query(select) # select desired SLAG
            vetoed, _, _ = cwb.veto_events(events, veto_segs_path, ifo, flag=False, verbose=False) # veto events

            # Display effects of veto:
            date = from_gps(gps_start)
            print(f'{date.year}-{date.month}-{date.day}:\t{(len(events)-len(vetoed))*100/len(events):0.2f}%')
        else: # report that there was no data
            date = from_gps(gps_start)
            print(f'{date.year}-{date.month}-{date.day}:\tN/A')
# --- END veto_perdiem
# 21 =======================================================
# --- START txt2segs
def txt2segs(txt):
    """
    Converts a text file of segments to a gwpy.SegmentList.
    
INPUTS:
    txt  = (str) path to segwizard formatted text file

OUTPUTS:
    segs = (gypy.SegmentList) segment list formatted for gwpy

Required libraries: gwpy
    """

    from gwpy.segments import SegmentList
    
    # Convert contents of file to SegmentList:
    segs = SegmentList.read(txt)
    
    return segs
# --- END txt2segs
# 22 =======================================================
# --- START segs2txt
def segs2txt(segs, txt):
    """
    Converts a gwpy.SegmentList to a text file of segments.
    
INPUTS:
    segs = (gypy.SegmentList) segment list formatted for gwpy
    txt  = (str) path to segwizard formatted text file
    
OUTPUTS:
    none - a segwizard file is writen to the path indicated in txt
    
Required libraries: gwpy

"""

    from gwpy.segments import SegmentList
    
    # Convert contents of file to SegmentList:
    segs.write(txt)
    
# --- END
# 23 =======================================================
# --- START txt2flag
def txt2flag(txt, flag_name, gps_start, gps_stop):
    """
    Converts a text file of segments to a gwpy.DataQualityflag.
    
INPUTS:
    txt       = (str) path to segwizard formatted text file
    flag_name = (str) flag label
    gps_start = (float) select segments after this GPS time
    gps_stop  = (float) select segments before this GPS time

OUTPUTS:
    flag      = (gwpy.DataQualityFlag) DQ flags formatted for gwpy

Required libraries: datetime, gwpy, numpy
    """

    import numpy as np
    import datetime
    from gwpy.time import to_gps, from_gps
    from gwpy.segments import DataQualityFlag
    
    # Import segments from segwizard formatted text file:
    _, fstart, fstop, _ = np.loadtxt(txt, unpack=True)

    # Find segments withing the [gps_start, gps_stop) time range 
    less = np.where(fstop < gps_stop)[0] # time less than veto end time indices 
    more = np.where(fstart >= gps_start)[0] # time more than veto start time indices 
    results = np.intersect1d(less, more) # vetoed indices = intersection
    results = np.sort(np.unique(results)).astype(int) # make int, sort, remove duplicates
    
    # Only select the found segments:
    fstart = fstart[results]
    fstop = fstop[results]
    
    # Determine range of days imported segments span:
    #start = from_gps(min(fstart))
    #stop = from_gps(max(fstop))
    #dur = stop-start
    #days = int(np.ceil(dur.total_seconds()/(24*3600)))

    # Determine day to start:
    #date_start = datetime.datetime(start.year,start.month,start.day,0,0,0)
    #gps_start = to_gps(date_start)
    
    # Determine day to stop:
    #gps_stop = to_gps(date_start + datetime.timedelta(days=days))

    # Create flag:
    flag = DataQualityFlag(flag_name, active=zip(fstart,fstop), known=[[gps_start,gps_stop]])
    
    return flag
# --- END txt2flag
# 24 =======================================================
# --- START mkfname()
def mkfname(path, ifo, trials, start_date, veto_segs_path, random_veto, on_off):
    """
    Format the path and file name to hold burst range results (final and intermediate).
    
INPUTS:
    path           = (str) home directory path for burst range results and veto files\n
    ifo            = (str) 'L1' or 'H1' only
    start_date     = (datetime) date of day to being analyzed \n
    veto_segs_path = (str) path to segwizard formatted text file \n
    random_veto    = (bool) switch to use random vetoes comparible to those found in veto_segs_path \n
    on_off         = (str) label for 'online' or 'offline' result \n
    
OUTPUTS:
    fname          = (str) file path and name for burst range results \n

Required libraries: NONE
    """
    
    if on_off.lower() == 'online': # this run what made on cWB daily results
        if veto_segs_path: 
            if random_veto: # this was random vetos
                fname = '{:s}/{:s}/brange-daily-rvetoed-{:d}-{:s}-{:04.0f}-{:02.0f}-{:02.0f}.p'.format(path, ifo, trials, ifo, start_date.year, start_date.month, start_date.day)
            else: # this was veto on a specific DQ flag
                fname = '{:s}/{:s}/brange-daily-vetoed-{:s}-{:04.0f}-{:02.0f}-{:02.0f}.p'.format(path, ifo, ifo, start_date.year, start_date.month, start_date.day)
        else: # no vetos applied
            fname = '{:s}/{:s}/brange-daily-{:s}-{:04.0f}-{:02.0f}-{:02.0f}.p'.format(path, ifo, ifo, start_date.year, start_date.month, start_date.day)
    else: # this run was made on cWB bulk results
        if veto_segs_path: 
            if random_veto: # this was random vetos
                fname = '{:s}/{:s}/brange-rvetoed-{:s}-{:04.0f}-{:02.0f}-{:02.0f}.p'.format(path, ifo, ifo, start_date.year, start_date.month, start_date.day)
            else: # this was veto on a specific DQ flag
                fname = '{:s}/{:s}/brange-vetoed-{:s}-{:04.0f}-{:02.0f}-{:02.0f}.p'.format(path, ifo, ifo, start_date.year, start_date.month, start_date.day)
        else: # no vetos applied
            fname = '{:s}/{:s}/brange-{:s}-{:04.0f}-{:02.0f}-{:02.0f}.p'.format(path, ifo, ifo, start_date.year, start_date.month, start_date.day)
    return fname
# -- END mkfname()
# 25 =======================================================    
# --- START integrate_exp_fit()
def integrate_exp_fit(a, b, lower):
    '''
    Definite integral of exp(a*x)*exp(b) between a finite x-value and infinity. N.B. This solution is for all real values of a < 0.
    
INPUTS:
    a     = fit parameter, coefficient of the x-data \n
    b     = fit parameter, constant \n
    lower = lower bound of integration \n

OUTPUTS:
    I     = definite inegral of exp(a*x)*exp(b) between val and infinity \n

Required libraries: numpy
    '''
    
    import numpy as np 
    
    if a >= 0: # check to make sure a < 0
        print('ERROR: int_exp_fit only accepts values of a < 0!')
        I = float('nan') # if a >= 0, return NaN
    else:
        I = (np.exp((a*lower)+b))/a # calculate analytical result
        
    return I
# -- END integrrate_exp_fit()
# 26 =======================================================
# --- START getSNR2
def getSNR2(hstart, hstop, thresh, det, live, events, unique, ref_SNR, thresh_on='rate', plot=False, study=False, verbose=True):
    """
    Get the SNR at a specified background false alarm rate.  This is calulated over a 1 hour period.

INPUTS:
    hstart     = \n
    hstop      = \n
    thresh     = \n
    det        = (int) \n
    live       = (pd.DataFrame) \n
    events     = (pd.DataFrame) \n
    unique     = \n
    thresh_on  = (str) 'prob' or 'rate' : apply threshold to false alarm probability or false alarm rate \n
    plot       = (bool) \n
    study      = (bool) switch to save intermediate data to thresh_temp_bkgnd.p for threshold studies \n
    verbose    = (bool) switch to display status updates \n

OUTPUTS:
    snr        = \n

Required libraries: cwBtoolbox, numpy, pandas, pickle
    """
    
    import pdb # pdb.set_trace()  <- to insert breakpooint
    import cWBtoolbox as cwb
    import numpy as np
    import pandas as pd
    import pickle
    
    # Get background live time:
    bkgndtime = live.query("start" + str(det) + " >= " + str(hstart) + " and stop" + str(det) + " <= " + str(hstop)).live.sum()
    print(f'-> background time = {bkgndtime}')
    # Find unique segments in background:
    hold = unique.query('start >= ' + str(hstart) + ' and stop <= ' + str(hstop))
    # Find zero-lag live time:
    zerolag = (hold.stop-hold.start).sum() # should I be using flags to query zero-lag time? -No! Seg length was 600 or bust.
    print(f'-> zero-lag time = {zerolag}')
    
    if type(events) is not dict:
        trials = 1
    else:
        trials = len(events.keys())
    
    snr = np.zeros(trials) # preallocate snr array
    for q in range(trials):
        if type(events) is dict:
            EVENTS = events[str(q)]
        else:
            EVENTS = events
        # Get events in this hour:
        data = pd.DataFrame()
        for m in range(len(hold)): # !!! MIKE:  I think the issue may be btwn here and line 762.
            # Get events in this hour:
            if det == 0:
                temp = EVENTS.query('GPS_L1 >= ' + str(hold.iloc[m].start) + " and GPS_L1 <" + str(hold.iloc[m].stop))
            elif det == 1:
                temp = EVENTS.query('GPS_H1 >= ' + str(hold.iloc[m].start) + " and GPS_H1 <" + str(hold.iloc[m].stop))
            else:
                print('ERROR: det must be 0 or 1')
            data = pd.concat([data, temp], axis=0).reset_index(drop=True)
        if data.empty:
            if verbose and q == trials-1:
                print('-> no events')
            snr[q] = float('nan') # return snr = NaN if there are no events in this hour
        else:
            if det == 0:
                rate, prob, bins = cwb.plotBkgnd(data.SNR_L1, 'snr', bkgndtime, zerolag, nbins=1000, ifo='L1', plot=plot)
            elif det == 1:
                rate, prob, bins = cwb.plotBkgnd(data.SNR_H1, 'snr', bkgndtime, zerolag, nbins=1000, ifo='H1', plot=plot)
            else:
                print('ERROR: det must be 0 or 1')
            
            if study:
                # This is needed for threshold studies:
                background = {'rate':rate, 'prob':prob, 'bins':bins, 'zerolag':zerolag, 'bkgndtime':bkgndtime}
                pickle.dump(background, open('thresh_temp_bkgnd.p', 'wb'))

            # Find indices where rate/prob is greater than the threshold:
            if thresh_on.lower() == 'rate':
                # Determine where rate >= rateThresh:
                ndx = np.where(rate[1,:] >= thresh)
            elif thresh_on.lower() == 'prob': 
                # Determine where prob >= thresh:
                ndx = np.where(prob[1,:] >= thresh)
            else:
                print('getSNR(): thresh_on must == "prob" or "rate"!')

            # Did thresh intersect with the background distribution?
            if len(ndx[0]) > 0: # The distribution crosses the threshold value:
                # Determine smallest SNR with rate >= rateThresh:
                snr[int(q)] = bins[ndx[0][-1]+1]
                # if snr[int(q)] > refSNR: # Red flag condition!
                #    pdb.set_trace() # this means pause and let me poke around the memory
                # !!!  MIKE: Poke around, may want to run something like this:
                # rate, prob, bins = cwb.plotBkgnd(data.SNR_H1, 'snr', bkgndtime, zerolag, nbins=1000, ifo='H1', plot=True)
                if verbose and q == trials-1:
                    print(f'-> SNR = {snr}')
            else: # Thresh doesn't cross distribution, don't calculate:
                snr[int(q)] = float('nan')
                if verbose: #and q == trials-1:
                    print(f'-> No SNR: thresh didn\'t intersect with background distribution!')
        if snr > ref_SNR:
            print(f'ALERT: SNR {snr} is less than reference SNR {ref_SNR}')
            
    return snr
# --- END getSNR2()
# ======================================================= 
# --- END cWBtoolbox