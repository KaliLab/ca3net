#!/usr/bin/python
# -*- coding: utf8 -*-
"""
helper file to extract dynamic features: checking replay interval by ISI, computing AC and PSD of population rate
authors: András Ecker, Bence Bagi, Eszter Vértes, Szabolcs Káli last update: 11.2017
"""

import pickle
import numpy as np
from scipy import signal, misc
import brian2.monitors.spikemonitor


def preprocess_monitors(sm, prm, calc_ISI=True):
    """
    preprocess Brian's SpikeMonitor and PopulationRateMonitor data for further analysis and plotting
    :param sm: Brian2 SpikeMonitor
    :param prm: PopulationRateMonitor
    :param calc_ISI: calculate ISIs or not
    :return spikeTimes, spikingNeurons: used for raster plots
            rate: firing rate of the population (hard coded to use 1*ms bins!)
            ISIhist and ISI_bin_edges: bin heights and edges of the histogram of the ISI of the population
    """

    # get data from monitors
    spikeTimes = np.array(sm.t_) * 1000.  # *1000 ms conversion
    spikingNeurons = np.array(sm.i_)     
    tmp_spikeTimes = sm.spike_trains().items() 
    rate = np.array(prm.rate_).reshape(-1, 10).mean(axis=1)

    if calc_ISI:
        ISIs = np.hstack([np.diff(spikes_i*1000) for i, spikes_i in tmp_spikeTimes])  # *1000 ms conversion
        ISIhist, bin_edges = np.histogram(ISIs, bins=20, range=(0,1000))

        return spikeTimes, spikingNeurons, rate, ISIhist, bin_edges
    else:
        return spikeTimes, spikingNeurons, rate


def preprocess_spikes(spiketimes, N_norm, calc_ISI=True):
    """
    preprocess Brian's SpikeMonitor data for further analysis and plotting
    -> more general (solves it without rate monitor), but slower version of preprocess_monitors()
    :param spiketimes: dictionary with keys as neuron IDs and spike time arrays (produced by Brian(1&2) SpikeMonitor)
    :param N_norm: normaliation factor for rate
    :return spikeTimes, spikingNeurons: used for raster plots
            rate: firing rate of the population (hard coded to use 1*ms bins!)
            ISIs: inter spike intervals (used for replay detection and plotting)
    """

    spikeTimes = []
    spikingNeurons = []
    rate = np.zeros((10000)); fs = 1000. # hard coded for 10000ms and 1ms bins
    if calc_ISI:
        ISIs = []
    for i, spikes_i in spiketimes.items():  # the order doesn't really matter...
        # create arrays for plotting
        nrn = i * np.ones_like(spikes_i)
        spikingNeurons = np.hstack([spikingNeurons, nrn])
        spikeTimes = np.hstack([spikeTimes, spikes_i*1000])  # *1000 ms conversion
        # calculate InterSpikeIntervals
        if calc_ISI:
            if len(spikes_i) >= 2:
                isi = np.diff(spikes_i*1000) # *1000 ms conversion
                ISIs = np.hstack([ISIs, isi])
        # updating firing rate (based on spikes from 1 neuron)
        spike_ids = (spikes_i*1000).astype(int)  # create indexing array (*1000 ms conversion)
        rate[spike_ids] += 1
        
    # normalize rate
    rate = rate/(N_norm*(1./fs))

    if calc_ISI:
        return spikeTimes, spikingNeurons, rate, ISIs
    else:
        return spikeTimes, spikingNeurons, rate


def replay(isi, th=0.7):
    """
    Decides if there is a replay or not:
    searches for the max # of spikes (and plus one bin one left- and right side) and checks again % threshold
    :param isi: Inter Spike Intervals (of the pyr. pop.)
    :param th: threshold for spike count in the highest and 2 nearest bins (set to 70%)
    :return avgReplayInterval: counted average replay interval
    """

    binsROI = isi
    binMeans = np.linspace(175, 825, 14)  # hard coded...
    maxInd = np.argmax(binsROI)
    
    if 1 <= maxInd <= len(binsROI) - 2:
        bins3 = binsROI[maxInd-1:maxInd+2]
        tmp = binsROI[maxInd-1]*binMeans[maxInd-1] + binsROI[maxInd]*binMeans[maxInd] + binsROI[maxInd+1]*binMeans[maxInd+1]
        avgReplayInterval = tmp / (binsROI[maxInd-1] + binsROI[maxInd] + binsROI[maxInd+1])
    else:
        bins3 = []

    replay_ = np.nan
    if sum(int(i) for i in binsROI) * th < sum(int(i) for i in bins3):
        replay_ = avgReplayInterval

    return replay_


def _autocorrelation(x):
    """
    Computes the autocorrelation/serial correlation of a time series (to find repeating patterns)
    R(\tau) = \frac{E[(X_t - \mu)(X_{t+\tau} - \mu)]}{\sigma^2}
    :param x: time series
    :return: autocorrelation
    """

    meanx = np.mean(x)
    xUb = x - meanx
    xVar = np.sum(xUb**2)
    xAC = np.correlate(xUb, xUb, mode='same') / xVar  # cross correlation of xUb and xUb -> autocorrelation

    return xAC[len(xAC)/2:]


# not used in the final version    
def _filter_rate(rate, fs, cut=2., order=5):
    """
    highpass filters rate (above 2Hz) with a Butterfly filter (just for PSD)
    :param rate: firing rate
    :param fs: sampling freq
    :param cut, order: cutting freq. and order of the filter
    :return filt_rate: high pass filtered rate
    """
    
    nyq = fs * 0.5
    # see more: https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.signal.butter.html
    b, a = signal.butter(order, cut/nyq, btype="highpass")
    #w, h = signal.freqz(b, a, worN=2000)
    #plt.plot((nyq/np.pi)*w, abs(h))
    filt_rate = signal.filtfilt(b, a, rate)
    
    return filt_rate


def analyse_rate(rate, fs=1000, TFR=False):
    """
    Basic analysis of firing rate: autocorrelatio, PSD, TFR (wavelet analysis)
    :param rate: firing rate of the neuron population
    :param fs: sampling frequency (for the spectral analysis)
    :param TFR: bool - calculate time freq. repr. (using wavelet analysis) - this might take some time and RAM...
    :return: rM, rAC: mean rate, autocorrelation of the rate
             maxAC, tMaxAC: maximum autocorrelation, time interval of maxAC
             f, Pxx: sample frequencies and power spectral density (results of PSD analysis)
             tfr, t, freqs: calculated TFR matrix and time points and frequencies used to calculate it      
    """
    
    rM = np.mean(rate)
    
    # analyse autocorrelation function
    rAC = _autocorrelation(rate)
    # random hard coded values by Eszter:(
    maxAC = rAC[1:].max()
    tMaxAC = rAC[1:].argmax()+1
    
    # highpass filter rate before calculating PSD
    #rate = _filter_rate(rate, fs)
    
    # get PSD - see more: http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.signal.welch.html
    f, Pxx = signal.welch(rate, fs, window="hamming", nperseg=512, scaling="spectrum")
    
    if not TFR:
        return rM, rAC, maxAC, tMaxAC, f, Pxx
    else:
        from tftb.processing import Scalogram
        
        tfr, t, freqs, _ = Scalogram(rate, fmin=0.02, fmax=0.45, nvoices=512).run()
        
        return rM, rAC, maxAC, tMaxAC, f, Pxx, tfr, t, freqs


def _fisher(Pxx):
    """
    Performs Fisher g-test on PSD (see Fisher 1929: http://www.jstor.org/stable/95247?seq=1#page_scan_tab_contents)
    :param Pxx: spectrum (returned by scipy.signal.welch)
    :param p: significance threshold
    :return pVal: p-value
    """

    fisherG = Pxx.max() / np.sum(Pxx)

    N = len(Pxx)
    upper = int(np.floor(1 / fisherG))
    I = []
    for i in range(1, upper):
        Nchoosei = misc.comb(N, i)
        I.append(np.power(-1, i-1) * Nchoosei * np.power((1-i*fisherG), N-1))
    pVal = np.sum(I)
    
    return pVal


def ripple(rAC, f, Pxx, p=0.01):
    """
    Decides if there is a significant high freq. ripple oscillation
    by applying Fisher g-test (on the spectrum)
    :param rAC: auto correlation function of rate see `analyse_rate()`
    :param Pxx, f: calculated power spectrum of the neural activity (and frequencies used to calculate it) see `analyse_rate()`
    :param p: significance threshold for Fisher g-test
    :return: maxAC, tMaxAC: maximum autocorrelation in ripple range, time interval of maxACR
             avgRippleF, rippleP: average frequency and power of the oscillation
    """

    # random hard coded values by Eszter:(
    maxACR = rAC[3:9].max()
    tMaxACR = rAC[3:9].argmax()+3
    
    # get ripple freq
    f = np.asarray(f)
    PxxRipple = Pxx[np.where((160 < f) & (f < 230))]
    
    # apply Fisher g-test
    pVal = _fisher(PxxRipple)
    avgRippleF = f[np.where(160 < f)[0][0] + PxxRipple.argmax()] if pVal < p else np.nan
        
    # get ripple power
    rippleP = (sum(PxxRipple) / sum(Pxx)) * 100

    return maxACR, tMaxACR, avgRippleF, rippleP


def gamma(f, Pxx, p=0.01):
    """
    Decides if there is a significant gamma freq. oscillation
    by applying Fisher g-test (on the spectrum)
    :param Pxx, f: calculated power spectrum of the neural activity (and frequencies used to calculate it) see `analyse_rate()`
    :param p: significance threshold for Fisher g-test
    :return: avgGammaF, gammaP: average frequency and power of the oscillation
    """

    # get gamma freq
    f = np.asarray(f)
    PxxGamma = Pxx[np.where((30 < f) & (f < 100))]
    
    # apply Fisher g-test
    pVal = _fisher(PxxGamma)
    avgGammaF = f[np.where(30 < f)[0][0] + PxxGamma.argmax()] if pVal < p else np.nan
    
    # get gamma power
    gammaP = (sum(PxxGamma) / sum(Pxx)) * 100

    return avgGammaF, gammaP
    
    
def load_Wee(fName):  # this function does not belong to here ... (should be eg. in helpers)
    """dummy function, just to make python close the file and clear the memory"""
    
    with open(fName, "rb") as f:
        Wee = pickle.load(f) * 1e9
    return Wee
    
    
    
