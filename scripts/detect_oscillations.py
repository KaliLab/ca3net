# -*- coding: utf8 -*-
"""
helper file to extract dynamic features: checking replay interval by ISI, computing AC and PSD of population rate
authors: András Ecker, Bence Bagi, Eszter Vértes, Szabolcs Káli last update: 11.2017
"""

import pickle
import numpy as np
from scipy import signal, misc


Erev_exc = 0.0
Erev_inh = -70.0
volume_cond = 1 / 3.54


def preprocess_monitors(SM, RM, calc_ISI=True):
    """
    preprocess Brian's SpikeMonitor and PopulationRateMonitor data for further analysis and plotting
    :param SM: Brian2 SpikeMonitor
    :param RM: Brian2 PopulationRateMonitor
    :param calc_ISI: bool for calculating ISIs
    :return spike_times, spiking_neurons: used for raster plots
            rate: firing rate of the population (hard coded to use 1*ms bins!)
            ISI_hist and ISI_bin_edges: bin heights and edges of the histogram of the ISI of the population
    """

    spike_times = np.array(SM.t_) * 1000.  # *1000 ms conversion
    spiking_neurons = np.array(SM.i_)     
    tmp_spike_times = SM.spike_trains().items() 
    rate = np.array(RM.rate_).reshape(-1, 10).mean(axis=1)

    if calc_ISI:
        ISIs = np.hstack([np.diff(spikes_i*1000) for i, spikes_i in tmp_spike_times])  # *1000 ms conversion
        ISI_hist, bin_edges = np.histogram(ISIs, bins=20, range=(0,1000))

        return spike_times, spiking_neurons, rate, ISI_hist, bin_edges
    else:
        return spike_times, spiking_neurons, rate


def replay(ISI, th=0.7):
    """
    Decides if there is a replay or not:
    searches for the max # of spikes (and plus one bin one left- and right side) and checks again % threshold
    :param ISI: Inter Spike Intervals (of the pyr. pop.)
    :param th: threshold for spike count in the highest and 2 nearest bins (set to 70%)
    :return avg_replay_interval: counted average replay interval
    """

    bins_ROI = ISI
    bin_means = np.linspace(175, 825, 14)  # hard coded...
    max_ID = np.argmax(bins_ROI)
    
    if 1 <= max_ID <= len(bins_ROI) - 2:
        bins_3 = bins_ROI[max_ID-1:max_ID+2]
        tmp = bins_ROI[max_ID-1]*bin_means[max_ID-1] + bins_ROI[max_ID]*bin_means[max_ID] + bins_ROI[max_ID+1]*bin_means[max_ID+1]
        avg_replay_interval = tmp / (bins_ROI[max_ID-1] + bins_ROI[max_ID] + bins_ROI[max_ID+1])
    else:
        bins_3 = []

    replay = np.nan
    if sum(int(i) for i in bins_ROI) * th < sum(int(i) for i in bins_3):
        replay = avg_replay_interval

    return replay


def _autocorrelation(x):
    """
    Computes the autocorrelation/serial correlation of a time series (to find repeating patterns)
    R(\tau) = \frac{E[(X_t - \mu)(X_{t+\tau} - \mu)]}{\sigma^2}
    :param x: time series
    :return: autocorrelation
    """

    x_var = np.var(x)
    x = x - np.mean(x)
    autocorrelation = np.correlate(x, x, mode="same") / x_var

    return autocorrelation[len(autocorrelation)/2:]


def analyse_rate(rate, fs=1000., TFR=False):
    """
    Basic analysis of firing rate: autocorrelatio, PSD, TFR (wavelet analysis)
    :param rate: firing rate of the neuron population
    :param fs: sampling frequency (for the spectral analysis)
    :param TFR: bool - calculate time freq. repr. (using wavelet analysis)
    :return: mean_rate, rate_ac: mean rate, autocorrelation of the rate
             max_ac, t_max_ac: maximum autocorrelation, time interval of maxAC
             f, Pxx: sample frequencies and power spectral density (results of PSD analysis)
             coefs, freqs: coefficients from wavelet transform and frequencies used
    """
    
    mean_rate = np.mean(rate)
    
    rate_ac = _autocorrelation(rate)
    max_ac = rate_ac[1:].max()
    t_max_ac = rate_ac[1:].argmax()+1
    
    f, Pxx = signal.welch(rate, fs, nperseg=512)
    
    if not TFR:
        return mean_rate, rate_ac, max_ac, t_max_ac, f, Pxx
    else:
        import pywt        
        #pywt.scale2frequency("morl", scale) / (1/fs)
        scales = np.linspace(3.5, 25, 800)        
        coefs, freqs = pywt.cwt(rate[4000:6000], scales, "morl", 1/fs)  # use only the middle 2s
        
        return mean_rate, rate_ac, max_ac, t_max_ac, f, Pxx, coefs, freqs


def _fisher(Pxx):
    """
    Performs Fisher g-test on PSD (see Fisher 1929: http://www.jstor.org/stable/95247?seq=1#page_scan_tab_contents)
    :param Pxx: spectrum (returned by scipy.signal.welch)
    :return p_val: p-value
    """

    fisher_g = Pxx.max() / np.sum(Pxx)
    n = len(Pxx); upper_lim = int(np.floor(1. / fisher_g))
    p_val = np.sum([np.power(-1, i-1) * misc.comb(n, i) * np.power((1-i*fisher_g), n-1) for i in range(1, upper_lim)])
    
    return p_val


def ripple(rate_ac, f, Pxx, p_th=0.05):
    """
    Decides if there is a significant high freq. ripple oscillation by applying Fisher g-test (on the spectrum)
    :param rate_ac: auto correlation function of rate see `analyse_rate()`
    :param Pxx, f: calculated power spectrum of the neural activity (and frequencies used to calculate it) see `analyse_rate()`
    :param p_th: significance threshold for Fisher g-test
    :return: max_ac_ripple, t_max_ac_ripple: maximum autocorrelation in ripple range, time interval of maxACR
             avg_ripple_freq, ripple_power: average frequency and power of ripple band oscillation
    """

    max_ac_ripple = rate_ac[3:9].max()  # random hard coded values by Eszter...
    t_max_ac_ripple = rate_ac[3:9].argmax()+3
    
    f = np.asarray(f)
    Pxx_ripple = Pxx[np.where((150 < f) & (f < 220))]
    
    p_val = _fisher(Pxx_ripple)
    avg_ripple_freq = f[np.where(150 < f)[0][0] + Pxx_ripple.argmax()] if p_val < p_th else np.nan
        
    ripple_power = (sum(Pxx_ripple) / sum(Pxx)) * 100

    return max_ac_ripple, t_max_ac_ripple, avg_ripple_freq, ripple_power


def gamma(f, Pxx, p_th=0.05):
    """
    Decides if there is a significant gamma freq. oscillation by applying Fisher g-test (on the spectrum)
    :param Pxx, f: calculated power spectrum of the neural activity (and frequencies used to calculate it) see `analyse_rate()`
    :param p_th: significance threshold for Fisher g-test
    :return: avg_gamma_freq, gamma_power: average frequency and power of the oscillation
    """

    f = np.asarray(f)
    Pxx_gamma = Pxx[np.where((30 < f) & (f < 100))]
    
    p_val = _fisher(Pxx_gamma)
    avg_gamma_freq = f[np.where(30 < f)[0][0] + Pxx_gamma.argmax()] if p_val < p_th else np.nan

    gamma_power = (sum(Pxx_gamma) / sum(Pxx)) * 100

    return avg_gamma_freq, gamma_power
    
    
def _estimate_LFP(StateM, subset):
    """
    Estimates LFP by summing synaptic currents to PCs (assuming that all neurons are at equal distance (1 um) from the electrode)
    :param StateM: Brian2 StateMonitor object (of the PC population)
    :param subset: IDs of the recorded neurons
    :return: t, LFP: estimated LFP (in uV) and corresponding time points (in ms)
    """

    t = StateM.t_ * 1000.  # *1000 ms convertion
    LFP = np.zeros_like(t)
    
    for i in subset:
        v = StateM[i].vm*1000 # *1000 mV conversion
        g_exc = StateM[i].g_ampa + StateM[i].g_ampaMF  # this is already in nS (see *z in the equations)
        i_exc = g_exc * (v - Erev_exc * np.ones_like(v))  # pA
        g_inh = StateM[i].g_gaba
        i_inh = g_inh * (v - Erev_inh * np.ones_like(v))  # pA
        LFP += -(i_exc + i_inh)  # (this is still in pA)
        
    LFP *= 1 / (4 * np.pi * volume_cond * 1e6)  # uV (*1e-6 um conversion)
    
    return t, LFP


def _filter_LFP(LFP, fs, cut=300.):
    """
    Low pass filters LFP (3rd order Butterworth filter)
    :param LFP: estimated LFP (see `_calculate_LFP()`)
    :param fs: sampling frequency
    :param cut: cut off frequency
    """
    
    b, a = signal.butter(3, cut/(fs/2.), btype="lowpass")
    
    return signal.filtfilt(b, a, LFP, axis=0)
    
    
def analyse_estimated_LFP(StateM, subset, fs=10000.):
    """
    Analyses estimated LFP (see also `_calculate_LFP()`)
    :param StateM, subset: see `_calculate_LFP()`
    :param fs: sampling frequency
    """

    t, LFP = _estimate_LFP(StateM, subset)
    LFP = _filter_LFP(LFP, fs)

    f, Pxx = signal.welch(LFP, fs, nperseg=8192)
    
    return t, LFP, f, Pxx
    

        

    

    
    
