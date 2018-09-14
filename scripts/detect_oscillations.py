# -*- coding: utf8 -*-
"""
Functions used to extract dynamic features: detecting sequence replay, computing AC and PSD of population rate, checking for significant frequencies, ...
authors: András Ecker, Bence Bagi, Eszter Vértes, Szabolcs Káli last update: 09.2018
"""

import pickle
import numpy as np
from scipy import signal, misc
from bayesian_decoding import load_tuning_curves, extract_binspikecount, calc_posterior, fit_trajectory, test_significance


Erev_E = 0.0  # mV
Erev_I = -70.0  # mV
len_sim = 10000  # ms
volume_cond = 1 / 3.54  # S/m


def preprocess_monitors(SM, RM, calc_ISI=True):
    """
    preprocess Brian's SpikeMonitor and PopulationRateMonitor data for further analysis and plotting
    :param SM: Brian2 SpikeMonitor
    :param RM: Brian2 PopulationRateMonitor
    :param calc_ISI: bool for calculating ISIs
    :return spike_times, spiking_neurons: 2 lists: spike times and corresponding neuronIDs
            rate: firing rate of the population
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


def replay_circular(ISI_hist, th=0.7):
    """
    Checks if there is sequence replay in the circular case (simply based on repetition seen in ISIs)
    :param ISI_hist: inter spike intervals (see `preprocess_monitors()`)
    :param th: threshold for spike count in the highest and 2 nearest bins
    :return: replay: 1/nan for detected/non-detected replay
    """

    max_ID = np.argmax(ISI_hist)
    bins_3 = ISI_hist[max_ID-1:max_ID+2] if 1 <= max_ID <= len(ISI_hist)-2 else []

    replay = 1 if sum(int(i) for i in ISI_hist) * th < sum(int(i) for i in bins_3) else np.nan

    return replay


def _avg_rate(rate, bin_, zoomed=False):
    """
    Averages rate (used also for bar plots)
    :param rate: np.array representing firing rates (hard coded for 10000 ms simulations)
    :param bin_: bin size
    :param zoomed: bool for zoomed in plots
    """
        
    t = np.linspace(0, len_sim, len(rate))
    t0 = 0 if not zoomed else 9900
    t1 = np.arange(t0, len_sim+bin_, bin_)
    t2 = t1 + bin_
    avg_rate = np.zeros_like(t1, dtype=np.float)
    for i, (t1_, t2_) in enumerate(zip(t1, t2)):
        avg_ = np.mean(rate[np.where((t1_ <= t) & (t < t2_))])
        if avg_ != 0.:
            avg_rate[i] = avg_
        
    return avg_rate


def _get_consecutive_sublists(list_):
    """
    Groups list into sublists of consecutive numbers
    :param list_: input list to group
    :return cons_lists: list of lists with consecutive numbers
    """
    
    # get upper bounds of consecutive sublists
    ubs = [x for x,y in zip(list_, list_[1:]) if y-x != 1]
    
    cons_lists = []; lb = 0    
    for ub in ubs:
        tmp = [x for x in list_[lb:] if x <= ub]
        cons_lists.append(tmp)
        lb += len(tmp)
    cons_lists.append([x for x in list_[lb:]])
    
    return cons_lists


def slice_high_activity(rate, bin_=20, min_len=200.0, th=1.75):
    """
    Slices out high network activity - which will be candidates for replay detection
    :param rate: firing rate of the population
    :param bin: bin size for rate averaging (see `_avg_rate()`)
    :param min_len: minimum length of continuous high activity (in ms)
    :param th: rate threshold (above which is 'high activity')
    """
    
    idx = np.where(_avg_rate(rate, bin_) >= th)[0]
    high_act = _get_consecutive_sublists(idx.tolist())
    slice_idx = []
    for tmp in high_act:
        if len(tmp) >= np.floor(min_len/bin_):
            slice_idx.append((tmp[0]*bin_, (tmp[-1]+1)*bin_))
            
    if not slice_idx:
        print "Sustained high network activity can't be detected (with bin size:%i, min length: %.1f and %.2f threshold)!"%(bin_, min_len, th)
            
    return slice_idx


def replay_linear(spike_times, spiking_neurons, slice_idx, pklf_name, N, delta_t=10, n_spatial_points=50):
    """
    Checks if there is sequence replay, using methods originating from Davison et al. 2009 (see more in `bayesian_decoding.py`)
    :param spike_times, spiking_neurons: see `preprocess_monitors()`
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :param pklf_name: filename of saved place fields (used for tuning curves, see `bayesian_decoding/load_tuning_curves()`)
    :param N: number of shuffled versions tested (significance test, see `bayesian_decoding/test_significance()`)
    :param delta_t: length of time bins used for decoding (in ms)
    :param n_spatial_points: number of spatial points to consider for decoding    
    :return: significance: 1/nan for significant/non-significant replay detected
             results: dictinary of stored results
    """
    
    if slice_idx: 
        spatial_points = np.linspace(0, 2*np.pi, n_spatial_points)
        tuning_curves = load_tuning_curves(pklf_name, spatial_points)

        sign_replays = []; results = {}
        for bounds in slice_idx:  # iterate through sustained high activity periods
            lb = bounds[0]; ub = bounds[1]
            idx = np.where((lb <= spike_times) & (spike_times < ub))
            spike_times_tmp = spike_times[idx]
            spiking_neurons_tmp = spiking_neurons[idx]
            t_bins = np.arange(np.floor(lb), np.ceil(ub)+delta_t, delta_t)
            bin_spike_counts = extract_binspikecount(t_bins, spike_times_tmp, spiking_neurons_tmp, tuning_curves)
            
            # decode place of the animal and try to fit path
            X_posterior = calc_posterior(bin_spike_counts, tuning_curves, delta_t)
            R, fitted_path, _ = fit_trajectory(X_posterior)
            sign, shuffled_Rs = test_significance(bin_spike_counts, tuning_curves, delta_t, R, N)
            
            sign_replays.append(sign)
            results[bounds] = {"X_posterior":X_posterior, "fitted_path":fitted_path, "R":R, "shuffled_Rs":shuffled_Rs, "significance":sign}
        
        significance = 1 if not np.isnan(sign_replays).all() else np.nan
           
        return significance, results
    else:
        return np.nan, {}


def _autocorrelation(time_series):
    """
    Computes the autocorrelation of a time series (to find repetitive patterns)
    R(\tau) = \frac{E[(X_t - \mu)(X_{t+\tau} - \mu)]}{\sigma^2}
    :param time_series: time series to analyse
    :return: autocorrelation
    """

    var = np.var(time_series)
    time_series = time_series - np.mean(time_series)
    autocorrelation = np.correlate(time_series, time_series, mode="same") / var

    return autocorrelation[len(autocorrelation)/2:]
    
    
def _calc_spectrum(time_series, fs, nperseg=512):
    """
    Estimates the power spectral density of the signal using Welch's method
    :param time_series: time series to analyse
    :param fs: sampling frequency
    :param nperseg: length of segments used in periodogram averaging
    :return f: frequencies used to evaluate PSD
            Pxx: estimated PSD
    """
    
    f, Pxx = signal.welch(time_series, fs, nperseg=nperseg)
    return f, Pxx
        
        
def analyse_rate(rate, fs, slice_idx, TFR=False):
    """
    Basic analysis of firing rate: autocorrelatio, PSD, TFR (wavelet analysis)
    :param rate: firing rate of the neuron population
    :param fs: sampling frequency (for the spectral analysis)
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :param TFR: bool - calculate time freq. repr. (using wavelet analysis)
    :return: mean_rate, rate_ac: mean rate, autocorrelation of the rate
             max_ac, t_max_ac: maximum autocorrelation, time interval of maxAC
             f, Pxx: sample frequencies and power spectral density (results of PSD analysis)
             coefs, freqs: coefficients from wavelet transform and frequencies used
    """
    
    if slice_idx:
        t = np.arange(0, 10000); rates = []
        for bounds in slice_idx:  # iterate through sustained high activity periods
            lb = bounds[0]; ub = bounds[1]
            rates.append(rate[np.where((lb <= t) & (t < ub))[0]])
            
        mean_rates = [np.mean(rate) for rate in rates]
        
        rate_acs = [_autocorrelation(rate) for rate in rates]
        max_acs = [rate_ac[1:].max() for rate_ac in rate_acs]
        t_max_acs = [rate_ac[1:].argmax()+1 for rate_ac in rate_acs]
                    
        PSDs = [_calc_spectrum(rate, fs=fs, nperseg=256) for rate in rates if rate.shape[0] > 256]
        f = PSDs[0][0]
        Pxxs = np.array([tmp[1] for tmp in PSDs])
        
        if not TFR:
            return np.mean(mean_rates), rate_acs, np.mean(max_acs), np.mean(t_max_acs), f, Pxxs
        else:
            import pywt
            
            scales = np.linspace(3.5, 5, 200)  # 162-232 Hz  pywt.scale2frequency("morl", scale) / (1/fs)
            wts = [pywt.cwt(rate, scales, "morl", 1/fs) for rate in rates]
            coefs = [tmp[0] for tmp in wts]
            freqs = wts[0][1]
            
            return np.mean(mean_rates), rate_acs, np.mean(max_acs), np.mean(t_max_acs), f, Pxxs, coefs, freqs
            
    else:
        rate_ac = _autocorrelation(rate)
        f, Pxx = _calc_spectrum(rate, fs=fs, nperseg=256)
        
        if not TFR:
            return np.mean(rate), rate_ac, rate_ac[1:].max(), rate_ac[1:].argmax()+1, f, Pxx
        else:
            import pywt
            
            scales = np.linspace(3.5, 5, 200)  # 162-232 Hz  pywt.scale2frequency("morl", scale) / (1/fs)
            coefs, freqs = pywt.cwt(rate, scales, "morl", 1/fs)
            
            return np.mean(rate), rate_ac, rate_ac[1:].max(), rate_ac[1:].argmax()+1, f, Pxx, coefs, freqs


def _fisher(Pxx):
    """
    Performs Fisher g-test on PSD (see Fisher 1929: http://www.jstor.org/stable/95247?seq=1#page_scan_tab_contents)
    :param Pxx: power spectral density (see `_calc_spectrum()`)
    :return p_val: p-value
    """

    fisher_g = Pxx.max() / np.sum(Pxx)
    n = len(Pxx); upper_lim = int(np.floor(1. / fisher_g))
    p_val = np.sum([np.power(-1, i-1) * misc.comb(n, i) * np.power((1-i*fisher_g), n-1) for i in range(1, upper_lim)])
    
    return p_val


def ripple(rate_acs, f, Pxx, slice_idx=[], p_th=0.05):
    """
    Decides if there is a significant high freq. ripple oscillation by applying Fisher g-test on the power spectrum
    :param rate_acs: auto correlation function(s) of rate see (`analyse_rate()`)
    :param f, Pxx: calculated power spectrum of the neural activity and frequencies used to calculate it (see `analyse_rate()`)
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :param p_th: significance threshold for Fisher g-test
    :return: max_ac_ripple, t_max_ac_ripple: maximum autocorrelation in ripple range, time interval of maxACR
             avg_ripple_freq, ripple_power: average frequency and power of ripple band oscillation
    """

    f = np.asarray(f)
    if slice_idx:
        max_ac_ripple = [rate_ac[3:9].max() for rate_ac in rate_acs]  # hard coded values in ripple range (works with 1ms binning...)
        t_max_ac_ripple = [rate_ac[3:9].argmax()+3 for rate_ac in rate_acs]
                
        p_vals = []; freqs = []; ripple_powers = []
        for i in range(Pxx.shape[0]):
            Pxx_ripple = Pxx[i, :][np.where((150 < f) & (f < 220))]
            p_vals.append(_fisher(Pxx_ripple))
            freqs.append(Pxx_ripple.argmax())
            ripple_powers.append((sum(Pxx_ripple) / sum(Pxx[i, :])) * 100)

        idx = np.where(np.asarray(p_vals) <= p_th)[0].tolist()
        if idx:
            avg_freq = np.mean(np.asarray(freqs)[idx])
            avg_ripple_freq = f[np.where(150 < f)[0][0] + int(avg_freq)]
        else:
            avg_ripple_freq = np.nan

        return np.mean(max_ac_ripple), np.mean(t_max_ac_ripple), avg_ripple_freq, np.mean(ripple_powers)
    else:
        Pxx_ripple = Pxx[np.where((150 < f) & (f < 220))]
        p_val = _fisher(Pxx_ripple)
        avg_ripple_freq = f[np.where(150 < f)[0][0] + Pxx_ripple.argmax()] if p_val < p_th else np.nan
        ripple_power = (sum(Pxx_ripple) / sum(Pxx)) * 100
        
        return rate_acs[3:9].max(), rate_acs[3:9].argmax()+3, avg_ripple_freq, ripple_power


def gamma(f, Pxx, slice_idx=[], p_th=0.05):
    """
    Decides if there is a significant gamma freq. oscillation by applying Fisher g-test on the power spectrum
    :param f, Pxx: calculated power spectrum of the neural activity and frequencies used to calculate it (see `analyse_rate()`)
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :param p_th: significance threshold for Fisher g-test
    :return: avg_gamma_freq, gamma_power: average frequency and power of the oscillation
    """

    f = np.asarray(f)
    if slice_idx:                
        p_vals = []; freqs = []; gamma_powers = []
        for i in range(Pxx.shape[0]):
            Pxx_gamma = Pxx[i, :][np.where((30 < f) & (f < 100))]
            p_vals.append(_fisher(Pxx_gamma))
            freqs.append(Pxx_gamma.argmax())
            gamma_powers.append((sum(Pxx_gamma) / sum(Pxx[i, :])) * 100)
        
        idx = np.where(np.asarray(p_vals) <= p_th)[0].tolist()
        if idx:
            avg_freq = np.mean(np.asarray(freqs)[idx])
            avg_gamma_freq = f[np.where(30 < f)[0][0] + int(avg_freq)]
        else:
            avg_gamma_freq = np.nan
        
        return avg_gamma_freq, np.mean(gamma_powers)
    else:
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
        i_exc = g_exc * (v - Erev_E * np.ones_like(v))  # pA
        g_inh = StateM[i].g_gaba
        i_inh = g_inh * (v - Erev_I * np.ones_like(v))  # pA
        LFP += -(i_exc + i_inh)  # (this is still in pA)
        
    LFP *= 1 / (4 * np.pi * volume_cond * 1e6)  # uV (*1e-6 um conversion)
    
    return t, LFP


def _filter(time_series, fs, cut=300.):
    """
    Low pass filters time series (3rd order Butterworth filter) - used for LFP
    :param time_series: time series to analyse
    :param fs: sampling frequency
    :param cut: cut off frequency
    """
    
    b, a = signal.butter(3, cut/(fs/2.), btype="lowpass")
    
    return signal.filtfilt(b, a, time_series, axis=0)
    
    
def analyse_estimated_LFP(StateM, subset, fs=10000.):
    """
    Analyses estimated LFP (see also `_calculate_LFP()`)
    :param StateM, subset: see `_calculate_LFP()`
    :param fs: sampling frequency
    """

    t, LFP = _estimate_LFP(StateM, subset)
    LFP = _filter(LFP, fs)

    f, Pxx = _calc_spectrum(LFP, fs, nperseg=8192)
    
    return t, LFP, f, Pxx
    
   
