# -*- coding: utf8 -*-
"""
Functions used to analyse oscillations: filtering, AC, phase, PSD, checking for significant frequencies...
authors: András Ecker, Bence Bagi, Eszter Vértes, Szabolcs Káli last update: 02.2019
"""

import pickle
import numpy as np
from scipy import signal
from scipy.special import comb
import pywt
from helper import _avg_rate, _estimate_LFP


def _autocorrelation(time_series):
    """
    Computes the autocorrelation of a time series
    R(\tau) = \frac{E[(X_t - \mu)(X_{t+\tau} - \mu)]}{\sigma^2}
    :param time_series: time series to analyse
    :return: autocorrelation
    """
    var = np.var(time_series)
    time_series = time_series - np.mean(time_series)
    autocorrelation = np.correlate(time_series, time_series, mode="same") / var
    return autocorrelation[int(len(autocorrelation)/2):]


def _calc_spectrum(time_series, fs, nperseg):
    """
    Estimates the power spectral density of the signal using Welch's method
    :param time_series: time series to analyse
    :param fs: sampling frequency
    :param nperseg: length of segments used in periodogram averaging
    :return f: frequencies used to evaluate PSD
            Pxx: estimated PSD
    """
    f, Pxx = signal.welch(time_series, fs=fs, window="hann", nperseg=nperseg)
    return f, Pxx


def analyse_rate(rate, fs, slice_idx=[]):
    """
    Basic analysis of firing rate: autocorrelatio and PSD
    :param rate: firing rate of the neuron population
    :param fs: sampling frequency (for the spectral analysis)
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :return: mean_rate, rate_ac: mean rate, autocorrelation of the rate
             max_ac, t_max_ac: maximum autocorrelation, time interval of maxAC
             f, Pxx: sample frequencies and power spectral density (results of PSD analysis)
    """

    if slice_idx:
        t = np.arange(0, 10000); rates = []
        for bounds in slice_idx:  # iterate through sustained high activity periods
            lb = bounds[0]; ub = bounds[1]
            rates.append(rate[np.where((lb <= t) & (t < ub))[0]])
        # AC and PSD are only analyised in the selected parts...
        rate_acs = [_autocorrelation(rate_tmp) for rate_tmp in rates]
        max_acs = [rate_ac[1:].max() for rate_ac in rate_acs]
        t_max_acs = [rate_ac[1:].argmax()+1 for rate_ac in rate_acs]

        PSDs = [_calc_spectrum(rate_tmp, fs=fs, nperseg=256) for rate_tmp in rates]
        f = PSDs[0][0]
        Pxxs = np.array([tmp[1] for tmp in PSDs])
        return np.mean(rate), rate_acs, np.mean(max_acs), np.mean(t_max_acs), f, Pxxs
    else:
        rate_ac = _autocorrelation(rate)
        f, Pxx = _calc_spectrum(rate, fs=fs, nperseg=512)
        return np.mean(rate), rate_ac, rate_ac[1:].max(), rate_ac[1:].argmax()+1, f, Pxx


def calc_TFR(rate, fs, slice_idx=[]):
    """
    Creates time-frequency representation using wavelet analysis
    :param rate: firing rate of the neuron population
    :param fs: sampling frequency (for the spectral analysis)
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :return: coefs, freqs: coefficients from wavelet transform and frequencies used
    """

    scales = np.linspace(3.5, 5, 300)  # 162-232 Hz  pywt.scale2frequency("morl", scale) / (1/fs)
    # 27-325 Hz for 10 kHz sampled LFP...
    # scales = np.concatenate((np.linspace(25, 80, 150), np.linspace(80, 300, 150)[1:]))

    if slice_idx:
        t = np.arange(0, 10000); rates = []
        for bounds in slice_idx:  # iterate through sustained high activity periods
            lb = bounds[0]; ub = bounds[1]
            rates.append(rate[np.where((lb <= t) & (t < ub))[0]])
        wts = [pywt.cwt(rate, scales, "morl", 1/fs) for rate in rates]
        coefs = [tmp[0] for tmp in wts]
        freqs = wts[0][1]
    else:
        coefs, freqs = pywt.cwt(rate, scales, "morl", 1/fs)
    return coefs, freqs


def ripple_AC(rate_acs, slice_idx=[]):
    """
    Analyses AC of rate (in the ripple freq)
    :param rate_acs: auto correlation function(s) of rate see (`analyse_rate()`)
    :return: max_ac_ripple, t_max_ac_ripple: maximum autocorrelation in ripple range, time interval of maxACR
    """
    if slice_idx:
        max_ac_ripple = [rate_ac[3:9].max() for rate_ac in rate_acs]  # hard coded values in ripple range (works with 1ms binning...)
        t_max_ac_ripple = [rate_ac[3:9].argmax()+3 for rate_ac in rate_acs]
        return np.mean(max_ac_ripple), np.mean(t_max_ac_ripple)
    else:
        return rate_acs[3:9].max(), rate_acs[3:9].argmax()+3


def _fisher(Pxx):
    """
    Performs Fisher g-test on PSD (see Fisher 1929: http://www.jstor.org/stable/95247?seq=1#page_scan_tab_contents)
    :param Pxx: power spectral density (see `_calc_spectrum()`)
    :return p_val: p-value
    """
    fisher_g = Pxx.max() / np.sum(Pxx)
    n = len(Pxx); upper_lim = int(np.floor(1. / fisher_g))
    p_val = np.sum([np.power(-1, i-1) * comb(n, i) * np.power((1-i*fisher_g), n-1) for i in range(1, upper_lim)])
    return p_val


def ripple(f, Pxx, slice_idx=[], p_th=0.05):
    """
    Decides if there is a significant high freq. ripple oscillation by applying Fisher g-test on the power spectrum
    :param f, Pxx: calculated power spectrum of the neural activity and frequencies used to calculate it (see `analyse_rate()`)
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :param p_th: significance threshold for Fisher g-test
    :return: avg_ripple_freq, ripple_power: average frequency and power of ripple band oscillation
    """

    f = np.asarray(f)
    if slice_idx:
        p_vals, freqs, ripple_powers = [], [], []
        for i in range(Pxx.shape[0]):
            Pxx_ripple = Pxx[i, :][np.where((150 < f) & (f < 220))]
            p_vals.append(_fisher(Pxx_ripple))
            freqs.append(Pxx_ripple.argmax())
            ripple_powers.append((sum(Pxx_ripple) / sum(Pxx[i, :])) * 100)
        idx = np.where(np.asarray(p_vals) <= p_th)[0].tolist()
        if len(idx) >= 0.25*len(slice_idx):  # if at least 25% are significant
            avg_freq = np.mean(np.asarray(freqs)[idx])
            avg_ripple_freq = f[np.where(150 < f)[0][0] + int(avg_freq)]
        else:
            avg_ripple_freq = np.nan
        return avg_ripple_freq, np.mean(ripple_powers)
    else:
        Pxx_ripple = Pxx[np.where((150 < f) & (f < 220))]
        p_val = _fisher(Pxx_ripple)
        avg_ripple_freq = f[np.where(150 < f)[0][0] + Pxx_ripple.argmax()] if p_val < p_th else np.nan
        ripple_power = (sum(Pxx_ripple) / sum(Pxx)) * 100
        return avg_ripple_freq, ripple_power


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
        p_vals, freqs, gamma_powers = [], [], []
        for i in range(Pxx.shape[0]):
            Pxx_gamma = Pxx[i, :][np.where((30 < f) & (f < 100))]
            p_vals.append(_fisher(Pxx_gamma))
            freqs.append(Pxx_gamma.argmax())
            gamma_powers.append((sum(Pxx_gamma) / sum(Pxx[i, :])) * 100)
        idx = np.where(np.asarray(p_vals) <= p_th)[0].tolist()
        if len(idx) >= 0.25*len(slice_idx):  # if at least 25% are significant
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


def lowfreq(f, Pxx, slice_idx=[], p_th=0.05):
    """
    Decides if there is a significant sub gamma (alpha, beta) freq. oscillation by applying Fisher g-test on the power spectrum
    (This function is only used during optimizations to supress low freq. oscillations and ensure that the oscillation
    we get is not a harmonic of those)
    :param f, Pxx: calculated power spectrum of the neural activity and frequencies used to calculate it (see `analyse_rate()`)
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :param p_th: significance threshold for Fisher g-test
    :return: avg_subgamma_freq, subgamma_power: average frequency and power of the oscillations
    """

    f = np.asarray(f)
    if slice_idx:
        p_vals, freqs, subgamma_powers = [], [], []
        for i in range(Pxx.shape[0]):
            Pxx_subgamma = Pxx[i, :][np.where(f < 30)]
            p_vals.append(_fisher(Pxx_subgamma))
            freqs.append(Pxx_subgamma.argmax())
            subgamma_powers.append((sum(Pxx_subgamma) / sum(Pxx[i, :])) * 100)
        idx = np.where(np.asarray(p_vals) <= p_th)[0].tolist()
        if len(idx) >= 0.25*len(slice_idx):  # if at least 25% are significant
            avg_subgamma_freq = int(np.mean(np.asarray(freqs)[idx]))
        else:
            avg_subgamma_freq = np.nan
        return avg_subgamma_freq, np.mean(subgamma_powers)
    else:
        Pxx_subgamma = Pxx[f < 30]
        p_val = _fisher(Pxx_subgamma)
        avg_subgamma_freq = np.max(Pxx_subgamma) if p_val < p_th else np.nan
        subgamma_power = (sum(Pxx_subgamma) / sum(Pxx)) * 100
        return avg_subgamma_freq, subgamma_power


def lowpass_filter(time_series, fs=10000., cut=500.):
    """
    Low-pass filters time series (3rd order Butterworth filter) - (used for LFP)
    :param time_series: time series to analyse
    :param fs: sampling frequency
    :param cut: cut off frequency
    :return: filtered time_series
    """
    b, a = signal.butter(3, cut/(fs/2.), btype="lowpass")
    return signal.filtfilt(b, a, time_series, axis=0)


def bandpass_filter(time_series, fs=10000., cut=np.array([25., 60.])):
    """
    Band-pass filters time series (3rd order Butterworth filter) - (used for LFP)
    :param time_series: time series to analyse
    :param fs: sampling frequency
    :param cut: cut off frequencies
    :return: filtered time_series
    """
    b, a = signal.butter(3, cut/(fs/2.), btype="bandpass")
    return signal.filtfilt(b, a, time_series, axis=0)


def calc_phase(time_series):
    """
    Gets phase of the signal from the Hilbert transform
    :param time_series: time series to analyse
    :return: exctracted phase of the time_series
    """
    z = signal.hilbert(time_series)
    return np.angle(z)


def analyse_estimated_LFP(StateM, subset, slice_idx=[], fs=10000.):
    """
    Analyses estimated LFP (see also `_calculate_LFP()`)
    :param StateM, subset: see `_calculate_LFP()`
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :param fs: sampling frequency
    :return: t, LFP: estimated LFP and corresponding time vector
             f, Pxx: sample frequencies and power spectral density (results of PSD analysis)
    """

    t, LFP = _estimate_LFP(StateM, subset)
    LFP = lowpass_filter(LFP, fs)

    LFPs = []
    if slice_idx:
        for bounds in slice_idx:  # iterate through sustained high activity periods
            lb = bounds[0]; ub = bounds[1]
            LFPs.append(LFP[np.where((lb <= t) & (t < ub))[0]])

            PSDs = [_calc_spectrum(LFP_tmp, fs, nperseg=2048) for LFP_tmp in LFPs]
        f = PSDs[0][0]
        Pxxs = np.array([tmp[1] for tmp in PSDs])
        # for comparable results cut spectrum at 500 Hz
        f = np.asarray(f)
        idx = np.where(f < 500)[0]
        f = f[idx]
        Pxxs = Pxxs[:, idx]
        return t, LFP, f, Pxxs
    else:
        f, Pxx = _calc_spectrum(LFP, fs, nperseg=4096)
        # for comparable results cut spectrum at 500 Hz
        f = np.asarray(f)
        idx = np.where(f < 500)[0]
        f = f[idx]
        Pxx = Pxx[idx]
        return t, LFP, f, Pxx
