# -*- coding: utf8 -*-
"""
Functions used for estimating position from spike trains and fitting trajectory of the animal (used for sequence replay detection)
based on: Davison et al. 2009 (the difference is that the tau_i(x) tuning curves are known here, since we generated them... see: `poisson_proc.py`)
author: András Ecker last update: 01.2020
"""

import os, copy, pickle
import numpy as np
import random as pyrandom
from scipy.signal import convolve2d
from scipy.special import factorial
import multiprocessing as mp
from helper import load_tuning_curves


infield_rate = 20.0  # avg. in-field firing rate [Hz]


def extract_binspikecount(lb, ub, delta_t, t_incr, spike_times, spiking_neurons, tuning_curves):
    """
    Builds container of spike counts in a given interval (bin)
    In order to save time in likelihood calculation only neurons which spike are taken into account
    :param lb, ub: lower and upper bounds for time binning
    :param delta_t: window size (in ms)
    :param t_incr: increment or step size (if less than delta_t than it's an overlapping sliding window)
    :param spike_times: np.array of ordered spike times (saved and loaded in ms)
    :param spiking_neurons: np.array (same shape as spike_times) with corresponding neuron IDx
    :param tuning_curves: dictionary of tuning curves {neuronID: tuning curve} (see `load_tuning_curves()`)
    :return: list (1 entry for every time bin) of dictionaries {i: n_i}
    """

    assert delta_t >= t_incr

    bin_spike_counts = []
    t_start = lb; t_end = lb + delta_t
    while t_end < ub + t_incr:
        n_spikes = {}
        neuron_idx, counts = np.unique(spiking_neurons[np.where((t_start <= spike_times) & (spike_times < t_end))], return_counts=True)
        for i, count in zip(neuron_idx, counts):
            if i in tuning_curves:
                n_spikes[i] = count
        bin_spike_counts.append(n_spikes)
        t_start += t_incr; t_end += t_incr

    return bin_spike_counts


def calc_posterior(bin_spike_counts, tuning_curves, delta_t):
    """
    Calculates posterior distribution of decoded place Pr(x|spikes) based on Davison et al. 2009
    Pr(spikes|x) = \prod_{i=1}^N \frac{(\Delta t*tau_i(x))^n_i}{n_i!} e^{-\Delta t*tau_i(x)} (* uniform prior...)
    (It actually implements it via log(likelihoods) for numerical stability)
    Assumptions: independent neurons; firing rates modeled with Poisson processes
    Vectorized implementation using only the spiking neurons in each bin (plus taking only the highest fraction before summing...)
    :param bin_spike_counts: list (1 entry for every time bin) of spike dictionaries {i: n_i} (see `extract_binspikecount()`)
    :param tuning_curves: dictionary of tuning curves {neuronID: tuning curve} (see `helper.py/load_tuning_curves()`)
    :param delta_t: delta t used for binning spikes (in ms)
    return: X_posterior: spatial_resolution*temporal_resolution array with calculated posterior probability Pr(x|spikes)
    """

    delta_t *= 1e-3  # convert back to second
    n_spatial_points = pyrandom.sample(list(tuning_curves.values()), 1)[0].shape[0]

    X_posterior = np.zeros((n_spatial_points, len(bin_spike_counts)))  # dim:x*t

    # could be a series of 3d array operations instead of this for loop...
    # ...but since only a portion of the 8000 neurons are spiking in every bin this one might be even faster
    for t, spikes in enumerate(bin_spike_counts):

        # prepare broadcasted variables
        n_spiking_neurons = len(spikes)
        expected_spikes = np.zeros((n_spatial_points, n_spiking_neurons))  # dim:x*i_spiking
        n_spikes = np.zeros_like(expected_spikes)  # dim:x*i_spiking
        n_factorials = np.ones_like(expected_spikes)  # dim:x*i_spiking
        for j, (neuron_id, n_spike) in enumerate(spikes.items()):
            tuning_curve = tuning_curves[neuron_id] * infield_rate
            tuning_curve[np.where(tuning_curve <= 0.1)] = 0.1
            expected_spikes[:, j] = tuning_curve * delta_t
            n_spikes[:, j] = n_spike
            n_factorials[:, j] = factorial(n_spike).item()

        # calculate log(likelihood)
        likelihoods = np.multiply(expected_spikes, 1.0/n_factorials)
        likelihoods = np.multiply(n_spikes, np.log(likelihoods))
        likelihoods = likelihoods - delta_t * expected_spikes
        likelihoods.sort(axis=1, kind="mergsort")
        if likelihoods.shape[1] > 100:
            likelihoods = likelihoods[:, -100:]  # take only the 100 highest values for numerical stability
        likelihoods = np.sum(likelihoods, axis=1)
        likelihoods -= np.max(likelihoods)  # normalize before exp()
        likelihoods = np.exp(likelihoods)

        # calculate posterior
        X_posterior[:, t] = likelihoods / np.sum(likelihoods)

    return X_posterior


def _line(x, a, b):
    """
    Dummy function used for line fitting
    :param x: independent variable
    :param a, b: slope and intercept
    """

    return a*x + b


def _evaluate_fit(X_posterior, y, band_size=3):
    """
    Calculates the goodness of fit based on Davison et al. 2009 (line fitting in a probability matrix)
    R(v, rho) = \frac{1}{n} \sum_{k=1}^n-1 Pr(|pos - (rho + v*k*\Delta t)| < d)
    Masking matrix is based on Olafsdottir et al. 2016's MATLAB implementation
    :param X_posterior: posterior matrix (see `get_posterior()`)
    :param y: candidate fitted line
    :param band_size: distance (up and down) from fitted line to consider
    :return: R: goodness of fit (in [0, 1])
    """

    n_spatial_points = X_posterior.shape[0]
    t = np.arange(0, X_posterior.shape[1])

    line_idx = np.clip(np.round(y)+n_spatial_points, 0, n_spatial_points*3-1).astype(int)  # convert line to matrix idx
    if len(np.where((n_spatial_points <= line_idx) & (line_idx < n_spatial_points*2))[0]) < n_spatial_points / 3.0:  # check if line is "long enough"
        return 0.0

    mask = np.zeros((n_spatial_points*3, X_posterior.shape[1]))  # extend on top and bottom
    mask[line_idx, t] = 1
    mask = convolve2d(mask, np.ones((2*band_size+1, 1)), mode="same")  # convolve with kernel to get the desired band width
    mask = mask[int(n_spatial_points):int(n_spatial_points*2), :]  # remove extra padding to get X_posterior's shape

    R = np.sum(np.multiply(X_posterior, mask)) / np.sum(X_posterior)

    return R


def fit_trajectory(X_posterior, slope_lims=(0.5, 3), grid_res=100):
    """
    Brute force trajectory fit in the posterior matrix (based on Davison et al. 2009, see: `_evaluate_fit()`)
    :param X_posterior: posterior matrix (see `get_posterior()`)
    :param slope_lims: lower and upper bounds of splopes to test
    :param grid_res: number of points to try along one dimension
    :return: highest_R: best goodness of fit (see `_evaluate_fit()`)
             fit: fitted line
             best_params: slope and offset parameter corresponding to the highest R
    """

    slopes = np.concatenate((np.linspace(-slope_lims[1], -slope_lims[0], int(grid_res/2.)), np.linspace(slope_lims[0], slope_lims[1], int(grid_res/2.))))
    offsets = np.linspace(-0.5*X_posterior.shape[0], X_posterior.shape[0]*1.5, grid_res)
    t = np.arange(0, X_posterior.shape[1])

    best_params = (slopes[0], offsets[0]); highest_R = 0.0
    for a in slopes:
        for b in offsets:
            y = _line(t, a, b)
            R = _evaluate_fit(X_posterior, y)
            if R > highest_R:
                highest_R = R
                best_params = (a, b)
    fit = _line(t, *best_params)

    return highest_R, fit, best_params


def _shuffle_tuning_curves(tuning_curves, seed):
    """
    Shuffles neuron IDx and corresponding tuning curves (used for significance test)
    :param tuning_curves: {neuronID: tuning curve}
    :param seed: random seed for shuffling
    """

    keys = list(tuning_curves.keys())
    vals = list(tuning_curves.values())

    np.random.seed(seed)
    np.random.shuffle(keys)

    return {key:vals[i] for i, key in enumerate(keys)}


def _test_significance_subprocess(inputs):
    """
    Subprocess used by multiprocessing pool for significance test: log(likelihood) calculation and line fit
    :param inputs: see `calc_log_likelihoods()`
    :return: R: see `fit_trajectory()`
    """

    X_posterior = calc_posterior(*inputs)
    R, _, _ = fit_trajectory(X_posterior)

    return R


def test_significance(bin_spike_counts, tuning_curves, delta_t, R, N):
    """
    Test significance of fitted trajectory (and detected sequence replay) by shuffling the data and re-fitting many times
    :param delta_t, bin_spike_counts, tuning_curves: see `calc_log_likelihoods()`
    :param R: reference goodness of fit (from unshuffled data)
    :param N: number of shuffled versions tested
    :return: Rs: list of goodness of fits from the shuffled events
    """

    orig_tuning_curves = copy.deepcopy(tuning_curves)  # just to make sure...
    shuffled_tuning_curves = [_shuffle_tuning_curves(orig_tuning_curves, seed=12345+i) for i in range(N)]

    n = N if mp.cpu_count()-1 > N else mp.cpu_count()-1
    pool = mp.Pool(processes=n)
    Rs = pool.map(_test_significance_subprocess,
                  zip([bin_spike_counts for _ in range(N)], shuffled_tuning_curves, [delta_t for _ in range(N)]))
    pool.terminate()

    significance = 1 if R > np.percentile(Rs, 95) else np.nan

    return significance, sorted(Rs)
