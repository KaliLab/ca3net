#!/usr/bin/python
# -*- coding: utf8 -*-
"""
Estimates position from spike trains and fits trajectory of the animal (used for sequence replay detection)
based on: Davison et al. 2009 (the difference is that the tau_i(x) tuning curves are known here, since we generated them... see: `poisson_proc.py`)
author: András Ecker last update: 09.2018
"""

import os
import copy
import pickle
import random
import numpy as np
from scipy.signal import convolve2d
from scipy.special import factorial
import multiprocessing as mp
from poisson_proc import get_tuning_curve


base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
nPCs = 8000
    
    
def load_spikes(npzf_name):
    """Loads in spike times and corresponding neuron IDx"""
    
    npz_f = np.load(npzf_name)
    spike_times = npz_f["spike_times"]
    spiking_neurons = npz_f["spiking_neurons"]
       
    return spike_times, spiking_neurons

    
def _load_PF_starts(pklf_name):
    """
    Loads in saved place field starting points [rad]
    :param pklf_name: filename of saved place fields
    :return: place_fields: dict neuronID: place field start (saved in `generate_spike_trains.py`)
    """
    
    with open(pklf_name, "rb") as f:
        place_fields = pickle.load(f)

    return place_fields
    
    
def load_tuning_curves(pklf_name, spatial_points):
    """
    Loads in tau_i(x) tuning curves (used for generating 'teaching' spike train, see `poisson_proc.py`)
    :param pklf_name: see `_load_PF_starts`
    :param spatial_points: spatial coordinates to evaluate the tuning curves
    :return: tuning_curves: dict of tuning curves {neuronID: tuning curve}
    """
    
    place_fields = _load_PF_starts(pklf_name)    
    tuning_curves = {neuron_id: get_tuning_curve(spatial_points, phi_start) for neuron_id, phi_start in place_fields.iteritems()}

    return tuning_curves
    
    
def extract_binspikecount(t_bins, spike_times, spiking_neurons, tuning_curves):
    """
    Builds container of spike counts in a given interval (bin)
    In order to save time in likelihood calculation only neurons which spike are taken into account
    :param t_bins: (binned) time vector
    :param spike_times: np.array of ordered spike times (saved and loaded in ms)
    :param spiking_neurons: np.array (same shape as spike_times) with corresponding neuron IDx
    :return: list (1 entry for every time bin) of dictionaries {i: n_i}
    """
    
    bin_spike_counts = []
    for t_from, t_to in zip(t_bins[:-1], t_bins[1:]):  # iterate over time bins
        n_spikes = {}
        neuron_idx, counts = np.unique(spiking_neurons[np.where((t_from <= spike_times) & (spike_times < t_to))], return_counts=True)
        for i, count in zip(neuron_idx, counts):
            if i in tuning_curves:
                n_spikes[i] = count
        bin_spike_counts.append(n_spikes)
        
    return bin_spike_counts


def calc_posterior(bin_spike_counts, tuning_curves, delta_t):
    """
    Calculates posterior distribution of decoded place Pr(x|spikes) based on Davison et al. 2009
    Pr(spikes|x) = \prod_{i=1}^N \frac{(\Delta t*tau_i(x))^n_i}{n_i!} e^{-\Delta t*tau_i(x)} (* uniform prior...)
    Assumptions: independent neurons; firing rates modeled with Poisson processes
    Vectorized implementation using only the spiking neurons in each bin (+ deleting loads of 0s afterwards because of 0.0 tuning curve values)
    #TODO: maybe update prior insted of leaving it uniform?
    :param bin_spike_counts: list (1 entry for every time bin) of spike dictionaries {i: n_i} (see `extract_binspikecount()`)
    :param tuning_curves: dictionary of tuning curves {neuronID: tuning curve} (see `load_tuning_curves()`)
    :param delta_t: delta t used for binning spikes (in ms)
    return: X_posterior: spatial_resolution*temporal_resolution array with calculated posterior probability Pr(x|spikes)
    """
        
    delta_t *= 1e-3  # convert back to second
    n_spatial_points = tuning_curves[random.sample(tuning_curves, 1)[0]].shape[0]
    
    X_posterior = np.zeros((n_spatial_points, len(bin_spike_counts)))  # dim:x*t
        
    # could be a series of 3d array operations instead of this for loop...
    # ...but since only a portion of the 4000 neurons are spiking in every bin this one is faster
    for t, spikes in enumerate(bin_spike_counts):
    
        # prepare broadcasted variables
        n_spiking_neurons = len(spikes.keys())
        expected_spikes = np.zeros((n_spatial_points, n_spiking_neurons))  # dim:x*i_spiking
        n_spikes = np.zeros_like(expected_spikes)  # dim:x*i_spiking
        n_factorials = np.ones_like(expected_spikes)  # dim:x*i_spiking
        for j, (neuron_id, n_spike) in enumerate(spikes.iteritems()):
            expected_spikes[:, j] = tuning_curves[neuron_id] * delta_t
            n_spikes[:, j] = n_spike
        for j in range(n_spatial_points):
            n_factorials[j, :] = factorial(spikes.values())
        
        # calculate likelihood
        likelihoods = np.power(expected_spikes, n_spikes)  # dim:x*i_spiking
        likelihoods = np.multiply(likelihoods, 1.0/n_factorials)
        likelihoods = np.multiply(likelihoods, np.exp(-expected_spikes))
        likelihoods[np.where(likelihoods == 0.0)] = 1.0  # replace 0s from many 0 tau_is before prod
        likelihoods = np.prod(likelihoods, axis=1)  # dim:x*1
        likelihoods[np.where(likelihoods == 1.0)] = 0.0  # replace rows with pure 0 tau_is
        
        # calculate posterior
        if np.sum(likelihoods) != 0.0:
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
    
    mask = np.zeros((n_spatial_points*3, X_posterior.shape[1]))  # extend on top and bottom
    line_idx = np.clip(np.round(y)+n_spatial_points, 0, n_spatial_points*3-1).astype(int)  # convert line to matrix idx
    mask[line_idx, t] = 1
    mask = convolve2d(mask, np.ones((2*band_size+1, 1)), mode="same")  # convolve with kernel to get the width
    mask = mask[int(n_spatial_points):int(n_spatial_points*2), :]  # remove extra padding to get X_posterior's shape
    assert mask.shape == X_posterior.shape, "Mask dimension mismatch..."
    
    R = np.sum(np.multiply(X_posterior, mask)) / np.sum(X_posterior)
    
    return R


def fit_trajectory(X_posterior, grid_res=100):
    """
    Brute force trajectory fit in the posterior matrix (based on Davison et al. 2009, see: `_evaluate_fit()`)
    :param X_posterior: posterior matrix (see `get_posterior()`)
    :param grid_res: number of points to try along one dimension
    :return: highest_R: best goodness of fit (`_evaluate_fit()`)
             fit: fitted line
             best_params: slope and offset parameter corresponding to the highest R
    """
    
    slopes = np.linspace(-5., 5.0, grid_res)
    offsets = np.linspace(-0.5*X_posterior.shape[0], X_posterior.shape[0]*1.5, grid_res)
    t = np.arange(0, X_posterior.shape[1])
    
    best_params = [slopes[0], offsets[0]]; highest_R = 0.0
    for a in slopes:
        for b in offsets:
            y = _line(t, a, b)
            R = _evaluate_fit(X_posterior, y)
            if R > highest_R:
                highest_R = R
                best_params = [a, b]
    fit = _line(t, *best_params)
               
    return highest_R, fit, best_params


def _shuffle_tuning_curves(tuning_curves, seed):
    """
    Shuffles neuron IDx and corresponding tuning curves (used for significance test)
    :param tuning_curves: {neuronID: tuning curve}
    :param seed: random seed for shuffling
    """

    keys = tuning_curves.keys()
    vals = tuning_curves.values()
    
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
    
    pool = mp.Pool(processes=mp.cpu_count()-1)
    Rs = pool.map(_test_significance_subprocess,
                  zip([bin_spike_counts for _ in range(N)], shuffled_tuning_curves, [delta_t for _ in range(N)]))
    pool.terminate()
    
    significance = 1 if R > np.percentile(Rs, 95) else np.nan
    
    return significance, sorted(Rs)


if __name__ == "__main__":

    # these functions are mostly used by `detect_oscillations.py` for replay detection, but can be tested by running this main

    import sys
    try:
        STDP_mode = sys.argv[1]       
    except:
        STDP_mode = "asym"
    assert STDP_mode in ["sym", "asym"]
    
    place_cell_ratio = 0.5
    f_in_spikes = "sim_spikes_%s.npz"%STDP_mode
    f_in_PFs = "PFstarts_%s_linear.pkl"%place_cell_ratio
    
    delta_t = 10.0  # ms
    temporal_points = np.arange(0, 10000, delta_t)
    n_spatial_points = 50
    spatial_points = np.linspace(0, 2*np.pi, n_spatial_points)
    
    npzf_name = os.path.join(base_path, "files", f_in_spikes)
    spike_times, spiking_neurons = load_spikes(npzf_name)   
    pklf_name = os.path.join(base_path, "files", f_in_PFs)
    tuning_curves = load_tuning_curves(pklf_name, spatial_points)
    
    bin_spike_counts = extract_binspikecount(temporal_points, spike_times, spiking_neurons, tuning_curves)
    X_posterior = calc_posterior(bin_spike_counts, tuning_curves, delta_t)
    R, _, _ = fit_trajectory(X_posterior)
    
    significance, _ = test_significance(bin_spike_counts, tuning_curves, delta_t, R, 10)


