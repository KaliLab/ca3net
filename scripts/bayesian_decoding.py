#!/usr/bin/python
# -*- coding: utf8 -*-
"""
Estimates position from spike trains and fits trajectory of the animal (used for sequence replay detection)
based on: Davison et al. 2009 (the difference is that the tau_i(x) tuning curves are known here, since we generated them... see: `poisson_proc.py`)
author: András Ecker last update: 09.2018
"""

import os
import pickle
import numpy as np
from scipy.special import factorial
import multiprocessing as mp
from tqdm import tqdm  # progress bar
from poisson_proc import get_tuning_curve


base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
nPCs = 8000

# temporal and spatial grid
sim_length = 10000.0  # [ms]
temporal_res = 10.0  # [ms]
spatial_res = 2*np.pi / 180.0  # [rad] ( == 2 degree)
temporal_points = np.arange(0, sim_length, temporal_res)
spatial_points = np.linspace(0, 2*np.pi, int(2*np.pi/spatial_res))
    
    
def load_spikes(npzf_name):
    """Loads in spike times and corresponding neuron IDs"""
    
    npz_f = np.load(npzf_name)
    spike_times = npz_f["spike_times"]
    spiking_neurons = npz_f["spiking_neurons"]    
    #dSpikes = {i: spike_times[np.where(spiking_neurons==1)] for i in range(nPCs)}  # brian2's SpikeMonitor().all_values() returns almost the same...
    return spike_times, spiking_neurons


def extract_binspikecount(spike_times, spiking_neurons):
    """
    Builds container of spike counts in a given interval (bin) - in order to save time in log(likelihood) calculation
    :param spike_times: np.array of ordered spike times (saved and loaded in ms)
    :param spiking_neurons: np.array (same shape as spike_times) with corresponding neuron IDs
    :return: list (1 entry for every time bin) of dictionaries {i: n_i}
    """
    
    bin_spike_counts = []
    for t_from, t_to in zip(temporal_points[:-1], temporal_points[1:]):  # iterate over time bins
        n_spikes = {i: 0 for i in range(nPCs)}
        neuronIDs, counts = np.unique(spiking_neurons[np.where((t_from < spike_times) & (spike_times < t_to))], return_counts=True)
        for i, count in zip(neuronIDs, counts):
            n_spikes[i] = count
        bin_spike_counts.append(n_spikes)
        
    return bin_spike_counts

    
def _load_PF_starts(pklf_name):
    """
    Loads in saved place field starting points [rad]
    :param pklf_name: name of saved file
    :return: place_fields: dict neuronID: place field start (saved in `generate_spike_trains.py`)
    """
    
    with open(pklf_name, "rb") as f:
        place_fields = pickle.load(f)

    return place_fields
    
    
def load_tuning_curves(pklf_name):
    """
    Loads in tau_i(x) tuning curves (used for generating 'teaching' spike train, see `poisson_proc.py`)
    :param pklf_name: see `_load_PF_starts`
    :return: tuning_curves: dict neuronID: tuning curve
    """
    
    place_fields = _load_PF_starts(pklf_name)
    
    tuning_curves = {}
    for neuron_id in range(nPCs):
        if neuron_id in place_fields:
            phi_start = place_fields[neuron_id]            
            tau_i = get_tuning_curve(spatial_points, phi_start)     
        else:
            tau_i = np.zeros_like(spatial_points)  # don't take these into account...
        tuning_curves[neuron_id] = tau_i

    return tuning_curves
    
    
def _build_tau_dict(tuning_curves):
    """
    Builds dictionary of [neuronIDs] + [tau_i(x)s] (where tau_i(x) isn't 0) - in order to save time in log(likelihood) calculation
    :param tuning_curves: dictionary of tuning curves {neuronID: tuning curve}
    :return: x_tau: x: {neurons: [IDs of neurons, whose tau_i isn't 0 at the given spatial point], taus: [corresponding tau_is]}
    """
    
    x_tau = {x:{"neuronIDs":[], "taus":[]} for x in spatial_points}
    for j, x in enumerate(spatial_points):
        for i, tau_i in tuning_curves.iteritems():
            if tau_i[j] != 0.0:
                x_tau[x]["neuronIDs"].append(i)
                x_tau[x]["taus"].append(tau_i[j])
                
    return x_tau


def calc_log_likelihoods(bin_spike_counts, tuning_curves, verbose=True):
    """
    Calculates log(likelihood) based on Davison et al. 2009
    log(likelihood): log(Pr(spikes|x)) = \sum_{i=1}^N n_ilog(\frac{\Delta t \tau_i(x)}{n_i!}) - \Delta t \sum_{i=1}^N \tau_i(x)
    :param bin_spike_counts: list (1 entry for every time bin) of dictionaries {i: n_i} (see `extract_binspikecount()`)
    :param tuning_curves: dictionary of tuning curves {neuronID: tuning curve} (see `load_tuning_curves()`)
    :param verbose: bool for progress bar
    return: log_likelihoods: list (1 entry for every time bin) of dictionaries x: log_likelihood
    """
        
    delta_t = temporal_res * 1e-3  # convert back to second
    x_tau = _build_tau_dict(tuning_curves)
    
    if verbose:
        pbar = tqdm(total=len(bin_spike_counts))
    
    log_likelihoods = []
    for n_spikes in bin_spike_counts:  # iterate over all temporal points
    
        log_likelihoods_tmp = {}
        for x, neuronIDs_taus in x_tau.iteritems():  # iterate over all spatial points
        
            log_likelihood = 0.0
            for i, neuronID in enumerate(neuronIDs_taus["neuronIDs"]):  # iterate over neurons whose tau isn't 0 in that point
                n_i = n_spikes[neuronID]
                if n_i != 0.0:  # (tau_i won't be zero! - see above)
                    tau_i = neuronIDs_taus["taus"][i]
                    log_likelihood += n_i * np.log((delta_t * tau_i) / factorial(n_i).item()) - delta_t * tau_i       
                            
            log_likelihoods_tmp[x] = log_likelihood
        log_likelihoods.append(log_likelihoods_tmp)
        if verbose:
            pbar.update()
    
    return log_likelihoods


def get_posterior(log_likelihoods):
    """
    Calculates posterior distribution Pr(x|spikes) for every time bin (assuming uniform prior)
    #TODO: maybe update prior insted of leaving it uniform?
    :param log_likelihoods: list (over the temporal points) of dictionaries with log likelihoods of the spatial points (see `calc_log_likelihoods()`)
    """
    
    X_posterior = np.zeros((spatial_points.size ,temporal_points.size))
    
    for i, log_likelihoods_tmp in enumerate(log_likelihoods):  # iterate over time bins
        idx = np.argsort(np.asarray(log_likelihoods_tmp.keys()))  # sorting from x
        log_likelihoods = np.asarray(log_likelihoods_tmp.values())[idx]
        likelihoods = np.exp(log_likelihoods)
        likelihoods[np.where(likelihoods == 1.0)] = 0.0  # exp(0) = 1, but we want 0s there...
        if np.sum(likelihoods) != 0:
            X_posterior[:, i] = likelihoods/np.sum(likelihoods)
        else:
            X_posterior[:, i] = np.zeros_like(spatial_points)
        
    return X_posterior


def _line(x, a, b):
    """
    Dummy function used for line fitting
    :param x: independent variable
    :param a, b: slope and intercept
    """

    return a*x + b


def _evaluate_fit(X_posterior, y, t):
    """
    Calculates the goodness of fit based on Davison et al. 2009 (line fitting in a probability matrix)
    R(v, rho) = \frac{1}{n} \sum_{k=1}^n-1 Pr(|pos - (rho + v*k*\Delta t)| < d)
    :param X_posterior: posterior matrix (see `get_posterior()`)
    :param y: candidate fitted line
    :param t: time bins (same length as indexed out posterior matrix)
    :return: R: goodness of fit (in [0, 1])
    """

    R_tmp = np.zeros((11, X_posterior.shape[1]))    
    for i, k in enumerate(range(-5, 6)):  # 11 idx corresponding to 20 degrees in total
        idx = np.mod(np.round(y)-k, 180).astype(int)
        R_tmp[i,:] = X_posterior[idx,t]        
    R = np.mean(np.sum(R_tmp, axis=0))
    
    return R


def fit_trajectory(X_posterior, grid_res=200):
    """
    Brute force trajectory fit in the posterior matrix (based on Davison et al. 2009, see: `_evaluate_fit()`)
    :param X_posterior: posterior matrix (see `get_posterior()`)
    :param grid_res: number of points to try along one dimension
    :return: highest_R: best goodness of fit (`_evaluate_fit()`)
             best_fit: slope and offset parameter corresponding to the highest R
    """
    
    slopes = np.linspace(-10., 10.0, grid_res)
    offsets = np.linspace(0.0, 180.0, grid_res)
    t = np.arange(0, X_posterior.shape[1])
    
    best_fit = [slopes[0], offsets[0]]; highest_R = 0.0
    for a in slopes:
        for b in offsets:
            y = _line(t, a, b)
            R = _evaluate_fit(X_posterior, y, t)
            if R > highest_R:
                highest_R = R
                best_fit = [a, b]
    return highest_R, best_fit


def _shuffle_tuning_curves(tuning_curves, seed):
    """
    Shuffles neuron IDs and corresponding tuning curves (used for significance test)
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

    log_likelihoods = calc_log_likelihoods(*inputs)
    X_posterior = get_posterior(log_likelihoods)
    R, _ = fit_trajectory(X_posterior)
    
    return R


def test_significance(bin_spike_counts, tuning_curves, R, N):
    """
    Test significance of fitted trajectory (and detected sequence replay) by shuffling the data and re-fitting many times
    :param bin_spike_counts, tuning_curves: see `calc_log_likelihoods()`
    :param R: reference goodness of fit (from unshuffled data)
    :param N: number of shuffled versions tested
    :return: Rs: list of goodness of fits from the shuffled events
    """
    
    shuffled_tuning_curves = [_shuffle_tuning_curves(tuning_curves, seed=12345+i) for i in range(N)]
    
    pool = mp.Pool(processes=mp.cpu_count()-1)
    Rs = pool.map(_test_significance_subprocess,
                  zip([bin_spike_counts for _ in range(N)], shuffled_tuning_curves, [False for _ in range(N)]))
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
    verbose = True
    f_in_spikes = "sim_spikes_%s.npz"%STDP_mode
    f_in_PFs = "PFstarts_%s.pkl"%place_cell_ratio
    
    npzf_name = os.path.join(base_path, "files", f_in_spikes)
    spike_times, spiking_neurons = load_spikes(npzf_name)   
    bin_spike_counts = extract_binspikecount(spike_times, spiking_neurons)
    
    pklf_name = os.path.join(base_path, "files", f_in_PFs)
    tuning_curves = load_tuning_curves(pklf_name)

    log_likelihoods = calc_log_likelihoods(bin_spike_counts, tuning_curves, verbose=verbose)
    X_posterior = get_posterior(log_likelihoods)
    R, _ = fit_trajectory(X_posterior)
    
    significance, _ = test_significance(bin_spike_counts, tuning_curves, R, 10)

    





