#!/usr/bin/python
# -*- coding: utf8 -*-
"""
Infers position from spike trains (maximum likelihood)
based on: Davison et al. 2009 (the difference is that the tau_i(x) tuning curves are known here, since we generated them... see: poisson_proc.py)
author: András Ecker last update: 04.2018
"""

import os
import sys
import pickle
import numpy as np
from scipy.misc import factorial
from scipy.optimize import curve_fit
from tqdm import tqdm  # progress bar
import matplotlib.pyplot as plt
from plots import *


base_path = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])
nPCs = 8000

# temporal and spatial grid
sim_length = 10000.0  # [ms]
temporal_res = 5.0  # [ms]
spatial_res = 2*np.pi / 360.0  # [rad] ( == 1 degree)
temporal_points = np.arange(0, sim_length, temporal_res)
spatial_points = np.linspace(0, 2*np.pi, int(2*np.pi/spatial_res))

# constants copied from poisson_proc.py
l_route = 300.0  # circumference [cm]
l_place_field = 30.0  # length of the place field [cm]
r = l_route / (2*np.pi)  # radius [cm]
phi_PF_rad = l_place_field / r  # (angle of) place field [rad]
outfield_rate = 0.1  # avg. firing rate outside place field [Hz]
infield_rate = 20.0  # avg. in-field firing rate [Hz]
    
    
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
    :return: list (1 entry for every 5ms time bin) of dictionaries {i: n_i}
    """
    
    bin_spike_counts = []
    for t_from, t_to in zip(temporal_points[:-1], temporal_points[1:]):  # iterate over time bins
        n_spikes = {i: 0 for i in range(nPCs)}
        neuronIDs, counts = np.unique(spiking_neurons[np.where((t_from < spike_times) & (spike_times < t_to))], return_counts=True)
        for i, count in zip(neuronIDs, counts):
            n_spikes[i] = count
        bin_spike_counts.append(n_spikes)
        
    return bin_spike_counts

    
def load_PF_starts(pklf_name):
    """Loads in saved place field starting points [rad]"""
    
    with open(pklf_name, "rb") as f:
        place_fields = pickle.load(f)

    return place_fields
    
    
def get_tuning_curves(place_fields):
    """
    Calculates (not estimates) tau_i(x) tuning curves
    :param place_fields: dict of place cell IDs and corresponding place field starting points (saved in `generate_spike_trains.py`)
    :return: tuning_curves: neuronID: tuning curve
    """
    
    tuning_curves = {}
    for neuron_id in range(nPCs):
        if neuron_id in place_fields:
            phi_start = place_fields[neuron_id]
            mid_PF = np.mod(phi_start + phi_PF_rad/2.0, 2*np.pi)
            phi_end = np.mod(phi_start + phi_PF_rad, 2*np.pi)
            # first generate full cos() and then zero out points outside of the place field
            tau_i = np.cos((2*np.pi) / (2 * phi_PF_rad) * (spatial_points - mid_PF)) * infield_rate
            if phi_start < phi_end:            
                tau_i[np.where(spatial_points < phi_start)] = 0.0
                tau_i[np.where(spatial_points > phi_end)] = 0.0
            else:
                tau_i[np.where((spatial_points < phi_start) & (spatial_points > phi_end))] = 0.0            
        else:
            tau_i = np.zeros_like(spatial_points)  #outfield_rate * np.ones_like(spatial_points)  # don't take these into account...
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


def calc_log_likelihoods(bin_spike_counts, tuning_curves, pklf_name):
    """
    Calculates log(likelihood) based on Davison et al. 2009
    log(likelihood): log(Pr(spikes|x)) = \sum_{i=1}^N n_ilog(\frac{\Delta t \tau_i(x)}{n_i!}) - \Delta t \sum_{i=1}^N \tau_i(x)
    #TODO: investigate into parallization of this...
    :param bin_spike_counts: list (1 entry for every 5ms time bin) of dictionaries {i: n_i} (see `extract_binspikecount()`)
    :param tuning_curves: dictionary of tuning curves {neuronID: tuning curve} (see `_build_tau_dict()`)
    :param pklf_name: name of the loaded/saved file
    return: log_likelihoods: list (1 entry for every 5ms time bin) of dictionaries x: log_likelihood
    """
    
    if os.path.isfile(pklf_name):  # try to load in previously saved file
        with open(pklf_name, "rb") as f:
            log_likelihoods = pickle.load(f)
        print "Log likelihoods loaded from previously saved file!"
    else:
        x_tau = _build_tau_dict(tuning_curves)
    
        delta_t = temporal_res * 1e-3  # convert back to second
        
        log_likelihoods = []
        print "Calculating log likelihoods..."
        for it, n_spikes in enumerate(tqdm(bin_spike_counts)):  # iterate over time bins
            log_likelihoods_tmp = {}
            for x, neuronIDs_taus in x_tau.iteritems():  # iterate over all spatial points
            
                log_likelihood = 0.0
                for i, neuronID in enumerate(neuronIDs_taus["neuronIDs"]):  # iterate over neurons whose tau isn't 0 in that point
                
                    n_i = n_spikes[neuronID]
                    tau_i = neuronIDs_taus["taus"][i]
                    if n_i != 0.0:  # (tau_i won't be zero! - see above)
                        log_likelihood += n_i * np.log((delta_t * tau_i) / factorial(n_i).item())
                        log_likelihood -= delta_t * tau_i
                        
                log_likelihoods_tmp[x] = log_likelihood
            log_likelihoods.append(log_likelihoods_tmp)
                    
        with open(pklf_name, "wb") as f:
            pickle.dump(log_likelihoods, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return log_likelihoods


def get_posterior(log_likelihoods):
    """
    Calculates posterior distribution Pr(x|spikes) for every time bin (assuming uniform prior)
    #TODO: update prior insted of leaving it uniform?
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


def _frac_func(x, a, b):
    """
    Dummy frac function to pass to curve_fit
    :param x: temporal points
    :param a, b: slope and offset of the line(s)
    """
    
    return (a*x+b) - np.floor(a*x+b)
    
    
def _neg_frac_func(x, a, b):
    """
    Dummy negative frac function to pass to curve_fit
    :param x: temporal points
    :param a, b: slope and offset of the line(s)
    """
    
    return (-a*x+b) - np.floor(-a*x+b)
    

def fit_trajectory_brute_force(ML_est, STDP_mode, grid_res=200):
    """
    Tries to fit a continuous trajectory to the estimated places with iterating through a parameter grid (brute force solution)
    :param ML_est: maximum likelihood estimate of the location
    :param STDP_mode: symmetric or asymmetric weight matrix flag (to handle reverse replay)
    :param grid_res:
    :param best_fit: parameters leading to the lowest sum of squares
    """

    # normalize to [0, 1]
    ML_est_tmp = ML_est / 360.
    temporal_points_tmp = temporal_points / sim_length
    
    slopes = np.linspace(10.0, 30.0, grid_res)
    offsets = np.linspace(0.0, 1.0, grid_res)
    
    best_fit = [slopes[0], offsets[0]]; lowest_SS = np.inf
    print "Brute force fit..."
    for a in tqdm(slopes):
        for b in offsets:
            y = _frac_func(temporal_points_tmp, a, b)
            SS = np.sum((ML_est_tmp - y)**2)
            if SS < lowest_SS:
                best_fit = [a, b]
                lowest_SS = SS
            if STDP_mode == "sym":
                y = _neg_frac_func(temporal_points_tmp, a, b)
                SS = np.sum((ML_est_tmp - y)**2)
                if SS < lowest_SS:
                    best_fit = [a, b]
                    lowest_SS = SS

    return best_fit
    
    
    
def fit_trajectory(ML_est, p0, STDP_mode):
    """
    Fits a continuous trajectory to the estimated places
    :param ML_est: maximum likelihood estimate of the location
    :param STDP_mode: symmetric or asymmetric weight matrix flag (to handle reverse replay)
    :param p0: initial values for scipy's curve_fit (see `fit_trajectory_brute_force()`)
    :return: y: best trajectory fit
    """

    # normalize to [0, 1]
    ML_est_tmp = ML_est / 360.
    temporal_points_tmp = temporal_points / sim_length
    
    if STDP_mode == "asym":
            
        popt, pcov = curve_fit(_frac_func, temporal_points_tmp, ML_est_tmp,
                               method="trf", p0=p0, bounds=[(10, 0), (30, 1)])
        y = _frac_func(temporal_points_tmp, *popt)
        SS = np.sum((ML_est_tmp - y)**2)
        
        print "Best fit: a:%.3f, b:%.3f, SS:%.3f"%(popt[0], popt[1], SS)
        
        return y
    
    elif STDP_mode == "sym":
        popt, pcov = curve_fit(_frac_func, temporal_points_tmp, ML_est_tmp,
                               method="trf", p0=p0, bounds=[(10, 0), (30, 1)])
        y = _frac_func(temporal_points_tmp, *popt)
        SS = np.sum((ML_est_tmp - y)**2)
    
        popt_neg, pcov_neg = curve_fit(_neg_frac_func, temporal_points_tmp, ML_est_tmp,
                           method="trf", p0=p0, bounds=[(10, 0), (30, 1)])
        y_neg = _neg_frac_func(temporal_points_tmp, *popt_neg)
        SS_neg = np.sum((ML_est_tmp - y_neg)**2)
        
        if SS < SS_neg:
            print "Best fit (positive slope): a:%.3f, b:%.3f, SS:%.3f"%(popt[0], popt[1], SS)
            return y
        else:
            print "Best fit (negative slope): a:%.3f, b:%.3f, SS:%.3f"%(popt_neg[0], popt_neg[1], SS_neg)
            return y_neg    


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[1]       
    except:
        STDP_mode = "asym"
    assert STDP_mode in ["sym", "asym"]
    
    place_cell_ratio = 0.5
    f_in_spikes = "sim_spikes_%s.npz"%STDP_mode
    f_in_PFs = "PFstarts_%s.pkl"%place_cell_ratio
    f_save_loglikelihoods = "log_likelihoods_%s_%s.pkl"%(STDP_mode, place_cell_ratio)
    
    npzf_name = os.path.join(base_path, "files", f_in_spikes)
    spike_times, spiking_neurons = load_spikes(npzf_name)
    
    bin_spike_counts = extract_binspikecount(spike_times, spiking_neurons)
    
    pklf_name = os.path.join(base_path, "files", f_in_PFs)
    phi_starts = load_PF_starts(pklf_name)
    
    tuning_curves = get_tuning_curves(phi_starts)
    
    print "Preprocessing done!"
    
    pklf_name = os.path.join(base_path, "files", f_save_loglikelihoods)
    log_likelihoods = calc_log_likelihoods(bin_spike_counts, tuning_curves, pklf_name)
    
    X_posterior = get_posterior(log_likelihoods)    
    ML_est = np.argmax(X_posterior, axis=0)  # maximum-likelihood estimate of position
    
    p0 = fit_trajectory_brute_force(ML_est, STDP_mode)
    best_fit = fit_trajectory(ML_est, p0, STDP_mode)
    
    plot_posterior(X_posterior, "posterior_%s"%STDP_mode)
    plot_trajectory(spike_times, spiking_neurons,
                    temporal_points/sim_length, ML_est/360., best_fit, os.path.join(base_path, "figures", "fitted_trajectory_%s"%STDP_mode))    
    plt.show()
    





