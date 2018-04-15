#!/usr/bin/python
# -*- coding: utf8 -*-
"""
Infers position from spike trains (maximum likelihood)
based on: Davison et al. 2009 (the difference is that the tau_i(x) tuning curves are known here, since we generated them... see: poisson_proc.py)
author: András Ecker last update: 04.2018
"""

import os
import numpy as np
from scipy.misc import factorial
import matplotlib.pyplot as plt
from plots import *

SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])


nNeurons = 4000

# temporal and spatial grid
sim_length = 10000.0  # [ms]
temporal_res = 5.0  # [ms]
spatial_res = 2*np.pi / 360.0  # [rad] ( == 1 degree)
temporal_points = np.arange(0, sim_length, temporal_res)
spatial_points = np.linspace(0, 2*np.pi, int(2*np.pi/spatial_res))

# constants copied from poisson_proc.py
lRoute = 300.0  # circumference [cm]
lPlaceField = 30.0  # length of the place field [cm]
r = lRoute / (2*np.pi)  # radius [cm]
phiPFRad = lPlaceField / r  # (angle of) place field [rad]
avgRate_inField = 20.0  # avg. in-field firing rate [Hz]
    
    
def load_spikes(fName):
    """loads in spike times and corresponding neuron IDs"""
    
    npzFile = np.load(fName)
    spikeTimes = npzFile["spikeTimes"]
    spikingNeurons = npzFile["spikingNeurons"]    
    #dSpikes = {i: spikeTimes[np.where(spikingNeurons==1)] for i in range(nNeurons)}  # brian2's SpikeMonitor().all_values() returns almost the same...
    return spikeTimes, spikingNeurons


def extract_binspikecount(spikeTimes, spikingNeurons):
    """
    Builds container of spike counts in a given interval (bin) - in order to save time in log(likelihood) calculation
    :param spikeTimes: np.array of ordered spike times (saved and loaded in ms)
    :param spikingNeurons: np.array (same shape as spikeTimes) with corresponding neuron IDs
    :return: list (1 entry for every 5ms time bin) of dictionaries {i: n_i}
    """
    
    lBin_spike_counts = []
    for t_from, t_to in zip(temporal_points[:-1], temporal_points[1:]):  # iterate over time bins
        dNum_spikes = {i: 0 for i in range(nNeurons)}
        neuronIDs, counts = np.unique(spikingNeurons[np.where((t_from < spikeTimes) & (spikeTimes < t_to))], return_counts=True)
        for i, count in zip(neuronIDs, counts):
            dNum_spikes[i] = count
        lBin_spike_counts.append(dNum_spikes)
        
    return lBin_spike_counts

    
def load_PF_starts(fName):
    """loads in saved place field starting points [rad]"""
    
    npzFile = np.load(fName)
    return npzFile["phiStarts"]
    
    
def get_tuning_curves(phiStarts):
    """
    Calculates (not estimates) tau_i(x) tuning curves
    :param phiStarts: starting point of place fileds (saved in generate_spike_trains.py)
    :return: dictionary neuronID: tuning curve
    """
    
    phiMids = np.mod(phiStarts + phiPFRad/2.0, 2*np.pi)
    phiEnds = np.mod(phiStarts + phiPFRad, 2*np.pi)
    
    dTuning_curves = {}
    for i, phiStart in enumerate(phiStarts):
        # first generate full cos() and then zero out points outside of the place field
        tau_i = np.cos((2*np.pi) / (2 * phiPFRad) * (spatial_points - phiMids[i])) * avgRate_inField
        if phiStart < phiEnds[i]:            
            tau_i[np.where(spatial_points < phiStart)] = 0.0
            tau_i[np.where(spatial_points > phiEnds[i])] = 0.0
        else:
            tau_i[np.where((spatial_points < phiStart) & (spatial_points > phiEnds[i]))] = 0.0
        dTuning_curves[i] = tau_i
        
    return dTuning_curves
    
    
def build_tau_dict(dTuning_curves):
    """
    Builds dictionary of [neuronIDs] + [tau_i(x)s] (where tau_i(x) isn't 0) - in order to save time in log(likelihood) calculation
    :param dTuning_curves: dictionary of tuning curves {neuronID: tuning curve}
    :return: dictionary x: {neurons: [IDs of neurons, whose tau_i isn't 0 at the given spatial point], taus: [corresponding tau_i s]}
    """
    
    dX_tau = {x:{"neuronIDs":[], "taus":[]} for x in spatial_points}
    for j, x in enumerate(spatial_points):
        for i, tau_i in dTuning_curves.iteritems():
            if tau_i[j] != 0.0:
                dX_tau[x]["neuronIDs"].append(i)
                dX_tau[x]["taus"].append(tau_i[j])
                
    return dX_tau


def calc_log_likelihoods(lBin_spike_counts, dX_tau, verbose=True):
    """
    Calculates log(likelihood) based on Davison et al. 2009
    log(likelihood): log(Pr(spikes|x)) = \sum_{i=1}^N n_ilog(\frac{\Delta t \tau_i(x)}{n_i!}) - \Delta t \sum_{i=1}^N \tau_i(x)
    #TODO: investigate into parallization of this...
    :param lBin_spike_counts: list (1 entry for every 5ms time bin) of dictionaries {i: n_i} - see extract_binspikecount()
    :param dX_tau: dictionary {x: [neuronIDs] + [tau_i(x)s] (where tau_i(x) isn't 0)} - see build_tau_dict()
    return: list (1 entry for every 5ms time bin) of dictionaries x: log_likelihood
    """
    
    delta_t = temporal_res * 1e-3  # convert back to second
    
    lLog_likelihoods = []
    for it, dNum_spikes in enumerate(lBin_spike_counts):  # iterate over time bins
        dLog_likelihoods = {}
        for x, dNeuronIDs_Taus in dX_tau.iteritems():  # iterate over all spatial points
        
            log_likelihood = 0
            for i, neuronID in enumerate(dNeuronIDs_Taus["neuronIDs"]):  # iterate over neurons whose tau isn't 0 in that point (instead of iterating over all of them...)
            
                n_i = dNum_spikes[neuronID]
                tau_i = dNeuronIDs_Taus["taus"][i]
                if n_i != 0.0:  # (tau_i won't be zero! - see above)
                    log_likelihood += n_i * np.log((delta_t * tau_i) / factorial(n_i).item())
                    log_likelihood -= delta_t * tau_i
                    
            dLog_likelihoods[x] = log_likelihood
        lLog_likelihoods.append(dLog_likelihoods)
        if verbose:
            if it % (1000/temporal_res) == 0 and it != 0:
                print "1000 ms processed"
    
    return lLog_likelihoods


def get_posterior(lLog_likelihoods):
    """
    Calculates posterior distribution Pr(x|spikes) for every time bin (assuming uniform prior)
    #TODO: update prior insted of leaving it uniform?
    """
    
    X_posterior = np.zeros((spatial_points.size ,temporal_points.size))
    
    for i, dLog_likelihoods in enumerate(lLog_likelihoods):  # iterate over time bins
        idx = np.argsort(np.asarray(dLog_likelihoods.keys()))  # sorting from x
        log_likelihoods = np.asarray(dLog_likelihoods.values())[idx]
        likelihoods = np.exp(log_likelihoods)
        likelihoods[np.where(likelihoods == 1.0)] = 0.0  # exp(0) = 1, but we want 0s there...
        if np.sum(likelihoods) != 0:
            X_posterior[:, i] = likelihoods/np.sum(likelihoods)
        else:
            X_posterior[:, i] = np.zeros(spatial_points.size)        
        
    return X_posterior


if __name__ == "__main__":

    fIn = "spikeTimesNeurons_OLD.npz"
    fName = os.path.join(SWBasePath, "files", fIn)
    spikeTimes, spikingNeurons = load_spikes(fName)
    
    lBin_spike_counts = extract_binspikecount(spikeTimes, spikingNeurons)
    
    fIn = "PFstarts_OLD.npz"
    fName = os.path.join(SWBasePath, "files", fIn)
    phiStarts = load_PF_starts(fName)
    
    dTuning_curves = get_tuning_curves(phiStarts)
    dX_tau = build_tau_dict(dTuning_curves)
    
    print "preprocessing done!"
    
    lLog_likelihoods = calc_log_likelihoods(lBin_spike_counts, dX_tau)
    X_posterior = get_posterior(lLog_likelihoods)
    plot_posterior(X_posterior, temporal_res, "posterior")
    plt.show()
    





