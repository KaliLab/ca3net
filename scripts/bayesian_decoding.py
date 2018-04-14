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

SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])


nNeurons = 4000
temporal_res = 5  # [ms]
spatial_res = 2*np.pi / 360.0  # [rad] ( == 1 degree)

# constants copied from poisson_proc.py
lRoute = 300.0  # circumference [cm]
lPlaceField = 30.0  # length of the place field [cm]
r = lRoute / (2*np.pi)  # radius [cm]
phiPFRad = lPlaceField / r  # (angle of) place field [rad]
avgRate_inField = 20.0  # avg. in-field firing rate [Hz]
    
    
def load_spikes(fName, nNeurons):
    """loads in spike times and organizes them"""
    
    npzFile = np.load(fName)
    spikeTimes = npzFile["spikeTimes"]
    spikingNeurons = npzFile["spikingNeurons"]
    
    dSpikes = {i: spikeTimes[np.where(spikingNeurons==1)] for i in range(nNeurons)}  # brian2's SpikeMonitor().all_values() returns almost the same...
    
    return spikeTimes, spikingNeurons, dSpikes
    
    
def load_PF_starts(fName):
    """loads in saved place field starting points [rad]"""
    
    npzFile = np.load(fName)
    return npzFile["phiStarts"]
    
    
def get_tuning_curves(phiStarts):
    """calculates (not estimates) tau_i(x) tuning curves"""
    
    spatial_points = np.linspace(0, 2*np.pi, int(2*np.pi / spatial_res))
    
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


def get_overlapping_PFs(phiStarts):
    """builds a dict with neuron ID's which have overlapping place fields (no need to iterate over every neurons later)
    #TODO: optimize this function..."""
    
    phiEnds = np.mod(phiStarts + phiPFRad, 2*np.pi)
    
    dOverlapping_PFs = {}
    for i, phiStart in enumerate(phiStarts):
        if phiStart < phiEnds[i]:
            dOverlapping_PFs[i] = [j for j, phiStart_tmp in enumerate(phiStarts) if phiStart <= phiStart_tmp and phiStart_tmp < phiEnds[i]]
        else:
            dOverlapping_PFs[i] = [j for j, phiStart_tmp in enumerate(phiStarts) if phiStart <= phiStart_tmp or phiStart_tmp < phiEnds[i]]
            
    return dOverlapping_PFs
    

if __name__ == "__main__":

    fIn = "spikeTimesNeurons_OLD.npz"
    fName = os.path.join(SWBasePath, "files", fIn)
    spikeTimes, spikingNeurons, dSpikes = load_spikes(fName, nNeurons)
    
    fIn = "PFstarts_OLD.npz"
    fName = os.path.join(SWBasePath, "files", fIn)
    phiStarts = load_PF_starts(fName)
    
    dTuning_curves = get_tuning_curves(phiStarts)
    dOverlapping_PFs = get_overlapping_PFs(phiStarts)
    
    print "...preprocessing done!"
    
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dTuning_curves[3900])
    plt.show()
    """
    






