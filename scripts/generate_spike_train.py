#!/usr/bin/python
# -*- coding: utf8 -*-
"""
generates hippocampal like spike trains (see also helper file: poisson_proc.py)
authors: András Ecker, Eszter Vértes, Szabolcs Káli last update: 02.2018
"""

import os
import numpy as np
import random as pyrandom
from poisson_proc import hom_poisson, inhom_poisson, refractoriness

SWBasePath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])


avgRate_outField = 0.1  # avg. firing rate outside place field [Hz]
avgRate_inField = 20.0  # avg. in-field firing rate [Hz]


def generate_spike_train(nNeurons, placeCell_ratio, seed=12345):
    """
    generates hippocampal like spike trains (used later for learning the weights via STDP)
    :param nNeurons: #{neurons} 
    :param placeCell_ratio: ratio of place cells in the whole population
    :param seed: starting seed for random number generation
    """
    
    # generate random neuronIDs being place cells and starting points for place field
    np.random.seed(seed)
    pyrandom.seed(seed)
    pfNeurons = np.sort(pyrandom.sample(range(0, nNeurons), int(nNeurons*placeCell_ratio)), kind="mergesort")
    phiStarts = np.sort(np.random.rand(nNeurons), kind="mergesort")[pfNeurons] * 2*np.pi
    
    # save place fields for further analysis
    fName = os.path.join(SWBasePath, "files", "PFstarts_%s.npz"%placeCell_ratio)
    np.savez(fName, phiStarts=phiStarts)
    
    i = 0
    spikeTrains = []
    for neuron in range(0, nNeurons):
        if neuron in pfNeurons:
            spikeTrain = inhom_poisson(avgRate_inField, phiStarts[i], seed)
            i += 1
        else:
            spikeTrain = hom_poisson(avgRate_outField, seed)
        spikeTrains.append(spikeTrain)
        seed += 1
        
        if neuron % 200 == 0:
            print "%i/%i spike trains generated"%(neuron, nNeurons)
                
    return spikeTrains


if __name__ == "__main__":

    nNeurons = 8000
    placeCell_ratio = 1.  # 0.2

    spikeTrains = generate_spike_train(nNeurons, placeCell_ratio)

    # clean spike train (based on refractory period)
    spikeTrains = refractoriness(spikeTrains)

    assert len(spikeTrains) == nNeurons

    # save results to .npz
    fOut = "spikeTrainsR_%s.npz"%placeCell_ratio
    fName = os.path.join(SWBasePath, "files", fOut)
    np.savez(fName, spikeTrains=spikeTrains)

