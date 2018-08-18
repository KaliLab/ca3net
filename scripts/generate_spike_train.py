#!/usr/bin/python
# -*- coding: utf8 -*-
"""
Generates hippocampal like spike trains (see also helper file: `poisson_proc.py`)
authors: András Ecker, Eszter Vértes, Szabolcs Káli last update: 08.2018
"""

import os
import pickle
import numpy as np
import random as pyrandom
from poisson_proc import hom_poisson, inhom_poisson, refractoriness


base_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])

outfield_rate = 0.1  # avg. firing rate outside place field [Hz]
infield_rate = 20.0  # avg. in-field firing rate [Hz]
t_max = 405.0  # [s]


def save_place_fields(place_cells, phi_starts, pklf_name):
    """
    Save place field starts and corresponding neuron IDs for further analysis (see `bayesian_decoding.py`)
    :param place_cells: list of place cell IDs 
    :param phi_starts: corresponding list of the starting degree of place fileds
    :param pklf_name: name of saved file
    """
    
    place_fields = {neuron_id:phi_starts[i] for i, neuron_id in enumerate(place_cells)}
    with open(pklf_name, "wb") as f:
        pickle.dump(place_fields, f, protocol=pickle.HIGHEST_PROTOCOL)


def generate_spike_train(n_neurons, place_cell_ratio, ordered=True, seed=12345):
    """
    Generates hippocampal like spike trains (used later for learning the weights via STDP)
    :param n_neurons: #{neurons}
    :param place_cell_ratio: ratio of place cells in the whole population
    :param ordered: bool to order neuronIDs based on their place fields
    :param seed: starting seed for random number generation    
    :return: spike_trains - list of lists with indiviual neuron's spikes
    """
    
    # generate random neuronIDs being place cells and starting points for place fields
    np.random.seed(seed)
    pyrandom.seed(seed)
    if ordered:
        place_cells = np.sort(pyrandom.sample(range(0, n_neurons), int(n_neurons*place_cell_ratio)), kind="mergesort")
        phi_starts = np.sort(np.random.rand(n_neurons), kind="mergesort")[place_cells] * 2*np.pi
        
        pklf_name = os.path.join(base_path, "files", "PFstarts_%s.pkl"%place_cell_ratio)
        save_place_fields(place_cells, phi_starts, pklf_name)
    else:
        place_cells = pyrandom.sample(range(0, n_neurons), int(n_neurons*place_cell_ratio))
        phi_starts = np.random.rand(n_neurons)[place_cells] * 2*np.pi
    
    # generate spike trains    
    spike_trains = []; i = 0
    for neuron_id in range(0, n_neurons):
        if neuron_id in place_cells:
            spike_train = inhom_poisson(infield_rate, t_max, phi_starts[i], seed)
            i += 1
        else:
            spike_train = hom_poisson(outfield_rate, t_max, seed)
        spike_trains.append(spike_train)
        seed += 1
        
        if neuron_id % 500 == 0:
            print "%i/%i spike trains generated"%(neuron_id, n_neurons)
                
    return spike_trains


if __name__ == "__main__":

    n_neurons = 8000
    place_cell_ratio = 0.5
    f_out = "spike_trains_%s.npz"%place_cell_ratio

    spike_trains = generate_spike_train(n_neurons, place_cell_ratio)  
    #spike_trains = generate_spike_train(n_neurons, place_cell_ratio, ordered=False) 
    spike_trains = refractoriness(spike_trains)  # clean spike train (based on refractory period)
   
    npzf_name = os.path.join(base_path, "files", f_out)
    np.savez(npzf_name, spike_trains=spike_trains)


