# -*- coding: utf8 -*-
"""
Generates hippocampal like spike trains (see also helper file: `poisson_proc.py`)
authors: András Ecker, Eszter Vértes, Szabolcs Káli last update: 10.2018
"""

import os, pickle
import numpy as np
from tqdm import tqdm  # progress bar
from poisson_proc import hom_poisson, inhom_poisson
from helper import save_place_fields, refractoriness 


base_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])

outfield_rate = 0.1  # avg. firing rate outside place field [Hz]
infield_rate = 20.0  # avg. in-field firing rate [Hz]
t_max = 405.0  # [s]


def generate_spike_train(n_neurons, place_cell_ratio, linear, ordered=True, seed=8888):
    """
    Generates hippocampal like spike trains (used later for learning the weights via STDP)
    :param n_neurons: #{neurons}
    :param place_cell_ratio: ratio of place cells in the whole population
    :param linear: flag for linear vs. circular track
    :param ordered: bool to order neuronIDs based on their place fields (used for teaching 2 environments - see `stdp_2nd_env.py`)
    :param seed: starting seed for random number generation
    :return: spike_trains - list of lists with indiviual neuron's spikes
    """

    assert n_neurons >= 1000, "The assumptions made during the setup hold only for a reasonably big group of neurons"

    neuronIDs = np.arange(0, n_neurons)
    # generate random neuronIDs being place cells and starting points for place fields
    if ordered:
        np.random.seed(seed)

        if linear:
            p_uniform = 1./n_neurons
            tmp = (1 - 2*2*100*p_uniform)/(n_neurons-200)
            p = np.concatenate([2*p_uniform*np.ones(100), tmp*np.ones(n_neurons-2*100), 2*p_uniform*np.ones(100)])  # slightly oversample (double prop.) the 2 ends (first and last 100 neurons) of the track
            place_cells = np.sort(np.random.choice(neuronIDs, int(n_neurons*place_cell_ratio), p=p, replace=False), kind="mergsort")
        else:
            place_cells = np.sort(np.random.choice(neuronIDs, int(n_neurons*place_cell_ratio), replace=False), kind="mergsort")
        phi_starts = np.sort(np.random.rand(n_neurons), kind="mergesort")[place_cells] * 2*np.pi

        if linear:
            phi_starts -= 0.1*np.pi  # shift half a PF against boundary effects (mid_PFs will be in [0, 2*np.pi]...)
            pklf_name = os.path.join(base_path, "files", "PFstarts_%s_linear.pkl"%place_cell_ratio)
        else:
            pklf_name = os.path.join(base_path, "files", "PFstarts_%s.pkl"%place_cell_ratio)

    else:
        np.random.seed(seed+1)

        place_cells = np.random.choice(neuronIDs, int(n_neurons*place_cell_ratio), replace=False)
        phi_starts = np.sort(np.random.rand(n_neurons)[place_cells], kind="mergesort") * 2*np.pi

        if linear:
            phi_starts -= 0.1*np.pi  # shift half a PF against boundary effects (mid_PFs will be in [0, 2*np.pi]...)
            pklf_name = os.path.join(base_path, "files", "PFstarts_%s_linear_no.pkl"%place_cell_ratio)
        else:
            pklf_name = os.path.join(base_path, "files", "PFstarts_%s_no.pkl"%place_cell_ratio)


    place_fields = {neuron_id:phi_starts[i] for i, neuron_id in enumerate(place_cells)}
    save_place_fields(place_fields, pklf_name)

    # generate spike trains
    spike_trains = []
    for neuron_id in tqdm(range(0, n_neurons)):
        if neuron_id in place_fields:
            spike_train = inhom_poisson(infield_rate, t_max, place_fields[neuron_id], linear, seed)
        else:
            spike_train = hom_poisson(outfield_rate, 100, t_max, seed)
        spike_trains.append(spike_train)
        seed += 1
    
    return spike_trains


if __name__ == "__main__":

    n_neurons = 8000
    place_cell_ratio = 0.5
    linear = True

    f_out = "spike_trains_%.1f_linear.npz"%place_cell_ratio if linear else "spike_trains_%.1f.npz"%place_cell_ratio; ordered = True
    #f_out = "intermediate_spike_trains_%.1f_linear.npz"%place_cell_ratio if linear else "intermediate_spike_trains_%.1f.npz"%place_cell_ratio; ordered = False

    spike_trains = generate_spike_train(n_neurons, place_cell_ratio, linear=linear, ordered=ordered)
    spike_trains = refractoriness(spike_trains)  # clean spike train (based on refractory period)

    npzf_name = os.path.join(base_path, "files", f_out)
    np.savez(npzf_name, *spike_trains)
    
