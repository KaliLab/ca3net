# -*- coding: utf8 -*-
"""
Generates hippocampal like spike trains (see also helper file: `poisson_proc.py`)
authors: András Ecker, Eszter Vértes, Szabolcs Káli last update: 10.2018
"""

import os, pickle
import numpy as np
import random as pyrandom
from tqdm import tqdm  # progress bar
from poisson_proc import hom_poisson, inhom_poisson
from helper import save_place_fields, refractoriness


base_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])

outfield_rate = 0.1  # avg. firing rate outside place field [Hz]
infield_rate = 20.0  # avg. in-field firing rate [Hz]
t_max = 405.0  # [s]


def generate_spike_train(n_neurons, place_cell_ratio, linear, ordered=True, seed=1234):
    """
    Generates hippocampal like spike trains (used later for learning the weights via STDP)
    :param n_neurons: #{neurons}
    :param place_cell_ratio: ratio of place cells in the whole population
    :param linear: flag for linear vs. circular track
    :param ordered: bool to order neuronIDs based on their place fields (used for teaching 2 environments - see `stdp_2nd_env.py`)
    :param seed: starting seed for random number generation
    :return: spike_trains - list of lists with indiviual neuron's spikes
    """

    np.random.seed(seed)
    pyrandom.seed(seed)

    # generate random neuronIDs being place cells and starting points for place fields
    if ordered:
        place_cells = np.sort(pyrandom.sample(range(0, n_neurons), int(n_neurons*place_cell_ratio)), kind="mergesort")
        phi_starts = np.sort(np.random.rand(n_neurons), kind="mergesort")[place_cells] * 2*np.pi

        if not linear:
            pklf_name = os.path.join(base_path, "files", "PFstarts_%s.pkl"%place_cell_ratio)
        else:
            phi_starts -= 0.1*np.pi  # shift half a PF against boundary effects (mid_PFs will be in [0, 2*np.pi]...)
            pklf_name = os.path.join(base_path, "files", "PFstarts_%s_linear.pkl"%place_cell_ratio)
        save_place_fields(place_cells, phi_starts, pklf_name)
    else:
        place_cells = pyrandom.sample(range(0, n_neurons), int(n_neurons*place_cell_ratio))
        phi_starts = np.random.rand(n_neurons)[place_cells] * 2*np.pi

        if not linear:
            pklf_name = os.path.join(base_path, "files", "PFstarts_%s_no.pkl"%place_cell_ratio)
        else:
            phi_starts -= 0.1*np.pi  # shift half a PF against boundary effects (mid_PFs will be in [0, 2*np.pi]...)
            pklf_name = os.path.join(base_path, "files", "PFstarts_%s_linear_no.pkl"%place_cell_ratio)
        save_place_fields(place_cells, phi_starts, pklf_name)

    # generate spike trains
    spike_trains = []; i = 0
    for neuron_id in tqdm(range(0, n_neurons)):
        if neuron_id in place_cells:
            spike_train = inhom_poisson(infield_rate, t_max, phi_starts[i], linear, seed)
            i += 1
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
    np.savez(npzf_name, spike_trains=spike_trains)
