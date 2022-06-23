# -*- coding: utf8 -*-
"""
`stdp.py` repeated 2 times (one spike train is 'unsorted' - see `generate_spike_trains.py`)
-> creates learned weight matrix for PC population representing 2 learned environments, used by spw*_2envs* scripts
author: András Ecker last update: 08.2018
"""

import os, sys, warnings
from brian2 import *
set_device("cpp_standalone")  # speed up the simulation with generated C++ code
import numpy as np
import random as pyrandom
import matplotlib.pyplot as plt
from helper import load_spike_trains, save_wmx, load_wmx
from plots import plot_STDP_rule, plot_wmx, plot_wmx_avg, plot_w_distr, save_selected_w, plot_weights


warnings.filterwarnings("ignore")
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])

nPCs = 8000  # #{neurons}


def learning_2nd_env(spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, intermediate_wmx):
    """
    Learns the second environment (very similar to `stdp.py/learning()` but initializes with a previous weight matrix)
    :param spiking_neurons, spike_times, taup, taum, Ap, Am, wmax: see `stdp.py/learning()`
    :param intermediate_wmx: weight matrix of the first environment
    :return: weightmx: learned weight matrix (represents 2 different environments)
    """

    np.random.seed(12345)
    pyrandom.seed(12345)
    #plot_STDP_rule(taup/ms, taum/ms, Ap/1e-9, Am/1e-9, "STDP_rule")

    PC = SpikeGeneratorGroup(nPCs, spiking_neurons, spike_times*second)
    STDP = Synapses(PC, PC,
            """
            w : 1
            dA_presyn/dt = -A_presyn/taup : 1 (event-driven)
            dA_postsyn/dt = -A_postsyn/taum : 1 (event-driven)
            """,
            on_pre="""
            A_presyn += Ap
            w = clip(w + A_postsyn, 0, wmax)
            """,
            on_post="""
            A_postsyn += Am
            w = clip(w + A_presyn, 0, wmax)
            """)
    # initialize weights from the intermediate weight matrix
    STDP.connect(i=intermediate_wmx.row, j=intermediate_wmx.col)
    STDP.w = intermediate_wmx.data

    run(400*second, report="text")

    weightmx = np.zeros((nPCs, nPCs))
    weightmx[STDP.i[:], STDP.j[:]] = STDP.w[:]

    return weightmx * 1e9  # *1e9 nS conversion


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[1]
    except:
        STDP_mode = "sym"
    assert STDP_mode in ["asym", "sym"]

    place_cell_ratio = 0.5
    linear = True
    f_in = "spike_trains_%.1f_linear.npz" % place_cell_ratio if linear else "spike_trains_%.1f.npz" % place_cell_ratio
    f_in_wmx = "intermediate_wmx_%s_%.1f_linear.npz" % (STDP_mode, place_cell_ratio) if linear else "intermediate_wmx_%s_%.1f.pkl" % (STDP_mode, place_cell_ratio)
    f_out = "wmx_%s_%.1f_2envs_linear.npz" % (STDP_mode, place_cell_ratio) if linear else "wmx_%s_%.1f_2envs.pkl" % (STDP_mode, place_cell_ratio)

    # STDP parameters (see `optimization/analyse_STDP.py`)
    if STDP_mode == "asym":
        taup = taum = 20 * ms
        Ap = 0.01
        Am = -Ap
        wmax = 4e-8  # S
        scale_factor = 1.27
    elif STDP_mode == "sym":
        taup = taum = 62.5 * ms
        Ap = Am = 4e-3
        wmax = 2e-8  # S
        scale_factor = 0.62
    Ap *= wmax; Am *= wmax  # needed to reproduce Brian1 results

    spiking_neurons, spike_times = load_spike_trains(os.path.join(base_path, "files", f_in))
    npzf_name = os.path.join(base_path, "files", f_in_wmx)
    intermediate_weightmx = load_wmx(npzf_name) / (scale_factor * 1e9)  # (scale only once, at the end)

    weightmx = learning_2nd_env(spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, intermediate_weightmx)
    weightmx *= scale_factor  # quick and dirty additional scaling! (in an ideal world the STDP parameters should be changed to include this scaling...)

    save_wmx(weightmx, os.path.join(base_path, "files", f_out))

    plot_wmx(weightmx, save_name=f_out[:-4])
    plot_wmx_avg(weightmx, n_pops=100, save_name="%s_avg" % f_out[:-4])
    plot_w_distr(weightmx, save_name="%s_distr" % f_out[:-4])
    selection = np.array([500, 2400, 4000, 5500, 7015])
    plot_weights(save_selected_w(weightmx, selection), save_name="%s_sel_weights" % f_out[:-4])
    plt.show()
