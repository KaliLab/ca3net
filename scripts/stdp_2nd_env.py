# -*- coding: utf8 -*-
"""
`stdp.py` repeated 2 times (one spike train is 'unsorted' - see `generate_spike_trains.py`) 
-> creates learned weight matrix for PC population representing 2 learned environments, used by spw*_2envs* scripts
author: András Ecker last update: 08.2018
"""

import os
import sys
from brian2 import *
set_device("cpp_standalone")  # speed up the simulation with generated C++ code
import numpy as np
import random as pyrandom
import matplotlib.pyplot as plt
from stdp import load_spike_trains, save_wmx
from spw_network import load_wmx
from plots import *


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
    
    plot_STDP_rule(taup/ms, taum/ms, Ap/1e-9, Am/1e-9, "STDP_rule")
    
    PC = SpikeGeneratorGroup(nPCs, spiking_neurons, spike_times*second)

    STDP = Synapses(PC, PC,
            """
            w : 1
            dA_pre/dt = -A_pre/taup : 1 (event-driven)
            dA_post/dt = -A_post/taum : 1 (event-driven)
            """,
            on_pre="""
            A_pre += Ap
            w = clip(w + A_post, 0, wmax)
            """,
            on_post="""
            A_post += Am
            w = clip(w + A_pre, 0, wmax)
            """)
             
    # initialize weights from the intermediate weight matrix
    nonzero_weights = np.nonzero(intermediate_wmx)
    STDP.connect(i=nonzero_weights[0], j=nonzero_weights[1])
    STDP.w = intermediate_wmx[nonzero_weights].flatten()

    run(400*second, report="text")
    
    weightmx = np.zeros((nPCs, nPCs))
    weightmx[STDP.i[:], STDP.j[:]] = STDP.w[:]

    return weightmx


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[1]       
    except:
        STDP_mode = "asym"
        
    assert STDP_mode in ["asym", "sym"]
    
    place_cell_ratio = 0.5
    f_in = "spike_trains_%.1f.npz"%place_cell_ratio
    f_in_wmx = "intermediate_wmx_%s_%.1f.pkl"%(STDP_mode, place_cell_ratio)
    f_out = "wmx_%s_%.1f_2envs.pkl"%(STDP_mode, place_cell_ratio)
                   
    # STDP parameters (see `optimization/analyse_STDP.py`)
    if STDP_mode == "asym":
        taup = taum = 20 * ms
        Ap = 0.01
        Am = -Ap
        wmax = 4e-8  # S
        scale_factor = 3.55
    elif STDP_mode == "sym":
        taup = taum = 62.5 * ms
        Ap = Am = 5e-3
        wmax = 2e-8  # S
        scale_factor = 1.3
    Ap *= wmax; Am *= wmax  # needed to reproduce Brian1 results
    
    f_name = os.path.join(base_path, "files", f_in)
    spiking_neurons, spike_times = load_spike_trains(f_name)
    
    pklf_name = os.path.join(base_path, "files", f_in_wmx)
    intermediate_weightmx = load_wmx(pklf_name) / scale_factor  # (scale only once, at the end)
       
    weightmx = learning_2nd_env(spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, intermediate_weightmx)    
    weightmx *= scale_factor  # quick and dirty additional scaling! (in an ideal world the STDP parameters should be changed to include this scaling...)
    
    
    pklf_name = os.path.join(base_path, "files", f_out)
    save_wmx(weightmx, pklf_name)

    plot_wmx(weightmx, save_name=f_out[:-4])
    plot_wmx_avg(weightmx, n_pops=100, save_name="%s_avg"%f_out[:-4])
    plot_w_distr(weightmx, save_name="%s_distr"%f_out[:-4])
    selection = np.array([500, 2000, 4000, 6000, 7500])
    plot_weights(save_selected_w(weightmx, selection), save_name="%s_sel_weights"%f_out[:-4])    
    plt.show()
    
    
