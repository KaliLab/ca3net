# -*- coding: utf8 -*-
"""
Loads in hippocampal like spike train (produced by `generate_spike_train.py`) and runs STD learning rule in a recurrent spiking neuron population
-> creates learned weight matrix for PC population, used by `spw*` scripts
updated to produce symmetric STDP curve as reported in Mishra et al. 2016 - 10.1038/ncomms11552
author: András Ecker, based on Eszter Vértes's code last update: 11.2017
"""

import os
import sys
import pickle
from brian2 import *
set_device("cpp_standalone")  # speed up the simulation with generated C++ code
import numpy as np
import random as pyrandom
import matplotlib.pyplot as plt
from plots import *


base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])

eps_pyr = 0.1  # sparseness
n_neurons = 8000


def load_spike_trains(f_name):
    """Loads in spike trains and converts it to 2 np.arrays for Brian2's SpikeGeneratorGroup"""

    npz_f = np.load(f_name)
    spike_trains = npz_f["spike_trains"]

    spiking_neurons = 0 * np.ones_like(spike_trains[0])
    spike_times = np.asarray(spike_trains[0])
    for neuron_id in range(1, n_neurons):
        tmp = neuron_id * np.ones_like(spike_trains[neuron_id])
        spiking_neurons = np.concatenate((spiking_neurons, tmp), axis=0)
        spike_times = np.concatenate((spike_times, np.asarray(spike_trains[neuron_id])), axis=0)
        
    return spiking_neurons, spike_times

 
def learning(spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, w_init):
    """
    Takes a spiking group of neurons, connects the neurons sparsely with each other, and learns the weight 'pattern' via STDP:
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param spiking_neurons, spike_times: np.arrays for Brian2's SpikeGeneratorGroup (list of lists created by `generate_spike_train.py`) - spike train used for learning
    :param taup, taum: time constant of weight change (in ms)
    :param Ap, Am: max amplitude of weight change
    :param wmax: maximum weight (in S)
    :param w_init: initial weights (in S)
    :return weightmx: numpy ndarray with the learned synaptic weights
    """
    
    np.random.seed(12345)
    pyrandom.seed(12345)
    
    plot_STDP_rule(taup/ms, taum/ms, Ap/1e-9, Am/1e-9, "STDP_rule")
    
    PC = SpikeGeneratorGroup(n_neurons, spiking_neurons, spike_times*second)

    # mimics Brian1's exponentialSTPD class, with interactions='all', update='additive'
    # see more on conversion: http://brian2.readthedocs.io/en/stable/introduction/brian1_to_2/synapses.html
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
             
    STDP.connect(condition="i!=j", p=eps_pyr)
    STDP.w = w_init

    run(400*second, report="text")
    
    weightmx = np.zeros((n_neurons, n_neurons))
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
    f_out = "wmx_%s_%.1f.pkl"%(STDP_mode, place_cell_ratio)
               
    # STDP parameters (see `optimization/analyse_STDP.py`)
    if STDP_mode == "asym":
        taup = taum = 20 * ms
        Ap = 0.01
        Am = -Ap
        wmax = 40e-9  # S
        scale_factor = 3.
    elif STDP_mode == "sym":
        taup = taum = 62.5 * ms
        Ap = Am = 0.005
        wmax = 20e-9  # S
        scale_factor = 1.4
    Ap *= wmax; Am *= wmax  # needed to reproduce Brian1 results
    w_init = 0.1e-9  # S
          
    f_name = os.path.join(base_path, "files", f_in)
    spiking_neurons, spike_times = load_spike_trains(f_name)
    
    weightmx = learning(spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, w_init)
    #weightmx *= scale_factor  # quick and dirty additional scaling! (in an ideal world the STDP parameters should be changed to include this scaling...)
    
    pklf_name = os.path.join(base_path, "files", f_out)
    with open(pklf_name, "wb") as f:
        pickle.dump(weightmx, f, protocol=pickle.HIGHEST_PROTOCOL)


    plot_wmx(weightmx, "wmx_%s"%STDP_mode)
    plot_wmx_avg(weightmx, 100, "wmx_avg_%s"%STDP_mode)
    plot_w_distr(weightmx, "w_distr_%s"%STDP_mode)

    selection = np.array([500, 1999, 4000, 6000, 7498])
    dWee = save_selected_w(weightmx, selection)
    plot_weights(dWee, "sel_weights_%s"%STDP_mode)
    plt.show()

    
   
    
    
