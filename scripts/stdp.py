# -*- coding: utf8 -*-
"""
Loads in hippocampal like spike train (produced by `generate_spike_train.py`) and runs STD learning rule in a recurrent spiking neuron population
-> creates learned weight matrix for PC population, used by `spw*` scripts
updated to produce symmetric STDP curve as reported in Mishra et al. 2016 - 10.1038/ncomms11552
author: András Ecker, based on Eszter Vértes's code last update: 11.2017
"""

import os, sys, pickle
import numpy as np
import random as pyrandom
from brian2 import *
set_device("cpp_standalone")  # speed up the simulation with generated C++ code
import matplotlib.pyplot as plt
from plots import *


base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])

connection_prob_PC = 0.1
nPCs = 8000


def load_spike_trains(npzf_name):
    """Loads in spike trains and converts it to 2 np.arrays for Brian2's SpikeGeneratorGroup"""

    npz_f = np.load(npzf_name)
    spike_trains = npz_f["spike_trains"]

    spiking_neurons = 0 * np.ones_like(spike_trains[0])
    spike_times = np.asarray(spike_trains[0])
    for neuron_id in range(1, nPCs):
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
    :return weightmx: learned synaptic weights
    """
    
    np.random.seed(12345)
    pyrandom.seed(12345)
    
    plot_STDP_rule(taup/ms, taum/ms, Ap/1e-9, Am/1e-9, "STDP_rule")
    
    PC = SpikeGeneratorGroup(nPCs, spiking_neurons, spike_times*second)

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
             
    STDP.connect(condition="i!=j", p=connection_prob_PC)
    STDP.w = w_init

    run(400*second, report="text")
    
    weightmx = np.zeros((nPCs, nPCs))
    weightmx[STDP.i[:], STDP.j[:]] = STDP.w[:]

    return weightmx


def save_wmx(weightmx, pklf_name):
    """
    Dummy function to save the weight matrix and make python clear the memory
    :param weightmx: synaptic weight matrix to save
    :param pklf_name: file name of the saved weight matrix
    """

    np.fill_diagonal(weightmx, 0.0)
    with open(pklf_name, "wb") as f:
        pickle.dump(weightmx, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[1]       
    except:
        STDP_mode = "asym"        
    assert STDP_mode in ["asym", "sym"]
    
    place_cell_ratio = 0.5
    linear = True
    f_in = "spike_trains_%.1f_linear.npz"%place_cell_ratio if linear else "spike_trains_%.1f.npz"%place_cell_ratio
    f_out = "wmx_%s_%.1f_linear.pkl"%(STDP_mode, place_cell_ratio) if linear else "wmx_%s_%.1f.pkl"%(STDP_mode, place_cell_ratio)
    #f_in = "intermediate_spike_trains_%.1f_linear.npz"%place_cell_ratio if linear else "intermediate_spike_trains_%.1f.npz"%place_cell_ratio
    #f_out = "intermediate_wmx_%s_%.1f_linear.pkl"%(STDP_mode, place_cell_ratio) if linear else "intermediate_wmx_%s_%.1f.pkl"%(STDP_mode, place_cell_ratio)
               
    # STDP parameters (see `optimization/analyse_STDP.py`)
    if STDP_mode == "asym":
        taup = taum = 20 * ms
        Ap = 0.01
        Am = -Ap
        wmax = 4e-8  # S
        scale_factor = 1.6
    elif STDP_mode == "sym":
        taup = taum = 62.5 * ms
        Ap = Am = 4e-3
        wmax = 2e-8  # S
        scale_factor = 0.8
    w_init = 1e-10  # S
    Ap *= wmax; Am *= wmax  # needed to reproduce Brian1 results
          
    npzf_name = os.path.join(base_path, "files", f_in)
    spiking_neurons, spike_times = load_spike_trains(npzf_name)
    
    weightmx = learning(spiking_neurons, spike_times, taup, taum, Ap, Am, wmax, w_init)
    weightmx *= scale_factor  # quick and dirty additional scaling! (in an ideal world the STDP parameters should be changed to include this scaling...)
    
    
    pklf_name = os.path.join(base_path, "files", f_out)
    save_wmx(weightmx, pklf_name)

    plot_wmx(weightmx, save_name=f_out[:-4])
    plot_wmx_avg(weightmx, n_pops=100, save_name="%s_avg"%f_out[:-4])
    plot_w_distr(weightmx, save_name="%s_distr"%f_out[:-4])
    selection = np.array([500, 2400, 4000, 5500, 7015])
    plot_weights(save_selected_w(weightmx, selection), save_name="%s_sel_weights"%f_out[:-4])    
    plt.show()

    
   
    
    
