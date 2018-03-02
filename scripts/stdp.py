#!/usr/bin/python
# -*- coding: utf8 -*-
"""
loads in hippocampal like spike train (produced by generate_spike_train.py) and runs STD learning rule in a recurrent spiking neuron population
-> creates learned weight matrix for PC population, used by spw* scripts
see more: https://drive.google.com/file/d/0B089tpx89mdXZk55dm0xZm5adUE/view
updated to produce sym stdp curve as reported in Mishra et al. 2016 - 10.1038/ncomms11552 (+ transported to brian2)
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

SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])


eps_pyr = 0.1 # sparseness
N = 8000  # #{neurons}

 
def learning(spikeTrains, taup, taum, Ap, Am, wmax, w_init):
    """
    Takes a spiking group of neurons, connects the neurons sparsely with each other,
    and learns the weight 'pattern' via STDP:
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param spikeTrains: list of lists (or eq. np.array) created in `generate_spike_train.py` - spike train used for learning
    :param taup, taum: time constant of weight change (in ms)
    :param Ap, Am: max amplitude of weight change
    :param wmax: maximum weight (in S)
    :param w_init: initial weights (in S)
    :return weightmx: numpy ndarray with the learned synaptic weights
            spikeM: SpikeMonitor of the network (for plotting and further analysis)
            mode_: ['asym', 'sym'] just for saving conventions (see saved wmx figures)
    """
    
    mode_ = plot_STDP_rule(taup/ms, taum/ms, Ap/1e-9, Am/1e-9, "STDP_rule")
    
    # create 2 numpy arrays for Brian2's SpikeGeneratorGroup
    spikingNrns = 0 * np.ones_like(spikeTrains[0])
    spikeTimes = np.asarray(spikeTrains[0])
    for neuron in range(1, N):
        nrn = neuron * np.ones_like(spikeTrains[neuron])
        spikingNrns = np.concatenate((spikingNrns, nrn), axis=0)
        tmp = np.asarray(spikeTrains[neuron])
        spikeTimes = np.concatenate((spikeTimes, tmp), axis=0)

    print "spike times loaded"
    
    PC = SpikeGeneratorGroup(N, spikingNrns, spikeTimes*second)
    
    np.random.seed(12345)
    pyrandom.seed(12345)

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

    # run simulation
    spikeM = SpikeMonitor(PC, record=True)
    run(400*second, report="text")  # the generated spike train is 500 sec long...
    
    # weight matrix
    weightmx = np.zeros((8000, 8000))
    weightmx[STDP.i[:], STDP.j[:]] = STDP.w[:]

    return weightmx, spikeM, mode_


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[1]       
    except:
        STDP_mode = "asym"
        
    assert STDP_mode in ["asym", "sym"]
        
    fIn = "spikeTrainsR.npz"
    fOut = "wmx_%s.pkl"%STDP_mode
               
    # STDP parameters (see `optimization/analyse_STDP.py`)
    if STDP_mode == "asym":
        taup = taum = 20 * ms
        Ap = 0.01
        Am = -Ap
        wmax = 40e-9  # S (w is dimensionless in the equations)
        scale_factor = 1.  # scaling factor necessary to get replay with the new cell models (not sure if this it should be here or somewhere else)
    elif STDP_mode == "sym":
        taup = taum = 62.5 * ms
        Ap = Am = 0.006
        wmax = 10e-9  # S (w is dimensionless in the equations)
        scale_factor = 1. # scaling factor necessary to get replay with the new cell models (not sure if this it should be here or somewhere else)
    Ap *= wmax  # needed to reproduce Brian1 results
    Am *= wmax  # needed to reproduce Brian1 results
    w_init = 0.1e-9  # S (w is dimensionless in the equations)
        
    # importing spike times from file
    fName = os.path.join(SWBasePath, "files", fIn)
    npzFile = np.load(fName)
    spikeTrains = npzFile["spikeTrains"]
    
    weightmx, spikeM, mode_ = learning(spikeTrains, taup, taum, Ap, Am, wmax, w_init)
    #weightmx *= scale_factor  # quick and dirty additional scaling! (in an ideal world the STDP parameters should be changed to include this scaling...)
    
    assert STDP_mode == mode_, "Inconsistency (detected during plotting STDP rule) in sym vs. asym STDP rule"
    
    
    # Plots (beware of additional scaling - compared to the STDP rule plot... - see above)
    plot_wmx(weightmx, "wmx_%s"%mode_)
    plot_wmx_avg(weightmx, 100, "wmx_avg_%s"%mode_)
    plot_w_distr(weightmx, "w_distr_%s"%mode_)

    selection = np.array([500, 1999, 4000, 6000, 7498])  # some random neuron IDs to save weigths
    dWee = save_selected_w(weightmx, selection)
    plot_weights(dWee, "sel_weights_%s"%mode_)

    # save weight matrix
    fName = os.path.join(SWBasePath, "files", fOut)
    with open(fName, "wb") as f:
        pickle.dump(weightmx, f, protocol=pickle.HIGHEST_PROTOCOL)

    plt.show()
    
    
