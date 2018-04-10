#!/usr/bin/python
# -*- coding: utf8 -*-
"""
analyse results from BluePyOpt checkpoints
author: András Ecker last update: 11.2017
"""

import os
import sys
import pickle
import numpy as np
import sim_evaluator
import matplotlib.pyplot as plt
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from detect_oscillations import *
from plots import *


# Optimized parameters (the bounds might be difference, but it's necessary to have this here...)
optconf = [("J_PyrInh_", 0.01, 0.1),
           ("J_BasExc_", 4.5, 5.5),
           ("J_BasInh_", 0.25, 1.),
           ("WeeMult_", 2.5, 3.5),
           ("J_PyrMF_", 20., 40.),
           ("rate_MF_", 10., 25.)]


def load_checkpoints(fName):
    """loads in saved checkpoints from pkl"""

    cp = pickle.load(open(fName))

    pop = cp["generation"]
    hof = cp["halloffame"]
    log = cp["logbook"]
    hist = cp["history"]
    
    # summary figure (about optimization)
    plot_evolution(log.select('gen'), np.array(log.select('min')), np.array(log.select('avg')),
                   np.array(log.select('std')), "fittnes_evolution")
               
    return pop, hof, log, hist


def run_simulation(Wee, best_indiv):
    """reruns simulation, using the optimizations (BluePyOpt's) structure"""

    evaluator = sim_evaluator.Brian2Evaluator(Wee, optconf)
    sme, smi, popre, popri = evaluator.generate_model(best_indiv, verbose=True)

    if sme.num_spikes > 0 and smi.num_spikes > 0:  # check if there is any activity
        
        # analyse spikes
        spikeTimesE, spikingNeuronsE, poprE, ISIhist, bin_edges = preprocess_monitors(sme, popre)
        spikeTimesI, spikingNeuronsI, poprI = preprocess_monitors(smi, popri, calc_ISI=False)
        # detect replay
        avgReplayInterval = replay(ISIhist[3:16])  # bins from 150 to 850 (range of interest)
        print "replay: %.3f"%avgReplayInterval
        
        # analyse rates
        meanEr, rEAC, maxEAC, tMaxEAC, fE, PxxE = analyse_rate(poprE)
        meanIr, rIAC, maxIAC, tMaxIAC, fI, PxxI = analyse_rate(poprI)
        maxEACR, tMaxEACR, avgRippleFE, ripplePE = ripple(rEAC, fE, PxxE)
        maxIACR, tMaxIACR, avgRippleFI, ripplePI = ripple(rIAC, fI, PxxI)
        avgGammaFE, gammaPE = gamma(fE, PxxE)       
        avgGammaFI, gammaPI = gamma(fI, PxxI)
        
        # print out some info
        print "Mean excitatory rate: %.3f"%meanEr
        print "Mean inhibitory rate: %.3f"%meanIr
        print "Average exc. ripple freq: %.3f"%avgRippleFE
        print "Exc. ripple power: %.3f"%ripplePE
        print "Average exc. gamma freq: %.3f"%avgGammaFE
        print "Exc. gamma power: %.3f"%gammaPE
        print "Average inh. ripple freq: %.3f"%avgRippleFI
        print "Inh. ripple power: %.3f"%ripplePI
        print "Average inh. gamma freq: %.3f"%avgGammaFI
        print "Inh. gamma power: %.3f"%gammaPI
        print "--------------------------------------------------"
        
        # plot results
        plot_raster_ISI(spikeTimesE, spikingNeuronsE, poprE, [ISIhist, bin_edges], "blue", multiplier_=1)
        plot_PSD(poprE, rEAC, fE, PxxE, "Pyr_population", "blue", multiplier_=1)
        plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", "green", multiplier_=1)
        plot_zoomed(spikeTimesE, spikingNeuronsE, poprE, "Pyr_population", "blue", multiplier_=1)
        plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier_=1, Pyr_pop=False)
        
    else:  # if there is no activity the auto-correlation function will throw an error!

        print "No activity !"
        print "--------------------------------------------------"
 

if __name__ == "__main__":

    fIn = "wmx_sym_1.pkl"
    cpIn = "checkpoint_sym_1_v0.pkl"
    
    # load in checkpoints
    fName = os.path.join(SWBasePath, "scripts", "optimization", "checkpoints", cpIn)
    _, hof, _, _ =  load_checkpoints(fName)
    
    # Get best individual
    best = hof[0]
    pnames = [name for name, _, _ in optconf]
    for pname, value in zip(pnames, best):
        print '%s = %.2f' % (pname, value)
    print 'Fitness value: ', best.fitness.values
    
    # load weight matrix
    fName = os.path.join(SWBasePath, "files", fIn)
    Wee = load_Wee(fName)
    
    # rerun simulation
    run_simulation(Wee, best)
    
    plt.show()
    

