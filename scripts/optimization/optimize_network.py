#!/usr/bin/python
# -*- coding: utf8 -*-
'''
optimize connection parameters (synaptic weights)
authors: Bence Bagi, András Ecker last update: 06.2017
'''

import os
import sys
import logging
import numpy as np
import sim_evaluator
import bluepyopt as bpop
import multiprocessing as mp
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-3])
# add the 'scripts' directory to the path (import the modules)
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from detect_oscillations import *
from plots import *
# print info into console
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


STDP_mode = "sym"
fIn = "wmx_%s.pkl"%STDP_mode
fName = os.path.join(SWBasePath, "files", fIn)

# Parameters to be fitted as a list of: (name, lower bound, upper bound)
optconf = [("J_PyrInh_", 0.1, 4),
           ("J_BasExc_", 4.5, 5.5),
           ("J_BasInh_", 4., 7.),
           ("WeeMult_", 0.5, 5.),
           ("J_PyrMF_", 5., 40.),
           ("rate_MF_", 5., 25.)]
           # the order matters! if you want to add more parameters - update run_sim.py too


if __name__ == "__main__":

    offspring_size = 2
    max_ngen = 4

    Wee = load_Wee(fName)

    # Create multiprocessing pool for parallel evaluation of fitness function
    pool = mp.Pool(processes=mp.cpu_count())

    # Create BluePyOpt optimization and run 
    evaluator = sim_evaluator.Brian2Evaluator(Wee, optconf)
    opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=offspring_size, map_function=pool.map,
                                              eta=20, mutpb=0.3, cxpb=0.7)
                                          
    pop, hof, log, hist = opt.run(max_ngen=max_ngen, cp_filename="checkpoints/checkpoint_%s.pkl"%STDP_mode)
    del pool; del opt
        
    # ====================================== end of optimization ======================================


    # summary figure (about optimization)
    plot_evolution(log.select("gen"), np.array(log.select("min")), np.array(log.select("avg")),
                   np.array(log.select("std")), "fittnes_evolution")

    pnames = [name for name, _, _ in optconf]
    # Get best individual
    best = hof[0]
    for pname, value in zip(pnames, best):
        print "%s = %.3f"% (pname, value)
    print "Fitness value: %.3f"%best.fitness.values

    # rerun with best parameters and save figures
    sme, smi, popre, popri = evaluator.generate_model(best, verbose=True)
    del evaluator

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
        
        # plot results
        plot_raster_ISI(spikeTimesE, spikingNeuronsE, poprE, [ISIhist, bin_edges], "blue", multiplier_=1)
        plot_PSD(poprE, rEAC, fE, PxxE, "Pyr_population", "blue", multiplier_=1)
        plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", "green", multiplier_=1)
        plot_zoomed(spikeTimesE, spikingNeuronsE, poprE, "Pyr_population", "blue", multiplier_=1)
        plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier_=1, Pyr_pop=False)
        
    else:  # if there is no activity the auto-correlation function will throw an error!

        print "No activity !"
        print "--------------------------------------------------"


