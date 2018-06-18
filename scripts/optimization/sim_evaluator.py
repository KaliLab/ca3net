#!/usr/bin/python
# -*- coding: utf8 -*-
"""
BluePyOpt evaluator for optimization
authors: Bence Bagi, András Ecker, Szabolcs Káli last update: 06.2018
"""

import os
import gc
import sys
import numpy as np
import bluepyopt as bpop
import run_sim as sim
SWBasePath = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add the "scripts" directory to the path (import the modules)
sys.path.insert(0, os.path.sep.join([SWBasePath, "scripts"]))
from detect_oscillations import *


class Brian2Evaluator(bpop.evaluators.Evaluator):
    """evaluator class required by BluePyOpt"""

    def __init__(self, Wee, params):
        """
        :param Wee: weight matrix (passing Wee with cPickle to the slaves (as BluPyOpt does) is still the fastest solution)
        :param params: list of parameters to fit - every entry must be a tuple: (name, lower bound, upper bound)
        """
        super(Brian2Evaluator, self).__init__()
        self.Wee = Wee
        self.params = params
        
        # Parameters to be optimized
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval))
                       for name, minval, maxval in self.params]
        self.objectives = ["replay_intE", "rippleE", "rippleI", "no_gamma_peakI", "ripple_ratioE", "ripple_ratioI", "rateE"]
                       
    def generate_model(self, individual, verbose=False):
        """runs simulation (see `run_sim.py`) and returns monitors"""
        sme, smi, popre, popri = sim.run_simulation(self.Wee, *individual, verbose=verbose)
        return sme, smi, popre, popri

    def evaluate_with_lists(self, individual):
        """fitness error used by BluePyOpt for the optimization"""
        
        sme, smi, popre, popri = self.generate_model(individual)
        gc.collect()
        
        wc_errors = [0., 0., 0., 0., 0., 0., 0.]  # worst case scenario
        if sme.num_spikes > 0 and smi.num_spikes > 0:  # check if there is any activity
            
            # analyse spikes
            spikeTimesE, spikingNeuronsE, poprE, ISIhist, bin_edges = preprocess_monitors(sme, popre)
            spikeTimesI, spikingNeuronsI, poprI = preprocess_monitors(smi, popri, calc_ISI=False)
            del sme; del smi; del popre; del popri; gc.collect()
            
            # detect replay
            avgReplayInterval = replay(ISIhist[3:16])  # bins from 150 to 850 (range of interest)
            
            if not np.isnan(avgReplayInterval):  # evaluate only if there's sequence replay!
            
                # analyse rates
                meanEr, rEAC, maxEAC, tMaxEAC, fE, PxxE = analyse_rate(poprE)
                meanIr, rIAC, maxIAC, tMaxIAC, fI, PxxI = analyse_rate(poprI)
                maxEACR, tMaxEACR, avgRippleFE, ripplePE = ripple(rEAC, fE, PxxE)
                maxIACR, tMaxIACR, avgRippleFI, ripplePI = ripple(rIAC, fI, PxxI)
                avgGammaFE, gammaPE = gamma(fE, PxxE)       
                avgGammaFI, gammaPI = gamma(fI, PxxI)
                
                # look for "fast replay"
                replay_intE = np.exp(-1/2*(avgReplayInterval-450)**2/50**2)
            
                # look for significant ripple peak close to 180 Hz
                ripple_peakE = np.exp(-1/2*(avgRippleFE-180.)**2/20**2) if not np.isnan(avgRippleFE) else 0.
                ripple_peakI = 2*np.exp(-1/2*(avgRippleFI-180.)**2/20**2) if not np.isnan(avgRippleFI) else 0.
                
                # penalize gamma peak (in inhibitory pop)
                no_gamma_peakI = 1. if np.isnan(avgGammaFI) else 0.  # this is a binary variable, which might not be the best for this algorithm
                
                # look for high ripple/gamma power ratio
                ripple_ratioE = np.clip(ripplePE/gammaPE, 0., 5.)
                ripple_ratioI = np.clip(2*ripplePI/gammaPI, 0., 10.)
                
                # look for "low" exc. population rate (around 2.5 Hz)
                rateE = np.exp(-1/2*(meanEr-2.5)**2/1.5**2)
                
                # *-1 since the algorithm tries to minimize... (max score is -21.)
                errors = -1. * np.array([replay_intE, ripple_peakE, ripple_peakI, no_gamma_peakI, ripple_ratioE, ripple_ratioI, rateE])  
        
                return errors.tolist()
            else:
                return wc_errors
        else:
            return wc_errors
    

