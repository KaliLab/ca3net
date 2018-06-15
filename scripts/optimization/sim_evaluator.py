#!/usr/bin/python
# -*- coding: utf8 -*-
"""
BluePyOpt evaluator for optimization
authors: Bence Bagi, András Ecker last update: 11.2017
"""

import os
import gc
import sys
import numpy as np
import bluepyopt as bpop
import run_sim as sim
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-3])
# add the 'scripts' directory to the path (import the modules)
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from detect_oscillations import *


class Brian2Evaluator(bpop.evaluators.Evaluator):
    """class required by BluePyOpt"""

    def __init__(self, Wee, params):
        """
        :param params: list of parameters to fit - every entry must be a tuple: (name, lower bound, upper bound). 
        """
        super(Brian2Evaluator, self).__init__()
        # passing Wee with cPickle to the slaves (as BluPyOpt does) is still the fastest solution (that's why it's in self)
        self.Wee = Wee
        self.params = params
        # Parameters to be optimized
        self.params = [bpop.parameters.Parameter(name, bounds=(minval, maxval))
                       for name, minval, maxval in self.params]
        self.objectives = ["fitness_scores"]  # random name for BluePyOpt
                       
    def generate_model(self, individual, verbose=False):
        """runs simulation (`run_sim.py`) and returns monitors"""
        sme, smi, popre, popri = sim.run_simulation(self.Wee, *individual, verbose=verbose)
        return sme, smi, popre, popri

    def evaluate_with_lists(self, individual):
        """fitness error used by BluePyOpt for the optimization"""
        
        sme, smi, popre, popri = self.generate_model(individual)
        gc.collect()
        
        wc_errors = [0., 0., 0., 0., 0.,]  # worst case scenario
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
            
                # look for significant ripple peak close to 180 Hz
                ripple_peakE = np.exp(-1/2*(avgRippleFE-180.)**2/20**2)
                ripple_peakI = 2*np.exp(-1/2*(avgRippleFI-180.)**2/20**2)
                
                # look for high ripple/gamma power ratio
                ripple_gammaE = ripplePE/gammaPE
                ripple_gammaI = 2*ripplePI/gammaPI
                
                # look for "low" exc. population rate (around 2.5 Hz)
                rateE = np.exp(-1/2*(meanEr-2.5)**2/1.5**2)
                
                errors = -1 * np.linalg.norm(np.asarray(wc_errors) - np.array([ripple_peakE, ripple_peakI, ripple_gammaE, ripple_gammaI, rateE]))
        
                return errors.tolist()
            else:
                return wc_errors
        else:
            return wc_errors
    

