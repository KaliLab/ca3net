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


def evaluate_power(ripple, gamma):
    """helper function to score ripple/gamma power ratio"""
    tmp = ripple/gamma
    scoreRG = 0
    if tmp > 20.:
        scoreRG = 3
    elif 10. < tmp <= 20.:
        scoreRG = 2
    elif 2. <= tmp <= 10.:
        scoreRG = 1
    return scoreRG


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
        self.objectives = ["fitness_score"]  # random name for BluePyOpt
                       
    def generate_model(self, individual, verbose=False):
        """runs simulation (`run_sim.py`) and returns monitors"""
        sme, smi, popre, popri = sim.run_simulation(self.Wee, *individual, verbose=verbose)
        return sme, smi, popre, popri

    def evaluate_with_lists(self, individual):
        """fitness error used by BluePyOpt for the optimization"""
        sme, smi, popre, popri = self.generate_model(individual)
        gc.collect()
        
        fitness = 0
        if sme.num_spikes > 0 and smi.num_spikes > 0:  # check if there is any activity
            fitness = -1e-4
            
            # analyse spikes
            spikeTimesE, spikingNeuronsE, poprE, ISIhist, bin_edges = preprocess_monitors(sme, popre)
            spikeTimesI, spikingNeuronsI, poprI = preprocess_monitors(smi, popri, calc_ISI=False)
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
                ripple_peakE = 0
                if not np.isnan(avgRippleFE):
                    ripple_peakE = 5 - (np.abs(avgRippleFE - 180) / 180)
                ripple_peakI = 0
                if not np.isnan(avgRippleFI):
                    ripple_peakI = 6 - (np.abs(avgRippleFI - 180) / 180)
                # penalize significant gamma peak
                bool_gammaE = int(np.isnan(avgGammaFE))
                bool_gammaI = int(np.isnan(avgGammaFI))
                # look for high ripple/gamma power ratio
                ripple_gammaE = evaluate_power(ripplePE, gammaPE)
                ripple_gammaI = evaluate_power(ripplePI, gammaPI)
                # look for "low" population rate (gauss around 2.5Hz for exc. pop.)
                rateE = np.exp(-1/2*(meanEr-2.5)**2/1.5**2)  # peak normalized to 1
                
                fitness = -1 * (ripple_peakE + ripple_peakI + bool_gammaE + 1.5*bool_gammaI + ripple_gammaE + ripple_gammaI + 3*rateE)
        
        return [fitness]  # single score but has to be a list for BluePyOpt
    

