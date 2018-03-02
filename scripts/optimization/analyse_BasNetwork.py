#!/usr/bin/python
# -*- coding: utf8 -*-
"""
analyse pure BC network with Poisson input
author: András Ecker last update: 02.2018
"""

import os
from brian2 import *
import numpy as np
import random as pyrandom
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # ignore scipy 0.18 sparse matrix warning...
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-3])
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from detect_oscillations import *
from plots import plot_PSD, plot_zoomed


NE = 8000
NI = 150
eps_pyr = 0.1
eps_bas = 0.25

# synaptic parameters
BasExc_rise = 1. * ms  # Lee 2014 (data from CA1) 
BasInh_rise = 0.25 * ms  # Bartos 2002 (20-80%)
BasExc_decay = 4.1 * ms  # Lee 2014 (data from CA1)
BasInh_decay = 1.2 * ms  # Bartos 2002
invpeak_BasExc = (BasExc_decay / BasExc_rise) ** (BasExc_rise / (BasExc_decay - BasExc_rise))
invpeak_BasInh = (BasInh_decay / BasInh_rise) ** (BasInh_rise / (BasInh_decay - BasInh_rise))
delay_BasExc = 0.9 * ms  # Geiger 1997 (data from DG)
delay_BasInh = 0.6 * ms  # Bartos 2002
E_Exc = 0.0 * mV
E_Inh = -70.0 * mV
z = 1 * nS

# parameters for bas cells (optimized by Bence)
gL_Bas = 7.0102757369e-3 * uS
tauMem_Bas = 37.7598232668 * ms
Cm_Bas = tauMem_Bas * gL_Bas
Vrest_Bas = -58.9682231705 * mV
reset_Bas = -39.1229822301 * mV
theta_Bas = -39.5972788689 * mV
tref_Bas = 1.06976577195 * ms
delta_T_Bas = 2.21103724225 * mV
v_spike_Bas = theta_Bas + 10 * delta_T_Bas

eqs_Bas = '''
dvm/dt = (-gL_Bas*(vm-Vrest_Bas) + gL_Bas*delta_T_Bas*exp((vm- theta_Bas)/delta_T_Bas) - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Bas : volt (unless refractory)
dg_ampa/dt = (invpeak_BasExc * x_ampa - g_ampa) / BasExc_rise : 1
dx_ampa/dt = -x_ampa/BasExc_decay : 1
dg_gaba/dt = (invpeak_BasInh * x_gaba - g_gaba) / BasInh_rise : 1
dx_gaba/dt = -x_gaba/BasInh_decay : 1
'''


def run_simulation(exc_rate):
    """Sets up the purely inhibitory network with outer input and runs simulation""" 

    np.random.seed(12345)
    pyrandom.seed(12345)
    
    PI = NeuronGroup(NI, model=eqs_Bas, threshold="vm>v_spike_Bas",
                     reset="vm=reset_Bas", refractory=tref_Bas, method="exponential_euler")
    PI.vm  = Vrest_Bas
    PI.g_ampa = 0
    PI.g_gaba = 0
       
    rate_ = NE * eps_pyr * exc_rate * Hz  # calc incoming rate
    outer_input = PoissonGroup(NI, rate_)
    
    Cext = Synapses(outer_input, PI, on_pre="x_ampa+=J_BasExc")
    Cext.connect(j='i')

    Cii = Synapses(PI, PI, on_pre='x_gaba+=J_BasInh')
    Cii.connect(p=eps_bas)
    Cii.delay = delay_BasInh

    smi = SpikeMonitor(PI)
    popri = PopulationRateMonitor(PI) 
    sMI = StateMonitor(PI, "vm", record=[75])  

    run(10000*ms, report="text")
    
    return smi, popri, sMI


def run_simulation_analyse_results(exc_rate):
    """runs simulation, prints out results and saves plots"""

    smi, popri, sMI = run_simulation(exc_rate)
    
    if smi.num_spikes > 0:  # check if there is any activity

        spikeTimesI, spikingNeuronsI, poprI = preprocess_monitors(smi, popri, calc_ISI=False)
        meanIr, rIAC, maxIAC, tMaxIAC, fI, PxxI = analyse_rate(poprI)
        maxIACR, tMaxIACR, avgRippleFI, ripplePI = ripple(rIAC, fI, PxxI)
        avgGammaFI, gammaPI = gamma(fI, PxxI)
        
        # print out some info
        print "Mean inhibitory rate: %.3f"%meanIr
        print "Average inh. ripple freq: %.3f"%avgRippleFI
        print "Inh. ripple power: %.3f"%ripplePI
        print "Average inh. gamma freq: %.3f"%avgGammaFI
        print "Inh. gamma power: %.3f"%gammaPI
        print "--------------------------------------------------"
        
        # Plots
        plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", "green", multiplier_=exc_rate)
        plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier_=exc_rate, Pyr_pop=False, sm=sMI)
        plt.close("all")

    else:  # if there is no activity the auto-correlation function will throw an error!

        print "No activity !"
        print "--------------------------------------------------"
        

if __name__ == "__main__":

    J_BasExc = 5.5
    J_BasInh = 5.5
    exc_rates = [1.5, 1.75, 2., 2.25, 2.5]

    for exc_rate in exc_rates:
    
        print "Excitatory rate: %.3f"%exc_rate
        run_simulation_analyse_results(exc_rate)
        
        
