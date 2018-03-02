#!/usr/bin/python
# -*- coding: utf8 -*-
"""
creates PC (adExp IF) and BC (exp IF) population in Brian2, loads in recurrent connection matrix for PC population
runs simulation and checks the dynamics
(updated network, parameters are/should be closer to the experimental data!)
authors: Bence Bagi, András Ecker last update: 11.2017
"""

import os
import sys
from brian2 import *
import numpy as np
import random as pyrandom
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # ignore scipy 0.18 sparse matrix warning...
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-2])
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from detect_oscillations import *
from plots import *


# size of populations
NE = 8000
NI = 150
# sparseness
eps_pyr = 0.1
eps_bas = 0.25

# synaptic time constants:
# rise time constants
PyrExc_rise = 1.3 * ms  # Guzman 2016 (only from Fig.1 H - 20-80%)
PyrExcMF_rise = 0.65 * ms  # Vyleta ... Jonas 2016 (20-80%)
PyrInh_rise = 0.3 * ms  # Bartos 2002 (20-80%)
BasExc_rise = 1. * ms  # Lee 2014 (data from CA1) 
BasInh_rise = 0.25 * ms  # Bartos 2002 (20-80%)
# decay time constants
PyrExc_decay = 9.5 * ms  # Guzman 2016 ("needed for temporal summation of EPSPs") 
PyrExcMF_decay = 5.4 * ms  # Vyleta ... Jonas 2016
PyrInh_decay = 3.3 * ms  # Bartos 2002
BasExc_decay = 4.1 * ms  # Lee 2014 (data from CA1)
BasInh_decay = 1.2 * ms  # Bartos 2002
# Normalization factors (normalize the peak of the PSC curve to 1)
invpeak_PyrExc = (PyrExc_decay / PyrExc_rise) ** (PyrExc_rise / (PyrExc_decay - PyrExc_rise))
invpeak_PyrExcMF = (PyrExcMF_decay / PyrExcMF_rise) ** (PyrExcMF_rise / (PyrExcMF_decay - PyrExcMF_rise))
invpeak_PyrInh = (PyrInh_decay / PyrInh_rise) ** (PyrInh_rise / (PyrInh_decay - PyrInh_rise))
invpeak_BasExc = (BasExc_decay / BasExc_rise) ** (BasExc_rise / (BasExc_decay - BasExc_rise))
invpeak_BasInh = (BasInh_decay / BasInh_rise) ** (BasInh_rise / (BasInh_decay - BasInh_rise))

# synaptic delays:
delay_PyrExc = 2.2 * ms  # Guzman 2016
delay_PyrInh = 1.1 * ms  # Bartos 2002
delay_BasExc = 0.9 * ms  # Geiger 1997 (data from DG)
delay_BasInh = 0.6 * ms  # Bartos 2002
        
# synaptic reversal potentials
E_Exc = 0.0 * mV
E_Inh = -70.0 * mV

# mossy fiber input
rate_MF = 20 * Hz

z = 1 * nS
# parameters for pyr cells (optimized by Bence)
gL_Pyr = 4.49581428461e-3 * uS
tauMem_Pyr = 37.97630516 * ms
Cm_Pyr = tauMem_Pyr * gL_Pyr
Vrest_Pyr = -59.710040237 * mV
reset_Pyr = -24.8988661181 * mV
theta_Pyr = -13.3139788756 * mV
tref_Pyr = 3.79313737057 * ms
a_Pyr = -0.255945300382 * nS
b_Pyr = 0.22030375858 * nA
delta_T_Pyr = 3.31719795927 * mV
tau_w_Pyr = 80.1747780694 * ms
v_spike_Pyr = theta_Pyr + 10 * delta_T_Pyr

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


eqs_Pyr = '''
dvm/dt = (-gL_Pyr*(vm-Vrest_Pyr) + gL_Pyr*delta_T_Pyr*exp((vm- theta_Pyr)/delta_T_Pyr) - w - ((g_ampa+g_ampaMF)*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Pyr : volt (unless refractory)
dw/dt = (a_Pyr*(vm- Vrest_Pyr )-w)/tau_w_Pyr : amp
dg_ampa/dt = (invpeak_PyrExc * x_ampa - g_ampa) / PyrExc_rise : 1
dx_ampa/dt = -x_ampa / PyrExc_decay : 1
dg_ampaMF/dt = (invpeak_PyrExcMF * x_ampaMF - g_ampaMF) / PyrExcMF_rise : 1
dx_ampaMF/dt = -x_ampaMF / PyrExcMF_decay : 1
dg_gaba/dt = (invpeak_PyrInh * x_gaba - g_gaba) / PyrInh_rise : 1
dx_gaba/dt = -x_gaba/PyrInh_decay : 1
'''

eqs_Bas = '''
dvm/dt = (-gL_Bas*(vm-Vrest_Bas) + gL_Bas*delta_T_Bas*exp((vm- theta_Bas)/delta_T_Bas) - (g_ampa*z*(vm-E_Exc) + g_gaba*z*(vm-E_Inh)))/Cm_Bas : volt (unless refractory)
dg_ampa/dt = (invpeak_BasExc * x_ampa - g_ampa) / BasExc_rise : 1
dx_ampa/dt = -x_ampa/BasExc_decay : 1
dg_gaba/dt = (invpeak_BasInh * x_gaba - g_gaba) / BasInh_rise : 1
dx_gaba/dt = -x_gaba/BasInh_decay : 1
'''


def run_simulation(Wee, STDP_mode="asym", detailed=True, verbose=True):
    """
    Sets up the network and runs simulation
    :param Wee: np.array representing synaptic weight matrix
    :param STDP_mode: symmetric or asymmetric weight matrix flag (used for synapse parameters)
    :param detailed: bool - useage of multi state monitor (for membrane pot and inh. and exc. inputs of some singe cells)
    :param verbose: bool - report status of simulation
    :return sme, smi, popre, popri, selection, msMe: brian2 monitors (+ array of selected cells used by multi state monitor) 
    """

    np.random.seed(12345)
    pyrandom.seed(12345)
    
    # synaptic weights
    J_PyrInh = 0.1
    if STDP_mode == "asym":    
        J_BasExc = 5
        J_BasInh = 0.4
        J_PyrMF = 24.25
    elif STDP_mode == "sym":
        J_BasExc = 5.5
        J_BasInh = 5.5
        J_PyrMF = 30
    # wmx scale factor already introduced in the stdp* script!

    PE = NeuronGroup(NE, model=eqs_Pyr, threshold="vm>v_spike_Pyr",
                     reset="vm=reset_Pyr; w+=b_Pyr", refractory=tref_Pyr, method="exponential_euler")
    PI = NeuronGroup(NI, model=eqs_Bas, threshold="vm>v_spike_Bas",
                     reset="vm=reset_Bas", refractory=tref_Bas, method="exponential_euler")
    PE.vm = Vrest_Pyr
    PE.g_ampa = 0
    PE.g_ampaMF = 0
    PE.g_gaba = 0
    PI.vm  = Vrest_Bas
    PI.g_ampa = 0
    PI.g_gaba = 0

    MF = PoissonGroup(NE, rate_MF)

    Cext = Synapses(MF, PE, on_pre="x_ampaMF+=J_PyrMF")
    Cext.connect(j='i')

    # weight matrix used here:
    Cee = Synapses(PE, PE, 'w_exc:1', on_pre='x_ampa+=w_exc')
    Cee.connect()
    Cee.w_exc = Wee.flatten()
    Cee.delay = delay_PyrExc
    del Wee  # clear memory

    Cei = Synapses(PE, PI, on_pre='x_ampa+=J_BasExc')
    Cei.connect(p=eps_pyr)
    Cei.delay = delay_BasExc

    Cie = Synapses(PI, PE, on_pre='x_gaba+=J_PyrInh')
    Cie.connect(p=eps_bas)
    Cie.delay = delay_PyrInh

    Cii = Synapses(PI, PI, on_pre='x_gaba+=J_BasInh')
    Cii.connect(p=eps_bas)
    Cii.delay = delay_BasInh

    # Monitors
    sme = SpikeMonitor(PE)
    smi = SpikeMonitor(PI)
    popre = PopulationRateMonitor(PE)
    popri = PopulationRateMonitor(PI)
    if detailed:
        selection = np.arange(0, 4800, 50)  # subset of neurons for recoring variables
        mSME = StateMonitor(PE, ["vm", "w", "g_ampa", "g_ampaMF","g_gaba"], record=selection.tolist())  # comment this out later (takes memory!)   
        sMI = StateMonitor(PI, "vm", record=[75])     
    
    if verbose:
        run(10000*ms, report="text")
    else:
        run(10000*ms)
    
    if detailed:
        return sme, smi, popre, popri, selection, mSME, sMI
    else:
        return sme, smi, popre, popri


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[1]       
    except:
        STDP_mode = "asym"
    assert(STDP_mode in ["sym", "asym"])
    
    fIn = "wmx_%s.pkl"%STDP_mode
    
    detailed = True
    TFR = False
    
    # load in weight matrix
    fName = os.path.join(SWBasePath, "files", fIn)
    Wee = load_Wee(fName)
        
    # run simulation
    if detailed:   
        sme, smi, popre, popri, selection, mSME, sMI = run_simulation(Wee, STDP_mode)
    else:
        sme, smi, popre, popri = run_simulation(Wee, STDP_mode, detailed=False)

    # analyse results
    if sme.num_spikes > 0 and smi.num_spikes > 0:  # check if there is any activity

        # analyse spikes
        spikeTimesE, spikingNeuronsE, poprE, ISIhist, bin_edges = preprocess_monitors(sme, popre)
        spikeTimesI, spikingNeuronsI, poprI = preprocess_monitors(smi, popri, calc_ISI=False)
        # detect replay
        avgReplayInterval = replay(ISIhist[3:16])  # bins from 150 to 850 (range of interest)
        
        # analyse rates
        if TFR:
            meanEr, rEAC, maxEAC, tMaxEAC, fE, PxxE, trfE, tE, freqsE = analyse_rate(poprE, TFR=True)
            meanIr, rIAC, maxIAC, tMaxIAC, fI, PxxI, trfI, tI, freqsI = analyse_rate(poprI, TFR=True)
        else:
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

        # Plots
        plot_raster_ISI(spikeTimesE, spikingNeuronsE, poprE, [ISIhist, bin_edges], "blue", multiplier_=1)
        if TFR:
            plot_PSD(poprE, rEAC, fE, PxxE, "Pyr_population", "blue", multiplier_=1,
                     TFR=True, tfr=trfE, t=tE, freqs=freqsE, fs=1000)
            plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", "green", multiplier_=1,
                     TFR=True, tfr=trfI, t=tI, freqs=freqsI, fs=1000)
        else:
            plot_PSD(poprE, rEAC, fE, PxxE, "Pyr_population", "blue", multiplier_=1)
            plot_PSD(poprI, rIAC, fI, PxxI, "Bas_population", "green", multiplier_=1)
   
        if detailed:
            subset = plot_zoomed(spikeTimesE, spikingNeuronsE, poprE, "Pyr_population", "blue", multiplier_=1,
                                 sm=mSME, selection=selection)
            plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier_=1,
                        Pyr_pop=False, sm=sMI)
            plot_detailed(mSME, subset, multiplier_=1, new_network=True)
        else:
            plot_zoomed(spikeTimesE, spikingNeuronsE, poprE, "Pyr_population", "blue", multiplier_=1)
            plot_zoomed(spikeTimesI, spikingNeuronsI, poprI, "Bas_population", "green", multiplier_=1, Pyr_pop=False)
            

    else:  # if there is no activity the auto-correlation function will throw an error!

        print "No activity !"
        print "--------------------------------------------------"

    plt.show()            
                 
                 
