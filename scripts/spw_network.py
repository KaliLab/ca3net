# -*- coding: utf8 -*-
"""
Creates PC (adExp IF) and BC (exp IF) population in Brian2, loads in recurrent connection matrix for PC population
runs simulation and checks the dynamics
(updated network, parameters are/should be closer to the experimental data!)
authors: Bence Bagi, András Ecker last update: 08.2018
"""

import os
import sys
import numpy as np
import random as pyrandom
from brian2 import *
import matplotlib.pyplot as plt
from detect_oscillations import *
from plots import *


base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])

# population size
nPCs = 8000
nBCs = 150
# sparseness
connection_prob_PC = 0.1
connection_prob_BC = 0.25

# synaptic time constants:
# rise time constants
rise_PC_E = 1.3 * ms  # Guzman 2016 (only from Fig.1 H - 20-80%)
rise_PC_MF = 0.65 * ms  # Vyleta ... Jonas 2016 (20-80%)
rise_PC_I = 0.3 * ms  # Bartos 2002 (20-80%)
rise_BC_E = 1. * ms  # Lee 2014 (data from CA1) 
rise_BC_I = 0.25 * ms  # Bartos 2002 (20-80%)
# decay time constants
decay_PC_E = 9.5 * ms  # Guzman 2016 ("needed for temporal summation of EPSPs") 
decay_PC_MF = 5.4 * ms  # Vyleta ... Jonas 2016
decay_PC_I = 3.3 * ms  # Bartos 2002
decay_BC_E = 4.1 * ms  # Lee 2014 (data from CA1)
decay_BC_I = 1.2 * ms  # Bartos 2002
# Normalization factors (normalize the peak of the PSC curve to 1)
norm_PC_E = (decay_PC_E / rise_PC_E) ** (rise_PC_E / (decay_PC_E - rise_PC_E))
norm_PC_MF = (decay_PC_MF / rise_PC_MF) ** (rise_PC_MF / (decay_PC_MF - rise_PC_MF))
norm_PC_I = (decay_PC_I / rise_PC_I) ** (rise_PC_I / (decay_PC_I - rise_PC_I))
norm_BC_E = (decay_BC_E / rise_BC_E) ** (rise_BC_E / (decay_BC_E - rise_BC_E))
norm_BC_I = (decay_BC_I / rise_BC_I) ** (rise_BC_I / (decay_BC_I - rise_BC_I))

# synaptic delays:
delay_PC_E = 2.2 * ms  # Guzman 2016
delay_PC_I = 1.1 * ms  # Bartos 2002
delay_BC_E = 0.9 * ms  # Geiger 1997 (data from DG)
delay_BC_I = 0.6 * ms  # Bartos 2002
        
# synaptic reversal potentials
Erev_exc = 0.0 * mV
Erev_inh = -70.0 * mV

# mossy fiber input
rate_MF = 18.5 * Hz

z = 1 * nS
# parameters for PCs (optimized by Bence)
g_leak_PC = 4.49581428461e-3 * uS
tau_mem_PC = 37.97630516 * ms
Cm_PC = tau_mem_PC * g_leak_PC
Vrest_PC = -59.710040237 * mV
Vreset_PC = -24.8988661181 * mV
theta_PC = -13.3139788756 * mV
tref_PC = 3.79313737057 * ms
delta_T_PC = 3.31719795927 * mV
spike_th_PC = theta_PC + 10 * delta_T_PC
a_PC = -0.255945300382 * nS
b_PC = 0.22030375858 * nA
tau_w_PC = 80.1747780694 * ms

# parameters for BCs (optimized by Bence)
g_leak_BC = 7.0102757369e-3 * uS
tau_mem_BC = 37.7598232668 * ms
Cm_BC = tau_mem_BC * g_leak_BC
Vrest_BC = -58.9682231705 * mV
Vreset_BC = -39.1229822301 * mV
theta_BC = -39.5972788689 * mV
tref_BC = 1.06976577195 * ms
delta_T_BC = 2.21103724225 * mV
spike_th_BC = theta_BC + 10 * delta_T_BC


eqs_PC = """
dvm/dt = (-g_leak_PC*(vm-Vrest_PC) + g_leak_PC*delta_T_PC*exp((vm- theta_PC)/delta_T_PC) - w - ((g_ampa+g_ampaMF)*z*(vm-Erev_exc) + g_gaba*z*(vm-Erev_inh)))/Cm_PC : volt (unless refractory)
dw/dt = (a_PC*(vm- Vrest_PC )-w)/tau_w_PC : amp
dg_ampa/dt = (norm_PC_E * x_ampa - g_ampa) / rise_PC_E : 1
dx_ampa/dt = -x_ampa / decay_PC_E : 1
dg_ampaMF/dt = (norm_PC_MF * x_ampaMF - g_ampaMF) / rise_PC_MF : 1
dx_ampaMF/dt = -x_ampaMF / decay_PC_MF : 1
dg_gaba/dt = (norm_PC_I * x_gaba - g_gaba) / rise_PC_I : 1
dx_gaba/dt = -x_gaba/decay_PC_I : 1
"""

eqs_BC = """
dvm/dt = (-g_leak_BC*(vm-Vrest_BC) + g_leak_BC*delta_T_BC*exp((vm- theta_BC)/delta_T_BC) - (g_ampa*z*(vm-Erev_exc) + g_gaba*z*(vm-Erev_inh)))/Cm_BC : volt (unless refractory)
dg_ampa/dt = (norm_BC_E * x_ampa - g_ampa) / rise_BC_E : 1
dx_ampa/dt = -x_ampa/decay_BC_E : 1
dg_gaba/dt = (norm_BC_I * x_gaba - g_gaba) / rise_BC_I : 1
dx_gaba/dt = -x_gaba/decay_BC_I : 1
"""


def load_wmx(pklf_name):
    """
    Dummy function to load in the excitatory weight matrix and make python clear the memory
    :param pklf_name: file name of the saved weight matrix
    :return: wmx_PC_E: excitatory weight matrix
    """
    
    with open(pklf_name, "rb") as f:
        wmx_PC_E = pickle.load(f)
        
    return wmx_PC_E


def run_simulation(wmx_PC_E, STDP_mode, detailed=True, verbose=True):
    """
    Sets up the network and runs simulation
    :param wmx_PC_E: np.array representing the recurrent excitatory synaptic weight matrix
    :param STDP_mode: symmetric or asymmetric weight matrix flag (used for synapse parameters)
    :param detailed: bool - useage of multi state monitor (for membrane pot and inh. and exc. inputs of some singe cells)
    :param verbose: bool - report status of simulation
    :return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC: brian2 monitors (+ array of selected cells used by multi state monitor) 
    """

    np.random.seed(12345)
    pyrandom.seed(12345)
    
    # synaptic weights
    w_PC_MF = 38
    w_PC_I = 0.25
    if STDP_mode == "asym":        
        w_BC_E = 4.5
        w_BC_I = 6.75
    elif STDP_mode == "sym":
        w_BC_E = 4.45
        w_BC_I = 7.5

    PE = NeuronGroup(nPCs, model=eqs_PC, threshold="vm>spike_th_PC",
                     reset="vm=Vreset_PC; w+=b_PC", refractory=tref_PC, method="exponential_euler")
    PE.vm = Vrest_PC; PE.g_ampa = 0.0; PE.g_ampaMF = 0.0; PE.g_gaba = 0.0
                     
    PI = NeuronGroup(nBCs, model=eqs_BC, threshold="vm>spike_th_BC",
                     reset="vm=Vreset_BC", refractory=tref_BC, method="exponential_euler")    
    PI.vm  = Vrest_BC; PI.g_ampa = 0.0; PI.g_gaba = 0.0

    MF = PoissonGroup(nPCs, rate_MF)
    C_PC_MF = Synapses(MF, PE, on_pre="x_ampaMF+=w_PC_MF")
    C_PC_MF.connect(j='i')
    
    # weight matrix used here
    C_PC_E = Synapses(PE, PE, 'w_exc:1', on_pre='x_ampa+=w_exc')
    nonzero_weights = np.nonzero(wmx_PC_E)    
    C_PC_E.connect(i=nonzero_weights[0], j=nonzero_weights[1])
    C_PC_E.w_exc = wmx_PC_E[nonzero_weights].flatten()
    C_PC_E.delay = delay_PC_E
    del wmx_PC_E

    C_PC_I = Synapses(PE, PI, on_pre='x_ampa+=w_BC_E')
    C_PC_I.connect(p=connection_prob_PC)
    C_PC_I.delay = delay_BC_E

    C_BC_E = Synapses(PI, PE, on_pre='x_gaba+=w_PC_I')
    C_BC_E.connect(p=connection_prob_BC)
    C_BC_E.delay = delay_PC_I

    C_BC_I = Synapses(PI, PI, on_pre='x_gaba+=w_BC_I')
    C_BC_I.connect(p=connection_prob_BC)
    C_BC_I.delay = delay_BC_I

    SM_PC = SpikeMonitor(PE)
    SM_BC = SpikeMonitor(PI)
    RM_PC = PopulationRateMonitor(PE)
    RM_BC = PopulationRateMonitor(PI)
    if detailed:
        selection = np.arange(0, nPCs, 20)  # subset of neurons for recoring variables (could be way less for detailed plots, it's many because of the LFP)
        StateM_PC = StateMonitor(PE, variables=["vm", "w", "g_ampa", "g_ampaMF","g_gaba"], record=selection.tolist(), dt=0.1*ms)
        StateM_BC = StateMonitor(PI, "vm", record=[nBCs/2], dt=0.1*ms)
    
    if verbose:
        run(10000*ms, report="text")
    else:
        run(10000*ms)
    
    if detailed:
        return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC
    else:
        return SM_PC, SM_BC, RM_PC, RM_BC
        
       
def analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, multiplier,
                    detailed=False, selection=None, StateM_PC=None, StateM_BC=None, TFR=False, analyse_LFP=False, verbose=False):
    """
    Analyses results from simulations (see `detect_oscillations.py`)
    :param SM_PC, SM_BC, RM_PC, RM_BC: Brian2 spike and rate monitors of PC and BC populations (see `run_simulation()`)
    :param multiplier: weight matrix multiplier (see `spw_network_wmx_mult.py`)
    :param detailed, selection, StateM_PC, StateM_BC: for more detailed plots
    :param TFR: bool - calculate time freq. repr. (using wavelet analysis)
    :param verbose: bool - printing results
    """

    if SM_PC.num_spikes > 0 and SM_BC.num_spikes > 0:  # check if there is any activity

        spike_times_PC, spiking_neurons_PC, rate_PC, ISI_hist, bin_edges = preprocess_monitors(SM_PC, RM_PC)
        spike_times_BC, spiking_neurons_BC, rate_BC = preprocess_monitors(SM_BC, RM_BC, calc_ISI=False)
        avg_replay_interval = replay(ISI_hist[3:16])  # bins from 150 to 850 (range of interest)
        
        if TFR:
            mean_rate_PC, rate_ac_PC, max_ac_PC, t_max_ac_PC, f_PC, Pxx_PC, coefs_PC, freqs_PC = analyse_rate(rate_PC, TFR=True)
            mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC, coefs_BC, freqs_BC = analyse_rate(rate_BC, TFR=True)
        else:
            mean_rate_PC, rate_ac_PC, max_ac_PC, t_max_ac_PC, f_PC, Pxx_PC = analyse_rate(rate_PC)
            mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC = analyse_rate(rate_BC)
        max_ac_ripple_PC, t_max_ac_ripple_PC, avg_ripple_freq_PC, ripple_power_PC = ripple(rate_ac_PC, f_PC, Pxx_PC)
        max_ac_ripple_BC, t_max_ac_ripple_BC, avg_ripple_freq_BC, ripple_power_BC = ripple(rate_ac_BC, f_BC, Pxx_BC)
        avg_gamma_freq_PC, gamma_power_PC = gamma(f_PC, Pxx_PC)       
        avg_gamma_freq_BC, gamma_power_BC = gamma(f_BC, Pxx_BC)
        
        if analyse_LFP:
            t_LFP, LFP, f_LFP, Pxx_LFP = analyse_estimated_LFP(StateM_PC, selection)
        
        if verbose:
            print "Average replay interval: %.3f"%avg_replay_interval
            print "Mean excitatory rate: %.3f"%mean_rate_PC
            print "Mean inhibitory rate: %.3f"%mean_rate_BC
            print "Average exc. ripple freq: %.3f"%avg_ripple_freq_PC
            print "Exc. ripple power: %.3f"%ripple_power_PC
            print "Average inh. ripple freq: %.3f"%avg_ripple_freq_BC
            print "Inh. ripple power: %.3f"%ripple_power_BC

        plot_raster_ISI(spike_times_PC, spiking_neurons_PC, rate_PC, [ISI_hist, bin_edges], "blue", multiplier_=multiplier)
        if TFR:
            plot_PSD(rate_PC, rate_ac_PC, f_PC, Pxx_PC, "PC_population", "blue", multiplier_=multiplier,
                     TFR=True, coefs=coefs_PC, freqs=freqs_PC)
            plot_PSD(rate_BC, rate_ac_BC, f_BC, Pxx_BC, "BC_population", "green", multiplier_=multiplier,
                     TFR=True, coefs=coefs_BC, freqs=freqs_BC)
        else:
            plot_PSD(rate_PC, rate_ac_PC, f_PC, Pxx_PC, "PC_population", "blue", multiplier_=multiplier)
            plot_PSD(rate_BC, rate_ac_BC, f_BC, Pxx_BC, "BC_population", "green", multiplier_=multiplier)
   
        if detailed:
            subset = plot_zoomed(spike_times_PC, spiking_neurons_PC, rate_PC, "PC_population", "blue", multiplier_=multiplier,
                                 StateM=StateM_PC, selection=selection)
            plot_zoomed(spike_times_BC, spiking_neurons_BC, rate_BC, "BC_population", "green", multiplier_=multiplier,
                        PC_pop=False, StateM=StateM_BC)
            plot_detailed(StateM_PC, subset, multiplier_=multiplier)
        else:
            plot_zoomed(spike_times_PC, spiking_neurons_PC, rate_PC, "PC_population", "blue", multiplier_=multiplier)
            plot_zoomed(spike_times_BC, spiking_neurons_BC, rate_BC, "BC_population", "green", multiplier_=multiplier, PC_pop=False)
            
        if analyse_LFP:
            plot_LFP(t_LFP, LFP, f_LFP, Pxx_LFP, multiplier_=multiplier)
            
    else:
        if verbose:
            print "No activity!"
            
    return [multiplier, avg_replay_interval, mean_rate_PC, mean_rate_BC,
            avg_ripple_freq_PC, ripple_power_PC, avg_ripple_freq_BC, ripple_power_BC,
            avg_gamma_freq_PC, gamma_power_PC, avg_gamma_freq_BC, gamma_power_BC,
            max_ac_PC, max_ac_ripple_PC, max_ac_BC, max_ac_ripple_BC]


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[1]       
    except:
        STDP_mode = "asym"
    assert STDP_mode in ["sym", "asym"]
    
    place_cell_ratio = 0.5
    f_in = "wmx_%s_%.1f.pkl"%(STDP_mode, place_cell_ratio)    
    detailed = True; TFR = False; analyse_LFP = True; verbose = True
    
    if not detailed:
        analyse_LFP = False
        print "Without `detailed` recording LFP can't be estimated..."
        
    pklf_name = os.path.join(base_path, "files", f_in)
    wmx_PC_E = load_wmx(pklf_name) * 1e9  # *1e9 nS conversion
        
    if detailed:
        SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC = run_simulation(wmx_PC_E, STDP_mode, detailed=True)
        _ = analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, multiplier=1,
                            detailed=True, selection=selection, StateM_PC=StateM_PC, StateM_BC=StateM_BC,
                            TFR=TFR, analyse_LFP=analyse_LFP, verbose=verbose)
    else:
        SM_PC, SM_BC, RM_PC, RM_BC = run_simulation(wmx_PC_E, STDP_mode, detailed=False)
        _ = analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, multiplier=1, detailed=False, TFR=TFR, verbose=verbose)

    plt.show()      
                 
                 
