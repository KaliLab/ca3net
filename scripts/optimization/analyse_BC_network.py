# -*- coding: utf8 -*-
"""
Analyse pure BC network with Poisson input (based on PC rate)
-> This is supposed to show that a pure BC network with sufficient external drive can oscillate on ripple freq (see also Schlingloff et al. 2014)
author: András Ecker last update: 10.2018
"""

import os
import numpy as np
import random as pyrandom
from brian2 import *
import matplotlib.pyplot as plt
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add "scripts" directory to the path (to import modules)
sys.path.insert(0, os.path.sep.join([base_path, "scripts"]))
from detect_oscillations import preprocess_monitors, analyse_rate, ripple, gamma
from plots import plot_PSD, plot_zoomed


# population size
nPCs = 8000
nBCs = 150
# sparseness
connection_prob_PC = 0.1
connection_prob_BC = 0.25

# synaptic time constants:
# rise time constants
rise_BC_E = 1. * ms  # Lee 2014 (data from CA1) 
rise_BC_I = 0.25 * ms  # Bartos 2002 (20-80%)
# decay time constants
decay_BC_E = 4.1 * ms  # Lee 2014 (data from CA1)
decay_BC_I = 1.2 * ms  # Bartos 2002
# Normalization factors (normalize the peak of the PSC curve to 1)
tmp = (decay_BC_E * rise_BC_E)/(decay_BC_E - rise_BC_E) * np.log(decay_BC_E/rise_BC_E)
norm_BC_E = 1.0 / (np.exp(-tmp/decay_BC_E) - np.exp(-tmp/rise_BC_E))
tmp = (decay_BC_I * rise_BC_I)/(decay_BC_I - rise_BC_I) * np.log(decay_BC_I/rise_BC_I)
norm_BC_I = 1.0 / (np.exp(-tmp/decay_BC_I) - np.exp(-tmp/rise_BC_I))

# synaptic delays:
delay_BC_E = 0.9 * ms  # Geiger 1997 (data from DG)
delay_BC_I = 0.6 * ms  # Bartos 2002
        
# synaptic reversal potentials
Erev_E = 0.0 * mV
Erev_I = -70.0 * mV

z = 1 * nS
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

eqs_BC = """
dvm/dt = (-g_leak_BC*(vm-Vrest_BC) + g_leak_BC*delta_T_BC*exp((vm- theta_BC)/delta_T_BC) - (g_ampa*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_BC : volt (unless refractory)
dg_ampa/dt = (x_ampa - g_ampa) / rise_BC_E : 1
dx_ampa/dt = -x_ampa/decay_BC_E : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_BC_I : 1
dx_gaba/dt = -x_gaba/decay_BC_I : 1
"""


def run_simulation(exc_rate):
    """Sets up the purely inhibitory network with outer input and runs simulation"""

    np.random.seed(12345)
    pyrandom.seed(12345)
    
    BCs = NeuronGroup(nBCs, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC", refractory=tref_BC, method="exponential_euler")    
    BCs.vm  = Vrest_BC; BCs.g_ampa = 0.0; BCs.g_gaba = 0.0
       
    PC_rate = nPCs * connection_prob_PC * exc_rate * Hz  # calc incoming rate
    outer_input = PoissonGroup(nBCs, PC_rate)
    
    C_BC_E = Synapses(outer_input, BCs, on_pre='x_ampa+=norm_BC_E*w_BC_E', delay=delay_BC_E)
    C_BC_E.connect(j='i')
    
    C_BC_I = Synapses(BCs, BCs, on_pre='x_gaba+=norm_BC_I*w_BC_I', delay=delay_BC_I)
    C_BC_I.connect(p=connection_prob_BC)
    
    SM_BC = SpikeMonitor(BCs)
    RM_BC = PopulationRateMonitor(BCs)
    StateM_BC = StateMonitor(BCs, "vm", record=[nBCs/2], dt=0.1*ms)

    run(10000*ms, report="text")
    
    return SM_BC, RM_BC, StateM_BC


def run_simulation_analyse_results(exc_rate):
    """Runs simulation, prints out results and saves plots"""

    SM_BC, RM_BC, StateM_BC = run_simulation(exc_rate)
    
    if SM_BC.num_spikes > 0:  # check if there is any activity

        spike_times_BC, spiking_neurons_BC, rate_BC = preprocess_monitors(SM_BC, RM_BC, calc_ISI=False)
        mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC, coefs_BC, freqs_BC = analyse_rate(rate_BC, fs=1000., slice_idx=[], TFR=True)
        max_ac_ripple_BC, t_max_ac_ripple_BC, avg_ripple_freq_BC, ripple_power_BC = ripple(rate_ac_BC, f_BC, Pxx_BC, slice_idx=[])
        avg_gamma_freq_BC, gamma_power_BC = gamma(f_BC, Pxx_BC, slice_idx=[])
        
        print "Mean inhibitory rate: %.3f"%mean_rate_BC
        print "Average inh. ripple freq: %.3f"%avg_ripple_freq_BC
        print "Inh. ripple power: %.3f"%ripple_power_BC
        print "Average inh. gamma freq: %.3f"%avg_gamma_freq_BC
        print "Inh. gamma power: %.3f"%gamma_power_BC
        
        plot_PSD(rate_BC, rate_ac_BC, f_BC, Pxx_BC, "BC_population", "green", multiplier_=exc_rate)        
        plot_zoomed(spike_times_BC, spiking_neurons_BC, rate_BC, "BC_population", "green", multiplier_=exc_rate,
                    PC_pop=False, StateM=StateM_BC)
        plt.close("all")

    else:

        print "No activity !"
        

if __name__ == "__main__":

    w_BC_E = 4.0
    w_BC_I = 7.0
    exc_rates = [1.75, 2., 2.25, 2.5, 2.75]

    for exc_rate in exc_rates:   
        print "Excitatory rate: %.3f"%exc_rate
        
        run_simulation_analyse_results(exc_rate)
        
        
