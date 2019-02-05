# -*- coding: utf8 -*-
"""
Analyse pure BC network with Poisson input (based on PC rate)
-> This is supposed to show that a pure BC network with sufficient external drive can oscillate on ripple freq (see also Schlingloff et al. 2014)
author: András Ecker last update: 02.2019
"""

import os, pickle
import numpy as np
import random as pyrandom
from brian2 import *
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add "scripts" directory to the path (to import modules)
sys.path.insert(0, os.path.sep.join([base_path, "scripts"]))
from detect_oscillations import preprocess_monitors, analyse_rate, ripple, gamma
from plots import plot_PSD, plot_zoomed, plot_summary_BC


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
#decay_BC_I = 1.2 * ms  # Bartos 2002
# Normalization factors (normalize the peak of the PSC curve to 1)
tp = (decay_BC_E * rise_BC_E)/(decay_BC_E - rise_BC_E) * np.log(decay_BC_E/rise_BC_E)  # time to peak
norm_BC_E = 1.0 / (np.exp(-tp/decay_BC_E) - np.exp(-tp/rise_BC_E))
#tp = (decay_BC_I * rise_BC_I)/(decay_BC_I - rise_BC_I) * np.log(decay_BC_I/rise_BC_I)
#norm_BC_I = 1.0 / (np.exp(-tp/decay_BC_I) - np.exp(-tp/rise_BC_I))

# synaptic delays:
delay_BC_E = 0.9 * ms  # Geiger 1997 (data from DG)
#delay_BC_I = 0.6 * ms  # Bartos 2002

# synaptic reversal potentials
Erev_E = 0.0 * mV
Erev_I = -70.0 * mV

# synaptic weights (optimized in /optimization/optimize_network.py by BluePyOpt):
#w_BC_E = 3.75
#w_BC_I = 7.5

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


def run_simulation(exc_rate=2, w_BC_E=3.75, w_BC_I=7.5, delay_BC_I=0.6, decay_BC_I=1.2):
    """
    Sets up the purely inhibitory network with outer input and runs simulation
    :param exc_rate: rate of PC population
    :param w_BC_E: PC-BC synaptic weight
    :param w_BC_I: BC-BC synaptic weight
    :param delay_BC_I: BC-BC synaptic delay
    :param decay_BC_I: BC-BC decay time constant
    :return: Brian2 monitors
    """

    delay_BC_I *= ms
    decay_BC_I *= ms
    tp = (decay_BC_I * rise_BC_I)/(decay_BC_I - rise_BC_I) * np.log(decay_BC_I/rise_BC_I)
    norm_BC_I = 1.0 / (np.exp(-tp/decay_BC_I) - np.exp(-tp/rise_BC_I))
    PC_rate = nPCs * connection_prob_PC * exc_rate * Hz  # calc incoming rate

    np.random.seed(12345)
    pyrandom.seed(12345)

    eqs_BC = """
    dvm/dt = (-g_leak_BC*(vm-Vrest_BC) + g_leak_BC*delta_T_BC*exp((vm- theta_BC)/delta_T_BC) - (g_ampa*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_BC : volt (unless refractory)
    dg_ampa/dt = (x_ampa - g_ampa) / rise_BC_E : 1
    dx_ampa/dt = -x_ampa/decay_BC_E : 1
    dg_gaba/dt = (x_gaba - g_gaba) / rise_BC_I : 1
    dx_gaba/dt = -x_gaba/decay_BC_I : 1
    """

    BCs = NeuronGroup(nBCs, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC", refractory=tref_BC, method="exponential_euler")
    BCs.vm  = Vrest_BC; BCs.g_ampa = 0.0; BCs.g_gaba = 0.0

    outer_input = PoissonGroup(nBCs, PC_rate)

    C_BC_E = Synapses(outer_input, BCs, on_pre='x_ampa+=norm_BC_E*w_BC_E', delay=delay_BC_E)
    C_BC_E.connect(j='i')

    C_BC_I = Synapses(BCs, BCs, on_pre='x_gaba+=norm_BC_I*w_BC_I', delay=delay_BC_I)
    C_BC_I.connect(p=connection_prob_BC)

    SM_BC = SpikeMonitor(BCs)
    RM_BC = PopulationRateMonitor(BCs)
    StateM_BC = StateMonitor(BCs, "vm", record=[nBCs/2], dt=0.1*ms)

    run(10000*ms)

    return SM_BC, RM_BC, StateM_BC


def analyse_results(SM_BC, RM_BC, StateM_BC):
    """
    Analyses results from simulations (see `detect_oscillations.py`)
    :param SM_BC, RM_BC, StateM_BC: Brian2 spike and rate monitors of BC population (see `run_simulation()`)
    """

    if SM_BC.num_spikes > 0:  # check if there is any activity

        spike_times_BC, spiking_neurons_BC, rate_BC = preprocess_monitors(SM_BC, RM_BC, calc_ISI=False)
        mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC = analyse_rate(rate_BC, fs=1000., slice_idx=[])
        avg_ripple_freq_BC, ripple_power_BC = ripple(f_BC, Pxx_BC, slice_idx=[])
        avg_gamma_freq_BC, gamma_power_BC = gamma(f_BC, Pxx_BC, slice_idx=[])

        #print "Mean inhibitory rate: %.3f"%mean_rate_BC
        print "Average inh. ripple freq: %.3f"%avg_ripple_freq_BC
        #print "Inh. ripple power: %.3f"%ripple_power_BC
        #print "Average inh. gamma freq: %.3f"%avg_gamma_freq_BC
        #print "Inh. gamma power: %.3f"%gamma_power_BC

        #plot_PSD(rate_BC, rate_ac_BC, f_BC, Pxx_BC, "BC_population", "green", multiplier_=1)
        #plot_zoomed(spike_times_BC, spiking_neurons_BC, rate_BC, "BC_population", "green", multiplier_=1,
        #            PC_pop=False, StateM=StateM_BC)

        return avg_ripple_freq_BC, ripple_power_BC

    else:

        print "No activity !"
        return np.nan, np.nan


if __name__ == "__main__":

    rate_wE = True
    rate_wI = False
    delay_dacay = False

    exc_rates = np.array([1., 1.25, 1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.])
    ws_BC_I = np.array([7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9])
    ws_BC_E = np.array([3.55, 3.6, 3.65, 3.7, 3.75, 3.8, 3.85, 3.9, 3.95])
    delays_BC_I = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    decays_BC_I = np.array([1., 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4])

    if rate_wE:

        freqs = np.zeros((len(exc_rates), len(ws_BC_E)))
        powers = np.zeros((len(exc_rates), len(ws_BC_E)))
        for k, exc_rate in enumerate(exc_rates):
            for h, w_BC_E in enumerate(ws_BC_E):
                print "rate:%.2f, w:%.2f"%(exc_rate, w_BC_E)
                SM_BC, RM_BC, StateM_BC = run_simulation(exc_rate=exc_rate, w_BC_E=w_BC_E)
                freq, power = analyse_results(SM_BC, RM_BC, StateM_BC)
                freqs[k, h] = freq; powers[k, h] = power
                del SM_BC; del RM_BC; del StateM_BC; plt.close("all")

        results = {"freqs":freqs, "powers":powers}
        pklf_name = os.path.join(base_path, "files", "results", "BC_network_rate_wE.pkl")
        with open(pklf_name, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        xlabel = "PC-BC weight (ns)"; ylabel = "PC population rate (Hz)"
        plot_summary_BC(freqs, powers, xlabel=xlabel, xticklabels=ws_BC_E,
                        ylabel=ylabel, yticklabels=exc_rates, save_name="BC_rate_wE")

    if rate_wI:

        freqs = np.zeros((len(exc_rates), len(ws_BC_I)))
        powers = np.zeros((len(exc_rates), len(ws_BC_I)))
        for k, exc_rate in enumerate(exc_rates):
            for h, w_BC_I in enumerate(ws_BC_I):
                print "rate:%.2f, w:%.2f"%(exc_rate, w_BC_I)
                SM_BC, RM_BC, StateM_BC = run_simulation(exc_rate=exc_rate, w_BC_I=w_BC_I)
                freq, power = analyse_results(SM_BC, RM_BC, StateM_BC)
                freqs[k, h] = freq; powers[k, h] = power
                del SM_BC; del RM_BC; del StateM_BC; plt.close("all")

        results = {"freqs":freqs, "powers":powers}
        pklf_name = os.path.join(base_path, "files", "results", "BC_network_rate_wI.pkl")
        with open(pklf_name, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        xlabel = "BC-BC weight (ns)"; ylabel = "PC population rate (Hz)"
        plot_summary_BC(freqs, powers, xlabel=xlabel, xticklabels=ws_BC_I,
                        ylabel=ylabel, yticklabels=exc_rates, save_name="BC_rate_wI")

    if delay_dacay:

        freqs = np.zeros((len(delays_BC_I), len(decays_BC_I)))
        powers = np.zeros((len(delays_BC_I), len(decays_BC_I)))
        for k, delay_BC_I in enumerate(delays_BC_I):
            for h, decay_BC_I in enumerate(decays_BC_I):
                print "decay:%.2f, delay:%.2f"%(decay_BC_I, delay_BC_I)
                SM_BC, RM_BC, StateM_BC = run_simulation(delay_BC_I=delay_BC_I, decay_BC_I=decay_BC_I)
                freq, power = analyse_results(SM_BC, RM_BC, StateM_BC)
                freqs[k, h] = freq; powers[k, h] = power
                del SM_BC; del RM_BC; del StateM_BC; plt.close("all")

        results = {"freqs":freqs, "powers":powers}
        pklf_name = os.path.join(base_path, "files", "results", "BC_network_delay_decay.pkl")
        with open(pklf_name, "wb") as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        xlabel = "BC-BC synaptic decay time constant (ms)"; ylabel = "BC-BC synaptic delay (ms)"
        plot_summary_BC(freqs, powers, xlabel=xlabel, xticklabels=decays_BC_I,
                        ylabel=ylabel, yticklabels=delays_BC_I, save_name="BC_delay_decay")
