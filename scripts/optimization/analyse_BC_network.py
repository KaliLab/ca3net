# -*- coding: utf8 -*-
"""
Analyse pure BC network with Poisson input (based on PC rate)
-> This is supposed to show that a pure BC network with sufficient external drive
can oscillate on ripple frequnecy (see also Schlingloff et al. 2014)
author: András Ecker, last update: 03.2022
"""

import os, pickle
from tqdm import tqdm
import numpy as np
import random as pyrandom
from brian2 import *
prefs.codegen.target = "numpy"
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add "scripts" directory to the path (to import modules)
sys.path.insert(0, os.path.sep.join([base_path, "scripts"]))
from helper import preprocess_monitors
from detect_oscillations import analyse_rate, ripple, gamma
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
decay_BC_I = 1.2 * ms  # Bartos 2002
# Normalization factors (normalize the peak of the PSC curve to 1)
tp = (decay_BC_E * rise_BC_E)/(decay_BC_E - rise_BC_E) * np.log(decay_BC_E/rise_BC_E)  # time to peak
norm_BC_E = 1.0 / (np.exp(-tp/decay_BC_E) - np.exp(-tp/rise_BC_E))
tp = (decay_BC_I * rise_BC_I)/(decay_BC_I - rise_BC_I) * np.log(decay_BC_I/rise_BC_I)
norm_BC_I = 1.0 / (np.exp(-tp/decay_BC_I) - np.exp(-tp/rise_BC_I))

# synaptic delays:
delay_BC_E = 0.9 * ms  # Geiger 1997 (data from DG)
delay_BC_I = 0.6 * ms  # Bartos 2002

# synaptic reversal potentials
Erev_E = 0.0 * mV
Erev_I = -70.0 * mV

z = 1 * nS
# parameters for BCs (re-optimized by Szabolcs)
g_leak_BC = 7.51454086502288 * nS
tau_mem_BC = 15.773412296065 * ms
Cm_BC = tau_mem_BC * g_leak_BC
Vrest_BC = -74.74167987795019 * mV
Vreset_BC = -64.99190523539687 * mV
theta_BC = -57.7092044103536 * mV
tref_BC = 1.15622717832178 * ms
delta_T_BC = 4.58413312063091 * mV
spike_th_BC = theta_BC + 5 * delta_T_BC
a_BC = 3.05640210724374 * nS
b_BC = 0.916098931234532 * pA
tau_w_BC = 178.581099914024 * ms

eqs_BC = """
dvm/dt = (-g_leak_BC*(vm-Vrest_BC) + g_leak_BC*delta_T_BC*exp((vm-theta_BC)/delta_T_BC) - w - (g_ampa*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_BC : volt (unless refractory)
dw/dt = (a_BC*(vm-Vrest_BC) - w)/tau_w_BC : amp
dg_ampa/dt = (x_ampa - g_ampa) / rise_BC_E : 1
dx_ampa/dt = -x_ampa/decay_BC_E : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_BC_I : 1
dx_gaba/dt = -x_gaba/decay_BC_I : 1
"""


def run_simulation(exc_rate, w_BC_E, w_BC_I, report_currents=False):
    """
    Sets up the purely inhibitory network with outer input and runs simulation
    :param exc_rate: rate of PC population
    :param w_BC_E: PC-BC synaptic weight
    :param w_BC_I: BC-BC synaptic weight
    :param report_currents: bool flag to report currents
    :return: Brian2 monitors
    """

    np.random.seed(12345)
    pyrandom.seed(12345)

    BCs = NeuronGroup(nBCs, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
    BCs.vm  = Vrest_BC; BCs.g_ampa = 0.0; BCs.g_gaba = 0.0

    PC_rate = nPCs * connection_prob_PC * exc_rate * Hz  # calc incoming rate
    outer_input = PoissonGroup(nBCs, PC_rate)

    C_BC_E = Synapses(outer_input, BCs, on_pre='x_ampa+=norm_BC_E*w_BC_E', delay=delay_BC_E)
    C_BC_E.connect(j='i')

    C_BC_I = Synapses(BCs, BCs, on_pre='x_gaba+=norm_BC_I*w_BC_I', delay=delay_BC_I)
    C_BC_I.connect(p=connection_prob_BC)

    SM_BC = SpikeMonitor(BCs)
    RM_BC = PopulationRateMonitor(BCs)
    if report_currents:
        StateM_BC = StateMonitor(BCs, variables=["vm", "g_ampa", "g_gaba"], record=[nBCs/2], dt=0.1*ms)
    else:
        StateM_BC = StateMonitor(BCs, "vm", record=[nBCs/2], dt=0.1*ms)

    run(10000*ms)

    return SM_BC, RM_BC, StateM_BC


def analyse_results(SM_BC, RM_BC, StateM_BC, return_freq="ripple", plot=False, analyse_currents=False):
    """
    Analyses results from simulations (see `detect_oscillations.py`)
    :params SM_BC, RM_BC, StateM_BC: Brian2 spike and rate monitors of BC population (see `run_simulation()`)
    :param return_freq: str - "ripple" or "gamma" for frequency range to analyse
    :param plot: bool for plotting or not
    :param analyse_currents: bool for analysing currents or not
    :return: avg. ripple/gamma frequency and ripple/gamma power
    """

    assert return_freq in ["ripple", "gamma"]

    if SM_BC.num_spikes > 0:  # check if there is any activity
        spike_times_BC, spiking_neurons_BC, rate_BC = preprocess_monitors(SM_BC, RM_BC, calc_ISI=False)
        mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC = analyse_rate(rate_BC, fs=1000., slice_idx=[])
        avg_ripple_freq_BC, ripple_power_BC = ripple(f_BC, Pxx_BC, slice_idx=[])
        avg_gamma_freq_BC, gamma_power_BC = gamma(f_BC, Pxx_BC, slice_idx=[])

        if analyse_currents:
            # simplified version of `../helper.py/_estimate_LFP()`
            t = StateM_BC.t_ * 1000.  # *1000 ms conversion
            v = StateM_BC[nBCs/2].vm
            g_exc = StateM_BC[nBCs/2].g_ampa*nS
            i_exc = g_exc * (v - (Erev_E * np.ones_like(v/mV)))  # pA
            g_inh = StateM_BC[nBCs/2].g_gaba*nS
            i_inh = g_inh * (v - (Erev_I * np.ones_like(v/mV)))  # pA
        if plot:
            plot_PSD(rate_BC, rate_ac_BC, f_BC, Pxx_BC, "BC_population", "green", multiplier_=1)
            plot_zoomed(spike_times_BC, spiking_neurons_BC, rate_BC, "BC_population", "green", multiplier_=1,
                        PC_pop=False, StateM=StateM_BC)

        if return_freq == "ripple":
            return avg_ripple_freq_BC, ripple_power_BC
        elif return_freq == "gamma":
            return avg_gamma_freq_BC, gamma_power_BC
    else:
        print("No activity !")
        return np.nan, np.nan


if __name__ == "__main__":
    freq_to_analyse = "ripple"

    if freq_to_analyse == "ripple":  # scan in Fig 8)
        exc_rates = np.array([2.5, 2.75, 3., 3.25, 3.5, 3.75, 4., 4.25, 4.5])
        ws_BC_E = np.array([0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95])
        ws_BC_I = np.array([4.8, 4.85, 4.9, 4.95, 5., 5.05, 5.1, 5.15, 5.2])
    elif freq_to_analyse == "gamma":
        exc_rates = np.array([3., 3.5, 4., 4.5, 5.])
        ws_BC_E = np.array([0.25, 0.3, 0.35, 0.4, 0.45])
        ws_BC_I = np.array([0.5, 1, 1.5, 2., 2.5])

    freqs = np.zeros((len(exc_rates), len(ws_BC_I), len(ws_BC_E)))
    powers = np.zeros_like(freqs)
    for j, exc_rate in enumerate(tqdm(exc_rates)):
        for k, w_BC_E in enumerate(tqdm(ws_BC_E, leave=False)):
            for h, w_BC_I in enumerate(ws_BC_I):
                SM_BC, RM_BC, StateM_BC = run_simulation(exc_rate, w_BC_E, w_BC_I)
                freq, power = analyse_results(SM_BC, RM_BC, StateM_BC, return_freq=freq_to_analyse)
                freqs[j, k, h], powers[j, k, h] = freq, power
                del SM_BC; del RM_BC; del StateM_BC; plt.close("all")

    results = {"freqs": freqs, "powers": powers}
    pklf_name = os.path.join(base_path, "files", "results", "BC_network_scan.pkl")
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    #xlabel = "PC-BC weight (ns)"; ylabel = "PC population rate (Hz)"
    #plot_summary_BC(freqs, powers, xlabel=xlabel, xticklabels=ws_BC_E,
    #                ylabel=ylabel, yticklabels=exc_rates, save_name="BC_rate_wE")

