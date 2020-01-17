# -*- coding: utf8 -*-
"""
Analyse pure BC network with direct current injection
-> This is supposed to show that a pure BC network with sufficient external drive can oscillate on ripple freq (see also Schlingloff et al. 2014)
author: András Ecker last update: 04.2019
"""

import os, pickle
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
nBCs = 150
# sparseness
connection_prob_BC = 0.25

# synaptic time constants:
# rise time constants
rise_BC_I = 0.25 * ms  # Bartos 2002 (20-80%)
# decay time constants
decay_BC_I = 1.2 * ms  # Bartos 2002
# Normalization factors (normalize the peak of the PSC curve to 1)
tp = (decay_BC_I * rise_BC_I)/(decay_BC_I - rise_BC_I) * np.log(decay_BC_I/rise_BC_I)
norm_BC_I = 1.0 / (np.exp(-tp/decay_BC_I) - np.exp(-tp/rise_BC_I))

# synaptic delays:
delay_BC_I = 0.6 * ms  # Bartos 2002

# synaptic reversal potentials
Erev_I = -70.0 * mV

# synaptic weights (optimized in /optimization/optimize_network.py by BluePyOpt):
w_BC_I = 5.

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


def run_simulation(I_mean=1.3, I_std=0.2, report_currents=False):
    """
    Sets up the purely inhibitory network with direct current injection and runs simulation
    :params I_mean, I_std: mean and std of input current (nA)
    :param report_currents: bool flag to report currents
    :return: Brian2 monitors
    """

    np.random.seed(12345)
    pyrandom.seed(12345)

    eqs_BC = """
    dvm/dt = (-g_leak_BC*(vm-Vrest_BC) + g_leak_BC*delta_T_BC*exp((vm-theta_BC)/delta_T_BC) - w - g_gaba*z*(vm-Erev_I) +I_inj)/Cm_BC : volt (unless refractory)
    dw/dt = (a_BC*(vm-Vrest_BC) - w)/tau_w_BC : amp
    dg_gaba/dt = (x_gaba - g_gaba) / rise_BC_I : 1
    dx_gaba/dt = -x_gaba/decay_BC_I : 1
    I_inj : amp (constant)
    """

    BCs = NeuronGroup(nBCs, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC", refractory=tref_BC, method="exponential_euler")
    BCs.vm  = Vrest_BC; BCs.g_gaba = 0.0; BCs.I_inj = 0.0*nA

    C_BC_I = Synapses(BCs, BCs, on_pre='x_gaba+=norm_BC_I*w_BC_I', delay=delay_BC_I)
    C_BC_I.connect(p=connection_prob_BC)

    SM_BC = SpikeMonitor(BCs)
    RM_BC = PopulationRateMonitor(BCs)
    if report_currents:
        StateM_BC = StateMonitor(BCs, variables=["vm", "g_gaba"], record=[nBCs/2], dt=0.1*ms)
    else:
        StateM_BC = StateMonitor(BCs, "vm", record=[nBCs/2], dt=0.1*ms)

    run(10*ms)
    I_injs = np.random.normal(I_mean, I_std, nBCs)
    if report_currents:
        print "Current injected to the recorded cell: %.2f"%I_injs[nBCs/2]
    BCs.I_inj = I_injs * nA
    run(9980*ms)
    BCs.I_inj = 0*nA
    run(10*ms)

    return SM_BC, RM_BC, StateM_BC


def analyse_results(SM_BC, RM_BC, StateM_BC, analyse_currents=False):
    """
    Analyses results from simulations (see `detect_oscillations.py`)
    :params SM_BC, RM_BC, StateM_BC: Brian2 spike and rate monitors of BC population (see `run_simulation()`)
    :param analyse_currents: bool flag to analyse currents
    :return: avg. ripple frequency and ripple power
    """

    if SM_BC.num_spikes > 0:  # check if there is any activity

        spike_times_BC, spiking_neurons_BC, rate_BC = preprocess_monitors(SM_BC, RM_BC, calc_ISI=False)
        mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC = analyse_rate(rate_BC, fs=1000., slice_idx=[])
        avg_ripple_freq_BC, ripple_power_BC = ripple(f_BC, Pxx_BC, slice_idx=[])
        avg_gamma_freq_BC, gamma_power_BC = gamma(f_BC, Pxx_BC, slice_idx=[])

        print "Average inh. ripple freq: %.3f"%avg_ripple_freq_BC
        print "Inh. ripple power: %.3f"%ripple_power_BC

        if analyse_currents:
            # simplified version of `../helper.py/_estimate_LFP()`
            t = StateM_BC.t_ * 1000.  # *1000 ms conversion
            v = StateM_BC[nBCs/2].vm
            g_inh = StateM_BC[nBCs/2].g_gaba*nS
            i_inh = g_inh * (v - (Erev_I * np.ones_like(v/mV)))  # pA

        plot_PSD(rate_BC, rate_ac_BC, f_BC, Pxx_BC, "BC_population", "green", multiplier_=1)
        plot_zoomed(spike_times_BC, spiking_neurons_BC, rate_BC, "BC_population", "green", multiplier_=1,
                    PC_pop=False, StateM=StateM_BC)

        return avg_ripple_freq_BC, ripple_power_BC

    else:

        print "No activity !"
        return np.nan, np.nan


if __name__ == "__main__":

    I_means = np.array([1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5])
    I_stds = np.array([0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])

    freqs = np.zeros((len(I_means), len(I_stds)))
    powers = np.zeros((len(I_means), len(I_stds)))
    for k, I_mean in enumerate(I_means):
        for h, I_std in enumerate(I_stds):
            print "mean:%.2f, std:%.2f"%(I_mean, I_std)
            SM_BC, RM_BC, StateM_BC = run_simulation(I_mean=I_mean, I_std=I_std)
            freq, power = analyse_results(SM_BC, RM_BC, StateM_BC)
            freqs[k, h] = freq; powers[k, h] = power
            del SM_BC; del RM_BC; del StateM_BC; plt.close("all")

    results = {"freqs":freqs, "powers":powers}
    pklf_name = os.path.join(base_path, "files", "results", "BC_network_I_inj.pkl")
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    xlabel = "std(I_inj) (nS)"; ylabel = "mean(I_inj) (nS)"
    plot_summary_BC(freqs, powers, xlabel=xlabel, xticklabels=I_stds,
                    ylabel=ylabel, yticklabels=I_means, save_name="BC_I_inj")
