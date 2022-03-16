# -*- coding: utf8 -*-
"""
Creates AdExpIF PC and BC populations in Brian2, loads in recurrent connection matrix for PC population
runs simulation and checks the dynamics
authors: András Ecker, Bence Bagi, Szabolcs Káli last update: 07.2019
"""

import os
import sys
import numpy as np
import random as pyrandom
from brian2 import *
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt
from helper import load_wmx, preprocess_monitors, generate_cue_spikes,\
                   create_dir, save_vars, save_PSD, save_TFR, save_LFP, save_replay_analysis
from detect_replay import replay_circular, slice_high_activity, replay_linear
from detect_oscillations import analyse_rate, ripple_AC, ripple, gamma, calc_TFR, analyse_estimated_LFP
from plots import plot_raster, plot_posterior_trajectory, plot_PSD, plot_TFR, plot_zoomed, plot_detailed, plot_LFP


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
tp = (decay_PC_E * rise_PC_E)/(decay_PC_E - rise_PC_E) * np.log(decay_PC_E/rise_PC_E)  # time to peak
norm_PC_E = 1.0 / (np.exp(-tp/decay_PC_E) - np.exp(-tp/rise_PC_E))
tp = (decay_PC_MF * rise_PC_MF)/(decay_PC_MF - rise_PC_MF) * np.log(decay_PC_MF/rise_PC_MF)
norm_PC_MF = 1.0 / (np.exp(-tp/decay_PC_MF) - np.exp(-tp/rise_PC_MF))
tp = (decay_PC_I * rise_PC_I)/(decay_PC_I - rise_PC_I) * np.log(decay_PC_I/rise_PC_I)
norm_PC_I = 1.0 / (np.exp(-tp/decay_PC_I) - np.exp(-tp/rise_PC_I))
tp = (decay_BC_E * rise_BC_E)/(decay_BC_E - rise_BC_E) * np.log(decay_BC_E/rise_BC_E)
norm_BC_E = 1.0 / (np.exp(-tp/decay_BC_E) - np.exp(-tp/rise_BC_E))
tp = (decay_BC_I * rise_BC_I)/(decay_BC_I - rise_BC_I) * np.log(decay_BC_I/rise_BC_I)
norm_BC_I = 1.0 / (np.exp(-tp/decay_BC_I) - np.exp(-tp/rise_BC_I))
# synaptic delays:
delay_PC_E = 2.2 * ms  # Guzman 2016
delay_PC_I = 1.1 * ms  # Bartos 2002
delay_BC_E = 0.9 * ms  # Geiger 1997 (data from DG)
delay_BC_I = 0.6 * ms  # Bartos 2002
# synaptic reversal potentials
Erev_E = 0.0 * mV
Erev_I = -70.0 * mV

rate_MF = 15.0 * Hz  # mossy fiber input freq

z = 1 * nS
# AdExpIF parameters for PCs (re-optimized by Szabolcs)
g_leak_PC = 4.31475791937223 * nS
tau_mem_PC = 41.7488927175169 * ms
Cm_PC = tau_mem_PC * g_leak_PC
Vrest_PC = -75.1884554193901 * mV
Vreset_PC = -29.738747396665072 * mV
theta_PC = -24.4255910105977 * mV
tref_PC = 5.96326930945599 * ms
delta_T_PC = 4.2340696257631 * mV
spike_th_PC = theta_PC + 5 * delta_T_PC
a_PC = -0.274347065652738 * nS
b_PC = 206.841448096415 * pA
tau_w_PC = 84.9358017225512 * ms
""" comment this back to run with ExpIF PC model...
# ExpIF parameters for PCs (optimized by Szabolcs)
g_leak_PC = 4.88880734814042 * nS
tau_mem_PC = 70.403501012992 * ms
Cm_PC = tau_mem_PC * g_leak_PC
Vrest_PC = -76.59966923496779 * mV
Vreset_PC = -58.8210432444992 * mV
theta_PC = -28.7739788756 * mV
tref_PC = 1.07004414539699 * ms
delta_T_PC = 10.7807538634886 * mV
spike_th_PC = theta_PC + 5 * delta_T_PC
a_PC = 0. * nS
b_PC = 0. * pA
tau_w_PC = 1 * ms
"""
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

eqs_PC = """
dvm/dt = (-g_leak_PC*(vm-Vrest_PC) + g_leak_PC*delta_T_PC*exp((vm- theta_PC)/delta_T_PC) - w - ((g_ampa+g_ampaMF)*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_PC : volt (unless refractory)
dw/dt = (a_PC*(vm-Vrest_PC) - w) / tau_w_PC : amp
dg_ampa/dt = (x_ampa - g_ampa) / rise_PC_E : 1
dx_ampa/dt = -x_ampa / decay_PC_E : 1
dg_ampaMF/dt = (x_ampaMF - g_ampaMF) / rise_PC_MF : 1
dx_ampaMF/dt = -x_ampaMF / decay_PC_MF : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_PC_I : 1
dx_gaba/dt = -x_gaba/decay_PC_I : 1
"""

eqs_BC = """
dvm/dt = (-g_leak_BC*(vm-Vrest_BC) + g_leak_BC*delta_T_BC*exp((vm- theta_BC)/delta_T_BC) - w - (g_ampa*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_BC : volt (unless refractory)
dw/dt = (a_BC*(vm-Vrest_BC) - w) / tau_w_BC : amp
dg_ampa/dt = (x_ampa - g_ampa) / rise_BC_E : 1
dx_ampa/dt = -x_ampa/decay_BC_E : 1
dg_gaba/dt = (x_gaba - g_gaba) / rise_BC_I : 1
dx_gaba/dt = -x_gaba/decay_BC_I : 1
"""


def run_simulation(wmx_PC_E, STDP_mode, cue, save, seed, verbose=True):
    """
    Sets up the network and runs simulation
    :param wmx_PC_E: np.array representing the recurrent excitatory synaptic weight matrix
    :param STDP_mode: asym/sym STDP mode used for the learning (see `stdp.py`) - here used only to set the weights
    :param cue: if True it adds an other Brian2 `SpikeGeneratorGroup` to stimulate a subpop in the beginning (cued replay)
    :param save: bool flag to save PC spikes after the simulation (used by `bayesian_decoding.py` later)
    :param seed: random seed used for running the simulation
    :param verbose: bool flag to report status of simulation
    :return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC: Brian2 monitors (+ array of selected cells used by multi state monitor)
    """

    np.random.seed(seed)
    pyrandom.seed(seed)

    # synaptic weights (see `/optimization/optimize_network.py`)
    w_PC_I = 0.65  # nS
    w_BC_E = 0.85
    w_BC_I = 5.
    if STDP_mode == "asym":
        w_PC_MF = 21.5
    elif STDP_mode == "sym":
        w_PC_MF = 19.15
    else:
        raise ValueError("STDP_mode has to be either 'sym' or 'asym'!")

    PCs = NeuronGroup(nPCs, model=eqs_PC, threshold="vm>spike_th_PC",
                      reset="vm=Vreset_PC; w+=b_PC", refractory=tref_PC, method="exponential_euler")
    PCs.vm = Vrest_PC; PCs.g_ampa = 0.0; PCs.g_ampaMF = 0.0; PCs.g_gaba = 0.0

    BCs = NeuronGroup(nBCs, model=eqs_BC, threshold="vm>spike_th_BC",
                      reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
    BCs.vm  = Vrest_BC; BCs.g_ampa = 0.0; BCs.g_gaba = 0.0

    MF = PoissonGroup(nPCs, rate_MF)
    C_PC_MF = Synapses(MF, PCs, on_pre="x_ampaMF+=norm_PC_MF*w_PC_MF")
    C_PC_MF.connect(j="i")

    if cue:
        spike_times, spiking_neurons = generate_cue_spikes()
        cue_input = SpikeGeneratorGroup(100, spiking_neurons, spike_times*second)
        # connects at the end of PC pop (...end of track in linear case)
        C_PC_cue = Synapses(cue_input, PCs, on_pre="x_ampaMF+=norm_PC_MF*w_PC_MF")
        C_PC_cue.connect(i=np.arange(0, 100), j=np.arange(7500, 7600))

    # weight matrix used here
    C_PC_E = Synapses(PCs, PCs, "w_exc:1", on_pre="x_ampa+=norm_PC_E*w_exc", delay=delay_PC_E)
    nonzero_weights = np.nonzero(wmx_PC_E)
    C_PC_E.connect(i=nonzero_weights[0], j=nonzero_weights[1])
    C_PC_E.w_exc = wmx_PC_E[nonzero_weights].flatten()
    del wmx_PC_E

    C_PC_I = Synapses(BCs, PCs, on_pre="x_gaba+=norm_PC_I*w_PC_I", delay=delay_PC_I)
    C_PC_I.connect(p=connection_prob_BC)

    C_BC_E = Synapses(PCs, BCs, on_pre="x_ampa+=norm_BC_E*w_BC_E", delay=delay_BC_E)
    C_BC_E.connect(p=connection_prob_PC)

    C_BC_I = Synapses(BCs, BCs, on_pre="x_gaba+=norm_BC_I*w_BC_I", delay=delay_BC_I)
    C_BC_I.connect(p=connection_prob_BC)

    SM_PC = SpikeMonitor(PCs)
    SM_BC = SpikeMonitor(BCs)
    RM_PC = PopulationRateMonitor(PCs)
    RM_BC = PopulationRateMonitor(BCs)

    selection = np.arange(0, nPCs, 20)   # subset of neurons for recoring variables
    StateM_PC = StateMonitor(PCs, variables=["vm", "w", "g_ampa", "g_ampaMF", "g_gaba"], record=selection.tolist(), dt=0.1*ms)
    StateM_BC = StateMonitor(BCs, "vm", record=[nBCs/2], dt=0.1*ms)

    if verbose:
        run(10000*ms, report="text")
    else:
        run(10000*ms)

    if save:
        save_vars(SM_PC, RM_PC, StateM_PC, selection, seed)

    return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC


def analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC, seed,
                    multiplier, linear, pklf_name, dir_name,
                    analyse_replay=True, TFR=True, save=True, verbose=True):
    """
    Analyses results from simulations (see `detect_oscillations.py`)
    :param SM_PC, SM_BC, RM_PC, RM_BC: Brian2 spike and rate monitors of PC and BC populations (see `run_simulation()`)
    :param selection: array of selected cells used by PC multi state monitor
    :param seed: random seed used to run the sim - here used only for saving
    :param multiplier: weight matrix multiplier (see `spw_network_wmx_mult.py`)
    :param linear: bool for linear/circular weight matrix (more advanced replay detection is used in linear case)
    :param pklf_name: file name of saved place fileds used for replay detection in the linear case
    :param dir_name: subdirectory name used to save replay detection (and optionally TFR) figures in linear case
    :param analyse_replay: bool for analysing replay or not
    :param TFR: bool for calculating time freq. repr. (using wavelet analysis) or not
    :param save: bool for saving results
    :param verbose: bool for printing results or not
    """

    if SM_PC.num_spikes > 0 and SM_BC.num_spikes > 0:  # check if there is any activity

        spike_times_PC, spiking_neurons_PC, rate_PC, ISI_hist_PC, bin_edges_PC = preprocess_monitors(SM_PC, RM_PC)
        spike_times_BC, spiking_neurons_BC, rate_BC = preprocess_monitors(SM_BC, RM_BC, calc_ISI=False)
        if not linear:
            plot_raster(spike_times_PC, spiking_neurons_PC, rate_PC, [ISI_hist_PC, bin_edges_PC], None, "blue",
                        multiplier_=multiplier)
        subset = plot_zoomed(spike_times_PC, spiking_neurons_PC, rate_PC, "PC_population", "blue",
                             multiplier_=multiplier, StateM=StateM_PC, selection=selection)
        plot_zoomed(spike_times_BC, spiking_neurons_BC, rate_BC, "BC_population", "green",
                    multiplier_=multiplier, PC_pop=False, StateM=StateM_BC)
        plot_detailed(StateM_PC, subset, multiplier_=multiplier)

        if not linear:
            slice_idx = []
            replay_ROI = np.where((150 <= bin_edges_PC) & (bin_edges_PC <= 850))
            replay = replay_circular(ISI_hist_PC[replay_ROI]) if analyse_replay else np.nan
        else:
            slice_idx = slice_high_activity(rate_PC, th=2, min_len=260)
            plot_raster(spike_times_PC, spiking_neurons_PC, rate_PC, [ISI_hist_PC, bin_edges_PC], slice_idx, "blue",
                        multiplier_=multiplier)
            if analyse_replay:
                replay, replay_results = replay_linear(spike_times_PC, spiking_neurons_PC, slice_idx, pklf_name, N=30)
            else:
                replay, replay_results = np.nan, {}
            if slice_idx and replay_results != {}:
                create_dir(dir_name)
                for bounds, tmp in replay_results.items():
                    fig_name = os.path.join(dir_name, "%i-%i_replay.png" % (bounds[0], bounds[1]))
                    plot_posterior_trajectory(tmp["X_posterior"], tmp["fitted_path"], tmp["R"], fig_name)
            if save:
                save_replay_analysis(replay, replay_results, seed)

        mean_rate_PC, rate_ac_PC, max_ac_PC, t_max_ac_PC, f_PC, Pxx_PC = analyse_rate(rate_PC, 1000., slice_idx)
        mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC = analyse_rate(rate_BC, 1000., slice_idx)
        plot_PSD(rate_PC, rate_ac_PC, f_PC, Pxx_PC, "PC_population", "blue", multiplier_=multiplier)
        plot_PSD(rate_BC, rate_ac_BC, f_BC, Pxx_BC, "BC_population", "green", multiplier_=multiplier)

        t_LFP, LFP, f_LFP, Pxx_LFP = analyse_estimated_LFP(StateM_PC, selection, slice_idx)
        plot_LFP(t_LFP, LFP, f_LFP, Pxx_LFP, multiplier_=multiplier)

        if save:
            save_LFP(t_LFP, LFP, seed)
            save_PSD(f_PC, Pxx_PC, f_BC, Pxx_BC, f_LFP, Pxx_LFP, seed)

        if TFR:
            coefs_PC, freqs_PC = calc_TFR(rate_PC, 1000., slice_idx)
            coefs_BC, freqs_BC = calc_TFR(rate_BC, 1000., slice_idx)
            coefs_LFP, freqs_LFP = calc_TFR(LFP[::10].copy(), 1000., slice_idx)
            if not linear or not slice_idx:
                plot_TFR(coefs_PC, freqs_PC, "PC_population",
                         os.path.join(base_path, "figures", "%.2f_PC_population_wt.png" % multiplier))
                plot_TFR(coefs_BC, freqs_BC, "BC_population",
                         os.path.join(base_path, "figures", "%.2f_BC_population_wt.png" % multiplier))
                plot_TFR(coefs_LFP, freqs_LFP, "LFP",
                         os.path.join(base_path, "figures", "%.2f_LFP_wt.png" % multiplier))
            else:
                for i, bounds in enumerate(slice_idx):
                    fig_name = os.path.join(dir_name, "%i-%i_PC_population_wt.png" % (bounds[0], bounds[1]))
                    plot_TFR(coefs_PC[i], freqs_PC, "PC_population", fig_name)
                    fig_name = os.path.join(dir_name, "%i-%i_BC_population_wt.png" % (bounds[0], bounds[1]))
                    plot_TFR(coefs_BC[i], freqs_PC, "BC_population", fig_name)
                    fig_name = os.path.join(dir_name, "%i-%i_LFP_wt.png" % (bounds[0], bounds[1]))
                    plot_TFR(coefs_LFP[i], freqs_LFP, "LFP", fig_name)

            if save:
                save_TFR(freqs_PC, coefs_PC, freqs_BC, coefs_BC, freqs_LFP, coefs_LFP, seed)

        max_ac_ripple_PC, t_max_ac_ripple_PC = ripple_AC(rate_ac_PC, slice_idx)
        max_ac_ripple_BC, t_max_ac_ripple_BC = ripple_AC(rate_ac_BC, slice_idx)
        avg_ripple_freq_PC, ripple_power_PC = ripple(f_PC, Pxx_PC, slice_idx)
        avg_ripple_freq_BC, ripple_power_BC = ripple(f_BC, Pxx_BC, slice_idx)
        avg_ripple_freq_LFP, ripple_power_LFP = ripple(f_LFP, Pxx_LFP, slice_idx)
        avg_gamma_freq_PC, gamma_power_PC = gamma(f_PC, Pxx_PC, slice_idx)
        avg_gamma_freq_BC, gamma_power_BC = gamma(f_BC, Pxx_BC, slice_idx)
        avg_gamma_freq_LFP, gamma_power_LFP = gamma(f_LFP, Pxx_LFP, slice_idx)

        if verbose:
            if not np.isnan(replay):
                print("Replay detected!")
            else:
                print("No replay detected...")
            print("Mean excitatory rate: %.3f" % mean_rate_PC)
            print("Mean inhibitory rate: %.3f" % mean_rate_BC)
            print("Average exc. ripple freq: %.3f" % avg_ripple_freq_PC)
            print("Exc. ripple power: %.3f" % ripple_power_PC)
            print("Average inh. ripple freq: %.3f" % avg_ripple_freq_BC)
            print("Inh. ripple power: %.3f" % ripple_power_BC)
            print("Average LFP ripple freq: %.3f" % avg_ripple_freq_LFP)
            print("LFP ripple power: %.3f" % ripple_power_LFP)

        return [multiplier, replay, mean_rate_PC, mean_rate_BC,
                avg_ripple_freq_PC, ripple_power_PC, avg_ripple_freq_BC,
                ripple_power_BC, avg_ripple_freq_LFP, ripple_power_LFP,
                avg_gamma_freq_PC, gamma_power_PC, avg_gamma_freq_BC,
                gamma_power_BC, avg_gamma_freq_LFP, gamma_power_LFP,
                max_ac_PC, max_ac_ripple_PC, max_ac_BC, max_ac_ripple_BC]
    else:
        if verbose:
            print("No activity!")
        return [np.nan for _ in range(20)]


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[1]
    except:
        STDP_mode = "sym"
    assert STDP_mode in ["sym", "asym"]
    linear = True
    cue = False
    place_cell_ratio = 0.5
    seed = 12345

    analyse_replay = True
    TFR = True
    save = False
    verbose = True

    f_in = "wmx_%s_%.1f_linear.pkl"%(STDP_mode, place_cell_ratio) if linear else "wmx_%s_%.1f.pkl" % (STDP_mode, place_cell_ratio)
    PF_pklf_name = os.path.join(base_path, "files", "PFstarts_%s_linear.pkl" % place_cell_ratio) if linear else None
    dir_name = os.path.join(base_path, "figures", "%.2f_replay_det_%s_%.1f" % (1, STDP_mode, place_cell_ratio)) if linear else None

    wmx_PC_E = load_wmx(os.path.join(base_path, "files", f_in)) * 1e9  # *1e9 nS conversion
    SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC = run_simulation(wmx_PC_E, STDP_mode, cue=cue,
                                                                                 save=save, seed=seed, verbose=verbose)
    _ = analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC, seed=seed,
                        multiplier=1, linear=linear, pklf_name=PF_pklf_name, dir_name=dir_name,
                        analyse_replay=analyse_replay, TFR=TFR, save=save, verbose=verbose)
    plt.show()
