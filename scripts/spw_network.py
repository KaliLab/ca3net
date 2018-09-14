# -*- coding: utf8 -*-
"""
Creates PC (adExp IF) and BC (exp IF) population in Brian2, loads in recurrent connection matrix for PC population
runs simulation and checks the dynamics
authors: András Ecker, Bence Bagi, Szabolcs Káli last update: 09.2018
"""

import os, shutil, sys, pickle
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
Erev_E = 0.0 * mV
Erev_I = -70.0 * mV

# mossy fiber input
rate_MF = 16.5 * Hz

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
dvm/dt = (-g_leak_PC*(vm-Vrest_PC) + g_leak_PC*delta_T_PC*exp((vm- theta_PC)/delta_T_PC) - w - ((g_ampa+g_ampaMF)*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_PC : volt (unless refractory)
dw/dt = (a_PC*(vm- Vrest_PC )-w)/tau_w_PC : amp
dg_ampa/dt = (norm_PC_E * x_ampa - g_ampa) / rise_PC_E : 1
dx_ampa/dt = -x_ampa / decay_PC_E : 1
dg_ampaMF/dt = (norm_PC_MF * x_ampaMF - g_ampaMF) / rise_PC_MF : 1
dx_ampaMF/dt = -x_ampaMF / decay_PC_MF : 1
dg_gaba/dt = (norm_PC_I * x_gaba - g_gaba) / rise_PC_I : 1
dx_gaba/dt = -x_gaba/decay_PC_I : 1
"""

eqs_BC = """
dvm/dt = (-g_leak_BC*(vm-Vrest_BC) + g_leak_BC*delta_T_BC*exp((vm- theta_BC)/delta_T_BC) - (g_ampa*z*(vm-Erev_E) + g_gaba*z*(vm-Erev_I)))/Cm_BC : volt (unless refractory)
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


def run_simulation(wmx_PC_E, STDP_mode, detailed, LFP, que, save_spikes, verbose=True):
    """
    Sets up the network and runs simulation
    :param wmx_PC_E: np.array representing the recurrent excitatory synaptic weight matrix
    :param STDP_mode: symmetric or asymmetric weight matrix flag (used for synapse parameters)
    :param detailed: bool flag for useage of multi state monitor (for membrane pot and inh. and exc. inputs of some singe cells)
    :param LFP: similar to `detailed` but more PCs are recorded
    :param que: if True it adds an other Brian2 `SpikeGeneratorGroup` to stimulate a subpop in the beginning (qued replay)
    :param save_spikes: bool flag to save PC spikes after the simulation (used by `bayesian_decoding.py` later)
    :param verbose: bool flag to report status of simulation
    :return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC: Brian2 monitors (+ array of selected cells used by multi state monitor)
    """

    np.random.seed(12345)
    pyrandom.seed(12345)
    
    # synaptic weights
    w_BC_E = 4.5
    if STDP_mode == "asym":      
        w_PC_I = 0.22       
        w_BC_I = 7.15
        w_PC_MF = 39.0
    elif STDP_mode == "sym":
        w_PC_I = 0.23
        w_BC_I = 6.4
        w_PC_MF = 37.0

    PCs = NeuronGroup(nPCs, model=eqs_PC, threshold="vm>spike_th_PC",
                     reset="vm=Vreset_PC; w+=b_PC", refractory=tref_PC, method="exponential_euler")
    PCs.vm = Vrest_PC; PCs.g_ampa = 0.0; PCs.g_ampaMF = 0.0; PCs.g_gaba = 0.0
                     
    BCs = NeuronGroup(nBCs, model=eqs_BC, threshold="vm>spike_th_BC",
                     reset="vm=Vreset_BC", refractory=tref_BC, method="exponential_euler")    
    BCs.vm  = Vrest_BC; BCs.g_ampa = 0.0; BCs.g_gaba = 0.0

    MF = PoissonGroup(nPCs, rate_MF)
    C_PC_MF = Synapses(MF, PCs, on_pre="x_ampaMF+=w_PC_MF")
    C_PC_MF.connect(j='i')
    
    if que:
        # generate short (200ms) Poisson spike train at 20Hz (with `PoissonGroup()` one can't specify the duration)
        from poisson_proc import hom_poisson
        
        spike_times = np.asarray(hom_poisson(20.0, 10, t_max=0.2, seed=12345))
        spiking_neurons = np.zeros_like(spike_times)
        for neuron_id in range(1, 100):
            spike_times_tmp = np.asarray(hom_poisson(20.0, 10, t_max=0.2, seed=12345+neuron_id))
            spike_times = np.concatenate((spike_times, spike_times_tmp), axis=0)
            spiking_neurons_tmp = neuron_id * np.ones_like(spike_times_tmp)
            spiking_neurons = np.concatenate((spiking_neurons, spiking_neurons_tmp), axis=0)           
        que_input = SpikeGeneratorGroup(100, spiking_neurons, spike_times*second)
    
        # connects at the end of PC pop (...end of PFs in linear case)
        C_PC_que = Synapses(que_input, PCs, on_pre="x_ampaMF+=w_PC_MF")
        C_PC_que.connect(i=np.arange(0, 100), j=np.arange(7000, 7100))
    
    # weight matrix used here
    C_PC_E = Synapses(PCs, PCs, 'w_exc:1', on_pre='x_ampa+=w_exc')
    nonzero_weights = np.nonzero(wmx_PC_E)    
    C_PC_E.connect(i=nonzero_weights[0], j=nonzero_weights[1])
    C_PC_E.w_exc = wmx_PC_E[nonzero_weights].flatten()
    C_PC_E.delay = delay_PC_E
    del wmx_PC_E

    C_PC_I = Synapses(PCs, BCs, on_pre='x_ampa+=w_BC_E')
    C_PC_I.connect(p=connection_prob_PC)
    C_PC_I.delay = delay_BC_E

    C_BC_E = Synapses(BCs, PCs, on_pre='x_gaba+=w_PC_I')
    C_BC_E.connect(p=connection_prob_BC)
    C_BC_E.delay = delay_PC_I

    C_BC_I = Synapses(BCs, BCs, on_pre='x_gaba+=w_BC_I')
    C_BC_I.connect(p=connection_prob_BC)
    C_BC_I.delay = delay_BC_I

    SM_PC = SpikeMonitor(PCs)
    SM_BC = SpikeMonitor(BCs)
    RM_PC = PopulationRateMonitor(PCs)
    RM_BC = PopulationRateMonitor(BCs)
    
    if detailed:
        selection = np.arange(0, nPCs, 100)  # subset of neurons for recoring variables 
        if LFP:
            selection = np.arange(0, nPCs, 20)  # record more neurons for better LFP estimate
        StateM_PC = StateMonitor(PCs, variables=["vm", "w", "g_ampa", "g_ampaMF","g_gaba"], record=selection.tolist(), dt=0.1*ms)
        StateM_BC = StateMonitor(BCs, "vm", record=[nBCs/2], dt=0.1*ms)
    
    if verbose:
        run(10000*ms, report="text")
    else:
        run(10000*ms)
        
    if save_spikes:
        spike_times = np.array(SM_PC.t_) * 1000.  # *1000 ms conversion
        spiking_neurons = np.array(SM_PC.i_)
        rate = np.array(RM_PC.rate_).reshape(-1, 10).mean(axis=1)
        f_name = os.path.join(base_path, "files", "sim_spikes_%s.npz"%STDP_mode)
        np.savez(f_name, spike_times=spike_times, spiking_neurons=spiking_neurons, rate=rate)
    
    if detailed:
        return SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC
    else:
        return SM_PC, SM_BC, RM_PC, RM_BC, None, None, None
        
       
def analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, multiplier, linear, pklf_name, dir_name,
                    detailed, selection, StateM_PC, StateM_BC, TFR, analyse_LFP, verbose=True):
    """
    Analyses results from simulations (see `detect_oscillations.py`)
    :param SM_PC, SM_BC, RM_PC, RM_BC: Brian2 spike and rate monitors of PC and BC populations (see `run_simulation()`)
    :param multiplier: weight matrix multiplier (see `spw_network_wmx_mult.py`)
    :param linear: bool for linear/circular weight matrix (more advanced replay detection is used in linear case)
    :param pklf_name: file name of saved place fileds used for replay detection in the linear case
    :param dir_name: subdirectory name used to save replay detection (and optionally TFR) figures in linear case
    :param detailed, selection, StateM_PC, StateM_BC: for more detailed plots
    :param TFR: bool for calculating time freq. repr. (using wavelet analysis) or not
    :param analyse_LFP: bool for analysing LFP or not
    :param verbose: bool for printing results or not
    """

    if SM_PC.num_spikes > 0 and SM_BC.num_spikes > 0:  # check if there is any activity

        spike_times_PC, spiking_neurons_PC, rate_PC, ISI_hist_PC, bin_edges_PC = preprocess_monitors(SM_PC, RM_PC)
        spike_times_BC, spiking_neurons_BC, rate_BC = preprocess_monitors(SM_BC, RM_BC, calc_ISI=False)
        
        if linear:
            if verbose:
                print "Detecting replay..."
            slice_idx = slice_high_activity(rate_PC)
            replay, replay_results = replay_linear(spike_times_PC, spiking_neurons_PC, slice_idx, pklf_name, N=20)
        else:
            slice_idx = []
            replay = replay_circular(ISI_hist_PC[3:16])  # bins from 150 to 850 (range of interest)
        
        if TFR:
            mean_rate_PC, rate_ac_PC, max_ac_PC, t_max_ac_PC, f_PC, Pxx_PC, coefs_PC, freqs_PC = analyse_rate(rate_PC, 1000., slice_idx, TFR=True)
            mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC, coefs_BC, freqs_BC = analyse_rate(rate_BC, 1000., slice_idx, TFR=True)
        else:
            mean_rate_PC, rate_ac_PC, max_ac_PC, t_max_ac_PC, f_PC, Pxx_PC = analyse_rate(rate_PC, 1000., slice_idx, TFR=False)
            mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC = analyse_rate(rate_BC, 1000., slice_idx, TFR=False)
            
        max_ac_ripple_PC, t_max_ac_ripple_PC, avg_ripple_freq_PC, ripple_power_PC = ripple(rate_ac_PC, f_PC, Pxx_PC, slice_idx)
        max_ac_ripple_BC, t_max_ac_ripple_BC, avg_ripple_freq_BC, ripple_power_BC = ripple(rate_ac_BC, f_BC, Pxx_BC, slice_idx)
        avg_gamma_freq_PC, gamma_power_PC = gamma(f_PC, Pxx_PC, slice_idx)       
        avg_gamma_freq_BC, gamma_power_BC = gamma(f_BC, Pxx_BC, slice_idx)
        
        if analyse_LFP:
            t_LFP, LFP, f_LFP, Pxx_LFP = analyse_estimated_LFP(StateM_PC, selection)
        
        if verbose:
            if not np.isnan(replay):
                print "Replay detected!"
            else:
                print "No replay..."
            print "Mean excitatory rate: %.3f"%mean_rate_PC
            print "Mean inhibitory rate: %.3f"%mean_rate_BC
            print "Average exc. ripple freq: %.3f"%avg_ripple_freq_PC
            print "Exc. ripple power: %.3f"%ripple_power_PC
            print "Average inh. ripple freq: %.3f"%avg_ripple_freq_BC
            print "Inh. ripple power: %.3f"%ripple_power_BC

        if linear:
            plot_raster(spike_times_PC, spiking_neurons_PC, rate_PC, [ISI_hist_PC, bin_edges_PC], True, "blue", multiplier_=multiplier)
            if slice_idx:
                if os.path.isdir(dir_name):
                    shutil.rmtree(dir_name)
                    os.mkdir(dir_name)
                else:
                    os.mkdir(dir_name)
                for bounds, tmp in replay_results.iteritems():
                    fig_name = os.path.join(dir_name, "%i-%i_replay.png"%(bounds[0], bounds[1]))
                    plot_posterior_trajectory(tmp["X_posterior"], tmp["fitted_path"], tmp["R"], fig_name)
        else:
            plot_raster(spike_times_PC, spiking_neurons_PC, rate_PC, [ISI_hist_PC, bin_edges_PC], False, "blue", multiplier_=multiplier)
        
        plot_PSD(rate_PC, rate_ac_PC, f_PC, Pxx_PC, "PC_population", "blue", multiplier_=multiplier)
        plot_PSD(rate_BC, rate_ac_BC, f_BC, Pxx_BC, "BC_population", "green", multiplier_=multiplier)
        if TFR:
            if linear:
                if slice_idx:
                    for i, bounds in enumerate(slice_idx):
                        fig_name = os.path.join(dir_name, "%i-%i_PC_population_wt.png"%(bounds[0], bounds[1]))
                        plot_TFR(coefs_PC[i], freqs_PC, "PC_population", fig_name)
                        fig_name = os.path.join(dir_name, "%i-%i_BC_population_wt.png"%(bounds[0], bounds[1]))
                        plot_TFR(coefs_BC[i], freqs_PC, "BC_population", fig_name)
            else:
                plot_TFR(coefs_PC, freqs_PC, "PC_population", os.path.join(base_path, figures, "%.2f_PC_population_wt.png"%multiplier))
                plot_TFR(coefs_BC, freqs_BC, "BC_population", os.path.join(base_path, figures, "%.2f_BC_population_wt.png"%multiplier))
   
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
            
    return [multiplier, replay, mean_rate_PC, mean_rate_BC,
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
    linear = True
    
    f_in = "wmx_%s_%.1f_linear.pkl"%(STDP_mode, place_cell_ratio) if linear else "wmx_%s_%.1f.pkl"%(STDP_mode, place_cell_ratio)
    PF_pklf_name = os.path.join(base_path, "files", "PFstarts_%s_linear.pkl"%place_cell_ratio) if linear else None
    dir_name = os.path.join(base_path, "figures", "%.2f_replay_det_%s_%.1f"%(1, STDP_mode, place_cell_ratio)) if linear else None
    
    que = False; save_spikes = True; verbose = True
    detailed = True; TFR = False; analyse_LFP = False
    if not detailed:
        analyse_LFP = False
        
    pklf_name = os.path.join(base_path, "files", f_in)
    wmx_PC_E = load_wmx(pklf_name) * 1e9  # *1e9 nS conversion
        
    SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC = run_simulation(wmx_PC_E, STDP_mode, detailed=detailed,
                                                                                 LFP=analyse_LFP, que=que, save_spikes=save_spikes, verbose=verbose)
    _ = analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, multiplier=1, linear=linear, pklf_name=PF_pklf_name, dir_name=dir_name,
                        detailed=detailed, selection=selection, StateM_PC=StateM_PC, StateM_BC=StateM_BC,
                        TFR=TFR, analyse_LFP=analyse_LFP, verbose=verbose)

    plt.show()
                 
                 
