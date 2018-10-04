# -*- coding: utf8 -*-
"""
Analyse EPSC & EPSP (with the given cell model and synapse parameters) based on the learned weight matrix
author: András Ecker last update: 10.2018
"""

import os, sys
import numpy as np
import random as pyrandom
from brian2 import *
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add "scripts" directory to the path (to import modules)
sys.path.insert(0, os.path.sep.join([base_path, "scripts"]))
from spw_network import load_wmx
from plots import plot_avg_EPS, plot_EPS_dist


# synaptic parameters:
rise_PC_E = 1.3 * ms  # Guzman 2016 (only from Fig.1 H - 20-80%)
decay_PC_E = 9.5 * ms  # Guzman 2016 ("needed for temporal summation of EPSPs") 
tp = (decay_PC_E * rise_PC_E)/(decay_PC_E - rise_PC_E) * np.log(decay_PC_E/rise_PC_E)  # time to peak
norm_PC_E = 1.0 / (np.exp(-tp/decay_PC_E) - np.exp(-tp/rise_PC_E))
delay_PC_E = 2.2 * ms  # Guzman 2016
Erev_E = 0.0 * mV

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

eqs_PC = """
dvm/dt = (-g_leak_PC*(vm-Vrest_PC) + g_leak_PC*delta_T_PC*exp((vm- theta_PC)/delta_T_PC) - w + I - EPSC)/Cm_PC : volt (unless refractory)
dw/dt = (a_PC*(vm- Vrest_PC )-w)/tau_w_PC : amp
dg_ampa/dt = (x_ampa - g_ampa) / rise_PC_E : 1
dx_ampa/dt = -x_ampa / decay_PC_E : 1
EPSC = g_ampa*z*(vm-Erev_E): amp
I : amp
"""


def sym_paired_recording(weight, i_hold=None):
    """
    Aims to mimic paired recording of 2 connected PCs: Clamps postsynaptic, deliver spikes from presynaptic and measure EPSP, EPSC
    :param weight: synaptic weight
    :param i_hold: holding current applied to clamp the postsynaptic cell
    :return: time, EPSP and EPSC
    """
    
    np.random.seed(12345)
    pyrandom.seed(12345)
    
    # postsynaptic neuron
    PC = NeuronGroup(1, model=eqs_PC, threshold="vm>spike_th_PC",
                     reset="vm=Vreset_PC; w+=b_PC", refractory=tref_PC, method="exponential_euler")
    PC.vm = Vrest_PC; PC.g_ampa = 0.0
    
    # presynaptic neuron is modelled only as a spike generator
    pre = SpikeGeneratorGroup(1, np.array([0]), np.array([250])*ms)
    
    C_PC_E = Synapses(pre, PC, on_pre="x_ampa+=norm_PC_E*weight", delay=delay_PC_E)    
    C_PC_E.connect(i=0, j=0)
    
    StateM_PC = StateMonitor(PC, variables=["vm", "EPSC"], record=True, dt=0.1*ms)
    
    run(10*ms)
    if i_hold:
        PC.I = i_hold * pA  # holding current (has to be precalculated)
    run(390*ms)
    
    return StateM_PC.t_ * 1000, StateM_PC[0].vm/mV, StateM_PC[0].EPSC/pA  # t, EPSP, EPSC


def get_peak_EPSP(t_, EPSP, i_hold=None, v_hold=None):
    """
    Extracts peak EPSP from simulated traces
    :param t, EPSP: time and EPSP (see `sym_paired_recording()`)
    :param i_hold: holding current
    :param v_hold: holding voltage
    :return: peak EPSP
    """   
    
    if i_hold:
        assert v_hold, "if I_hold is specified, V_hold has to be specified too"
        return np.max(EPSP[np.where((250 < t_) & (t_ < 350))]) - v_hold
    else:
        assert i_hold is None, "If V_hold is specfied, I_hold should be too"
        return np.max(EPSP) - Vrest_Pyr/mV


if __name__ == "__main__":

    try:
        n = int(sys.argv[1])       
    except:
        n = 500  # number of random weights
    
    v_hold = -70.  # mV
    i_hold = -43.638  # pA (calculated by `clamp_cell.py`)
    
    STDP_mode = "asym"
    place_cell_ratio = 0.5
    f_in = "wmx_%s_%.1f.pkl"%(STDP_mode, place_cell_ratio)
    
    pklf_name = os.path.join(base_path, "files", f_in)
    wmx_PC_E = load_wmx(pklf_name) * 1e9  # *1e9 nS conversion

    nonzero_weights = wmx_PC_E[np.nonzero(wmx_PC_E)]
    print "mean(nonzero weights): %s (nS)"%np.mean(nonzero_weights)
    
    weights = np.random.choice(nonzero_weights, n, replace=False)
    
    EPSPs = np.zeros((n, 4000)); EPSCs = np.zeros((n, 4000))  # 4000 is hard coded for sim length    
    peak_EPSPs = np.zeros(n); peak_EPSCs = np.zeros(n)
    for i, weight in enumerate(weights):   
     
        t_, EPSP, EPSC = sym_paired_recording(weight, i_hold)
        
        EPSPs[i,:] = EPSP; EPSCs[i,:] = EPSC
        peak_EPSPs[i] = get_peak_EPSP(t_, EPSP, i_hold, v_hold)
        peak_EPSCs[i] = np.min(EPSC)
        
        if i % 50 == 0:
            print("%s/%s done..."%(i, n))

    # finall run with the average of all nonzero weights
    t_, EPSP, EPSC = sym_paired_recording(np.mean(nonzero_weights), i_hold)

    # Plots
    plot_avg_EPS(t_, EPSPs, EPSP, EPSCs, EPSC, np.mean(nonzero_weights), "EPS*_%s"%STDP_mode)
    plot_EPS_dist(peak_EPSPs, peak_EPSCs, "distEPS*_%s"%STDP_mode)
    plt.show()
    
    
