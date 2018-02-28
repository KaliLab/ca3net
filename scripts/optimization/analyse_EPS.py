#!/usr/bin/python
# -*- coding: utf8 -*-
"""
analyse EPSC & EPSP (with the given cell model and synapse parameters) based on the learned weight matrix
author: András Ecker last update: 11.2017
"""

import os
import sys
from brian2 import *
import numpy as np
import random as pyrandom
import matplotlib.pyplot as plt
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-3])
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from detect_oscillations import load_Wee
from plots import plot_avg_EPS, plot_EPS_dist


# synaptic parameters:
PyrExc_rise = 1.3 * ms  # Gupta 2016 (only from Fig.1 H - 20-80%)
PyrExc_decay = 9.5 * ms  # Gupta 2016 ("needed for temporal summation of EPSPs") 
invpeak_PyrExc = (PyrExc_decay / PyrExc_rise) ** (PyrExc_rise / (PyrExc_decay - PyrExc_rise))
delay_PyrExc = 2.2 * ms  # Gupta 2016
E_Exc = 0.0 * mV

# parameters for pyr cells (optimized by Bence)
z = 1 * nS
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

eqs_Pyr = '''
dvm/dt = (-gL_Pyr*(vm-Vrest_Pyr) + gL_Pyr*delta_T_Pyr*exp((vm- theta_Pyr)/delta_T_Pyr) - w + I - EPSC)/Cm_Pyr : volt (unless refractory)
dw/dt = (a_Pyr*(vm- Vrest_Pyr ) - w)/tau_w_Pyr : amp
dg_ampa/dt = (invpeak_PyrExc * x_ampa - g_ampa) / PyrExc_rise : 1
dx_ampa/dt = -x_ampa / PyrExc_decay : 1
EPSC = g_ampa*z*(vm-E_Exc): amp
I : amp
'''


def sym_paired_recording(weight, i_hold=None):
    """Aims to mimic paired recording of 2 connected PCs: Clamps postsynaptic, deliver spikes from presynaptic and measure EPSP, EPSC"""
    
    np.random.seed(12345)
    pyrandom.seed(12345)
    
    # postsynaptic neuron
    PE = NeuronGroup(1, model=eqs_Pyr, threshold="vm>v_spike_Pyr",
                     reset="vm=reset_Pyr; w+=b_Pyr", refractory=tref_Pyr, method="exponential_euler")
    PE.vm = Vrest_Pyr
    PE.g_ampa = 0
    
    # presynaptic neuron is modelled only as a spike generator
    inp = SpikeGeneratorGroup(1, np.array([0]), np.array([250])*ms)

    Cee = Synapses(inp, PE, 'w_exc:1', on_pre='x_ampa+=w_exc')
    Cee.connect(i=0, j=0)
    Cee.w_exc = weight  # nS
    Cee.delay = delay_PyrExc

    vSM = StateMonitor(PE, 'vm', record=True)
    iSM = StateMonitor(PE, 'EPSC', record=True)
    
    run(10*ms)
    if i_hold:
        PE.I = i_hold * pA  # holding current has to be precalculated
    run(390*ms)
    
    return vSM.t_ * 1000, vSM[0].vm/mV, iSM[0].EPSC/pA  # t, EPSP, EPSC


def get_peak_EPSP(t_, EPSP, i_hold=None, v_hold=None):
    """extracts peak EPSP from simulated traces"""   
    
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
    
    STDP_mode = "sym"
    fIn = "wmxR_%s.txt"%STDP_mode

    wmx = load_Wee(os.path.join(SWBasePath, "files", fIn))
    wmx_nz = wmx[np.nonzero(wmx)]
    print "mean(nonzero weights): %s (nS)"%np.mean(wmx_nz)
    
    weights = np.random.choice(wmx_nz, n, replace=False)
    
    EPSPs = np.zeros((n, 4000))
    EPSCs = np.zeros((n, 4000))
    peakEPSPs = np.zeros(n)
    peakEPSCs = np.zeros(n)    
    for i, weight in enumerate(weights):   
     
        t_, EPSP, EPSC = sym_paired_recording(weight, i_hold)
        
        EPSPs[i,:] = EPSP; EPSCs[i,:] = EPSC
        peakEPSPs[i] = get_peak_EPSP(t_, EPSP, i_hold, v_hold)
        peakEPSCs[i] = np.min(EPSC)
        
        if i % 50 == 0:
            print("%s/%s done..."%(i, n))

    # finall run with the average of all nonzero weights
    t_, EPSP, EPSC = sym_paired_recording(np.mean(wmx_nz), i_hold)

    # Plots
    plot_avg_EPS(t_, EPSPs, EPSP, EPSCs, EPSC, np.mean(wmx_nz), "EPS*_%s"%STDP_mode)
    plot_EPS_dist(peakEPSPs, peakEPSCs, "distEPS*_%s"%STDP_mode)

    plt.show()
    
    
