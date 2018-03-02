#!/usr/bin/python
# -*- coding: utf8 -*-
'''
runs single simulation (used by optimize_network.py)
authors: Bence Bagi, András Ecker last update: 11.2017
'''

import os
import numpy as np
import random as pyrandom
from brian2 import *
prefs.codegen.target = "cython"  # weave is not multiprocess-safe!
import warnings
warnings.filterwarnings("ignore") # ignore scipy 0.18 sparse matrix warning...


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
# size of populations
NE = 8000
NI = 150
# sparseness
eps_pyr = 0.1
eps_bas = 0.25

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


def run_simulation(Wee, J_PyrInh_, J_BasExc_, J_BasInh_, WeeMult_, J_PyrMF_, rate_MF_, verbose=False):
    """
    runs single simulation, with specified input parameters and synaptic weights (to be optimized by BluePyOpt)
    :param Wee: weight matrix (this will not be optimized)
    :param J_PyrInh_: weight of inhibitory input to PSc
    :param J_BasExc_: weight of excitatory input to BCs
    :param J_BasInh_: weight of inhibitory input to BCs
    :param WeeMult_: multiplier of Wee values (not the same as in the *_automatized scripts!)
    :param J_PyrMF: weight of outer (MF) excitatory input to PCs
    :param rate_MF: rate of outer (MF) excitatory input to PCs
    .verbose: bool - report during running sims
    """
    
    # synaptic weights (to be optimized...)
    J_PyrInh = J_PyrInh_
    J_BasExc = J_BasExc_
    J_BasInh = J_BasInh_
    Wee = Wee * WeeMult_
    J_PyrMF = J_PyrMF_
    # input freq (to be optimized...)
    rate_MF = rate_MF_ * Hz

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

    np.random.seed(12345)
    pyrandom.seed(12345)

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

    if verbose:
        run(10000*ms,  report="text")
    else:
        run(10000*ms)
    
    return sme, smi, popre, popri
                 
                 
