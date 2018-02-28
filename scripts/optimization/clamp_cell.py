#!/usr/bin/python
# -*- coding: utf8 -*-
"""
find holding current for specified holding voltage (with the given cell model)
author: András Ecker last update: 11.2017
"""

import os
import sys
from brian2 import *
import numpy as np
import random as pyrandom
from scipy.optimize import bisect
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-3])
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from plots import plot_SS_voltage


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
dvm/dt = (-gL_Pyr*(vm-Vrest_Pyr) + gL_Pyr*delta_T_Pyr*exp((vm- theta_Pyr)/delta_T_Pyr) - w + I)/Cm_Pyr : volt (unless refractory)
dw/dt = (a_Pyr*(vm- Vrest_Pyr )-w)/tau_w_Pyr : amp
I : amp
'''


def calculate_SS_voltage(i, check_for_spiking=True, plot_=False):
    """calculates steady state voltage for a given current input"""
    
    np.random.seed(12345)
    pyrandom.seed(12345)
    
    PE = NeuronGroup(1, model=eqs_Pyr, threshold="vm>v_spike_Pyr",
                     reset="vm=reset_Pyr; w+=b_Pyr", refractory=tref_Pyr, method="exponential_euler")
    PE.vm = Vrest_Pyr
          
    vSM = StateMonitor(PE, 'vm', record=True) 
    
    run(200 * ms)
    PE.I = i * pA
    run(800 * ms)
    
    t = vSM.t_ * 1000  # *1000 ms convertion
    v = vSM[0].vm/mV
        
    SS_voltage = np.mean(v[np.where((800. < t) & (t < 1000.))])
    
    if plot_:
        plot_SS_voltage(t, v, SS_voltage, i)
        
    if check_for_spiking:
        if len(np.nonzero(v[np.where(100.0 < t)] > 0.)[0]) > 0:  # check for voltage crossings
            return None
    
    return SS_voltage


def holding_current(currents, v_hold):
    """finds holding current for a given clamping voltage"""
    
    # get SS voltages for different holding currents
    voltages = []
    for i in currents:
        voltages.append(calculate_SS_voltage(i, check_for_spiking=False))
        
    # get an estimate of the root of IV curve
    (m, bis) = np.polyfit(currents, voltages, 1)  # fit a linear line (IV curve)
    center = (v_hold - bis) / m  # center: estimated clamping current (to hold the cell at v_hold)
    
    def iv(i):
        """local helper function based on calculate_SS_voltage"""
        
        SS_voltage = calculate_SS_voltage(i)
        if SS_voltage is not None:
            return SS_voltage - v_hold
        else:
            return 20.  # return large diff. above v_hold and let the program run instead of throwing an error
            
    # zoom in a bit
    failed = False
    dx = 1.5
    a = center - dx/m; b = center + dx/m
    while iv(a)*iv(b) >= 0:
        dx -= 0.25
        a = center - dx/m; b = center + dx/m
        if dx <= 0.25:
            failed = True
            break
    if failed:
        failed = False
        dx = 1.75
        a = center - dx/m; b = center + dx/m
        while iv(a)*iv(b) >= 0:
            dx += 0.25
            a = center - dx/m; b = center + dx/m
            if dx >= 3.5:
                failed = True
                break

    # get more precise bisection point
    i_hold = bisect(iv, a, b, xtol=0.01, maxiter=100)
    
    # plot for i_hold
    calculate_SS_voltage(i_hold, check_for_spiking=True, plot_=True)

    return i_hold
 
if __name__ == "__main__":
    
    try:
        v_hold = float(sys.argv[1])       
    except:
        v_hold = -70.  # mV
    currents = [-55, -50, -45]  # pA
    
    i_hold = holding_current(currents, v_hold)
    print("Holding current for %s mV is %.3f pA"%(v_hold, i_hold))
    
    
    
    
    
