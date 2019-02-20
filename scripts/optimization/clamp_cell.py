# -*- coding: utf8 -*-
"""
Finds holding current for specified holding voltage (with the given cell model)
author: András Ecker last update: 11.2017
"""

import os, sys
import numpy as np
import random as pyrandom
from brian2 import *
prefs.codegen.target = "numpy"
from scipy.optimize import bisect
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add "scripts" directory to the path (to import modules)
sys.path.insert(0, os.path.sep.join([base_path, "scripts"]))
from plots import plot_SS_voltage


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
dvm/dt = (-g_leak_PC*(vm-Vrest_PC) + g_leak_PC*delta_T_PC*exp((vm- theta_PC)/delta_T_PC) - w + I)/Cm_PC : volt (unless refractory)
dw/dt = (a_PC*(vm- Vrest_PC )-w)/tau_w_PC : amp
I : amp
"""

def _calculate_SS_voltage(i, check_for_spiking=True, plot_=False):
    """
    Calculates steady state voltage for a given current input
    :param i: input current (in pA)
    :param check_for_spiking: bool - if True None is returned if any spike is detected
    :param plot_: bool for plotting voltages
    :return: SS_voltage: steady stage voltage for a given input current
    """

    np.random.seed(12345)
    pyrandom.seed(12345)

    PC = NeuronGroup(1, model=eqs_PC, threshold="vm>spike_th_PC",
                     reset="vm=Vreset_PC; w+=b_PC", refractory=tref_PC, method="exponential_euler")
    PC.vm = Vrest_PC

    StateM_PC = StateMonitor(PC, "vm", record=True, dt=0.1*ms)

    run(200 * ms)
    PC.I = i * pA
    run(800 * ms)

    t = StateM_PC.t_ * 1000  # *1000 ms convertion
    v = StateM_PC[0].vm/mV

    SS_voltage = np.mean(v[np.where((800. < t) & (t < 1000.))])

    if plot_:
        plot_SS_voltage(t, v, SS_voltage, i)

    if check_for_spiking:
        if len(np.nonzero(v[np.where(100.0 < t)] > 0.)[0]) > 0:  # check for voltage crossings
            return None

    return SS_voltage


def holding_current(currents, v_hold):
    """
    Finds holding current for a given clamping voltage
    :param currents: list of input currents to use to fit IV curve
    :param v_hold: target voltage
    :return: i_hold: current necessary to clamp the cell at the given voltage
    """

    # get SS voltages for different holding currents
    voltages = []
    for i in currents:
        voltages.append(_calculate_SS_voltage(i, check_for_spiking=False))

    # get an estimate of the root of IV curve
    (m, bis) = np.polyfit(currents, voltages, 1)  # fit a linear line (IV curve)
    center = (v_hold - bis) / m  # center: estimated clamping current (to hold the cell at v_hold)

    def iv(i):
        """
        Local helper function for finding the bisection point of the IV curve
        :param i: holding current (see `_calculate_SS_voltage()`)
        :return: diff between steady stage voltage (for a given current) and desired holding current
        """

        SS_voltage = _calculate_SS_voltage(i)
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

    _calculate_SS_voltage(i_hold, check_for_spiking=True, plot_=True)

    return i_hold

if __name__ == "__main__":

    try:
        v_hold = float(sys.argv[1])
    except:
        v_hold = -70.  # mV
    currents = [-55, -50, -45]  # pA

    i_hold = holding_current(currents, v_hold)
    print("Holding current for %s mV is %.3f pA"%(v_hold, i_hold))
