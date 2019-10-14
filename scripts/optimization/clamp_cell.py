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
dvm/dt = (-g_leak_PC*(vm-Vrest_PC) + g_leak_PC*delta_T_PC*exp((vm- theta_PC)/delta_T_PC) - w + I)/Cm_PC : volt (unless refractory)
dw/dt = (a_PC*(vm-Vrest_PC) - w) / tau_w_PC : amp
I : amp
"""

eqs_BC = """
dvm/dt = (-g_leak_BC*(vm-Vrest_BC) + g_leak_BC*delta_T_BC*exp((vm- theta_BC)/delta_T_BC) - w + I)/Cm_BC : volt (unless refractory)
dw/dt = (a_BC*(vm-Vrest_BC) - w) / tau_w_BC : amp
I : amp
"""


def _calculate_SS_voltage(i, PC, check_for_spiking=True, plot_=False):
    """
    Calculates steady state voltage for a given current input
    :param i: input current (in pA)
    :param check_for_spiking: bool - if True None is returned if any spike is detected
    :param plot_: bool for plotting voltages
    :return: SS_voltage: steady stage voltage for a given input current
    """

    np.random.seed(12345)
    pyrandom.seed(12345)
    if PC:
        C = NeuronGroup(1, model=eqs_PC, threshold="vm>spike_th_PC",
                        reset="vm=Vreset_PC; w+=b_PC", refractory=tref_PC, method="exponential_euler")
        C.vm = Vrest_PC
        StateM = StateMonitor(C, "vm", record=True, dt=0.1*ms)
    else:
        C = NeuronGroup(1, model=eqs_BC, threshold="vm>spike_th_BC",
                        reset="vm=Vreset_BC; w+=b_BC", refractory=tref_BC, method="exponential_euler")
        C.vm = Vrest_BC
        StateM = StateMonitor(C, "vm", record=True, dt=0.1*ms)

    run(200 * ms)
    C.I = i * pA
    run(800 * ms)

    t = StateM.t_ * 1000  # *1000 ms convertion
    v = StateM[0].vm/mV

    SS_voltage = np.mean(v[np.where((800. < t) & (t < 1000.))])

    if plot_:
        plot_SS_voltage(t, v, SS_voltage, i)

    if check_for_spiking:
        if len(np.nonzero(v[np.where(100.0 < t)] > 0.)[0]) > 0:  # check for voltage crossings
            return None

    return SS_voltage


def holding_current(currents, v_hold, PC=True):
    """
    Finds holding current for a given clamping voltage
    :param currents: list of input currents to use to fit IV curve
    :param v_hold: target voltage
    :return: i_hold: current necessary to clamp the cell at the given voltage
    """

    # get SS voltages for different holding currents
    voltages = []
    for i in currents:
        voltages.append(_calculate_SS_voltage(i, PC, check_for_spiking=False))

    # get an estimate of the root of IV curve
    (m, bis) = np.polyfit(currents, voltages, 1)  # fit a linear line (IV curve)
    center = (v_hold - bis) / m  # center: estimated clamping current (to hold the cell at v_hold)

    def iv(i):
        """
        Local helper function for finding the bisection point of the IV curve
        :param i: holding current (see `_calculate_SS_voltage()`)
        :return: diff between steady stage voltage (for a given current) and desired holding current
        """

        SS_voltage = _calculate_SS_voltage(i, PC)
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

    _calculate_SS_voltage(i_hold, PC, check_for_spiking=True, plot_=True)

    return i_hold

if __name__ == "__main__":

    try:
        v_hold = float(sys.argv[1])
    except:
        v_hold = -70.  # mV
    currents = [25, 50, 100]  # pA

    i_hold = holding_current(currents, v_hold, PC=True)
    print("Holding current for %s mV is %.3f pA"%(v_hold, i_hold))
