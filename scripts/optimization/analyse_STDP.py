#!/usr/bin/python
# -*- coding: utf8 -*-
"""
analyse the "effect" of an STDP rule (with the given cell model and synapse parameters)
protocol based on Mishra et al. 2016 - 10.1038/ncomms11552
author: András Ecker last update: 11.2017
"""

import os
import json
from brian2 import *
import numpy as np
import random as pyrandom
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from analyse_EPS import sym_paired_recording, get_peak_EPSP
SWBasePath = os.path.sep.join(os.path.abspath('__file__').split(os.path.sep)[:-3])
sys.path.insert(0, os.path.sep.join([SWBasePath, 'scripts']))
from plots import plot_STDP_rule, plot_learned_EPSPs, plot_compare_STDP_to_orig, plot_STDP2


def sim_pairing_prot(delta_ts, taup, taum, Ap, Am, wmax, w_init):
    """
    Aims to mimic spike pairing induced LTP protocol from Mishra et al. 2016 (300 repetition with different $delta$t-s at 1Hz)
    (Simulated for all different $delta$t-s in the same time, since it's way more effective than one-by-one)
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param delta_ts: list of $delta$t intervals between pre and post spikes (in ms)
    :param taup, taum: time constant of weight change (in ms)
    :param Ap, Am: max amplitude of weight change
    :param wmax: maximum weight (in S)
    :param w_init: initial weights (in S)
    :return: np.array with learned weights (same order as delta_ts)
    """
    
    np.random.seed(12345)
    pyrandom.seed(12345)
          
    pre_spikeTrain = np.arange(5, 305)
    # create 2 numpy arrays for Brian2's SpikeGeneratorGroup
    spikeTimes = pre_spikeTrain
    spikingNrns = 0 * np.ones_like(pre_spikeTrain)
    for i in range(0, len(delta_ts)):     
        post_spikeTrain = pre_spikeTrain + delta_ts[i]/1000.  # /1000 ms conversion
        spikingNrns = np.concatenate((spikingNrns, (i+1)*np.ones_like(post_spikeTrain)), axis=0)        
        spikeTimes = np.concatenate((spikeTimes, post_spikeTrain), axis=0)

    PC = SpikeGeneratorGroup(1+len(delta_ts), spikingNrns, spikeTimes*second)

    # mimics Brian1's exponentialSTPD class, with interactions='all', update='additive'
    # see more on conversion: http://brian2.readthedocs.io/en/stable/introduction/brian1_to_2/synapses.html
    STDP = Synapses(PC, PC,
            """
            w : 1
            dA_pre/dt = -A_pre/taup : 1 (event-driven)
            dA_post/dt = -A_post/taum : 1 (event-driven)
            """,
            on_pre="""
            A_pre += Ap
            w = clip(w + A_post, 0, wmax)
            """,
            on_post="""
            A_post += Am
            w = clip(w + A_pre, 0, wmax)
            """)
             
    STDP.connect(i=0, j=np.arange(1, len(delta_ts)+1).tolist())  # connect pre, to every post
    STDP.w = w_init

    # run simulation
    sm = SpikeMonitor(PC, record=True)
    run(310*second, report="text")
    
    return STDP.w[:]


def get_EPSP_change(delta_ts, dPeakEPSPs):
    """calculate the change (in percantage as in Mishra et al. 2016) of PSP amplitude"""
    
    dEPSPchanges = {"time":[], "change":[]}
    baselineEPSP = dPeakEPSPs["baseline"]
    for delta_t in delta_ts:
        peakEPSP = dPeakEPSPs[delta_t]
        dEPSPchanges["time"].append(delta_t)
        dEPSPchanges["change"].append(((peakEPSP*100.) / baselineEPSP) - 100.)  # percentage change from 100% 
        
    return dEPSPchanges


def _f_exponential(x, A, tau):
    """helper function to `scipy.optimize.curve_fit`"""
    
    return A * np.exp(-x/tau)
  

def fit_exponential(delta_ts, dEPSPchanges, A_0, tau_0):
    """
    fit exponential curve to simulated EPSP changes (to compare with the STDP rule later)
    same form as STDP rule: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    """
    
    # preprocess data
    if len(dEPSPchanges) == 2:  # simulated values
        xdata = []; ydata = [];
        delta_ts = dEPSPchanges["time"]; changes = dEPSPchanges["change"]
        for delta_t, change in zip(delta_ts, changes):
            if delta_t > 0:
                xdata.append(delta_t); ydata.append(change)
    
    else:  # the dict is the dOrigData () -> in vitro values
        xdata = []; ydatap = []; ydatam = [];
        delta_ts = dEPSPchanges["time(ms)"]; changes = dEPSPchanges["mean(%)"]
        for delta_t, change in zip(delta_ts, changes):
            if delta_t > 0:
                xdata.append(delta_t); ydatap.append(change)
            elif delta_t < 0:  # reuse negative valeus (the fitted exponential will be symmetrical any way)
                ydatam.append(change)            
        ydata = [(a+b)/2. for a, b in zip(ydatap, ydatam[::-1])]  # average pos. and neg. values
    
    # optimize parameters
    popt, pcov = curve_fit(_f_exponential, xdata, ydata, p0=(A_0, tau_0), bounds=(0, [np.inf, 100]), method="trf")
    
    return popt



if __name__ == "__main__":
    
    #delta_ts = [-100., -50., -20., -10., 10., 20., 50., 100.]  # (ms) same as Mishra et al. 2016  (large weights gets cropped...)
    delta_ts = [40., 50., 60., 70., 80., 90., 100., 110.]  # (ms) more diverse values to fit better exponential
    
    # STDP parameters
    taup = taum = 62.5 * ms
    Ap = Am = 0.006
    wmax = 10e-9  # S (w is dimensionless in the equations)
    Ap *= wmax  # needed to reproduce Brian1 results
    Am *= wmax  # needed to reproduce Brian1 results
    
    mode_ = plot_STDP_rule(taup/ms, taum/ms, Ap/1e-9, Am/1e-9, "STDP_rule")
    
    # w:0.54 as mean value from Guzman 2016 (from Kali - Guzmann e-mail: 0.52...) -> but that's after learning...
    w_init = 0.1e-9  # S (w is dimensionless in the equations)
        
    # baseline EPS
    v_hold = -70  # mV
    i_hold = -43.638  # pA (calculated by `clamp_cell.py`)
    t_, EPSP, _ = sym_paired_recording(w_init*1e9, i_hold)  # * 1e9 nS conversion
    peakEPSP = get_peak_EPSP(t_, EPSP, i_hold, v_hold)
    dEPSPs = {"t":t_, "baseline":EPSP}; dPeakEPSPs = {"baseline":peakEPSP}
    
    # apply spike pairing LTP protocol     
    weights = sim_pairing_prot(delta_ts, taup, taum, Ap, Am, wmax, w_init)
    
    # EPSPs after learning
    max_peakEPSP = 0  # later used as initial guess of A, during fitting the exponential
    for delta_t, weight in zip(delta_ts, weights):
        t_, EPSP, _ = sym_paired_recording(weight*1e9, i_hold)  # get trace  * 1e9 nS conversion
        peakEPSP = get_peak_EPSP(t_, EPSP, i_hold, v_hold)  # extract peak EPSP
        dEPSPs[delta_t] = EPSP; dPeakEPSPs[delta_t] = peakEPSP  # store results
        max_peakEPSP = peakEPSP if peakEPSP > max_peakEPSP else max_peakEPSP
    
      
    # plot EPSPs after learning
    plot_learned_EPSPs(delta_ts, dEPSPs, "learned_EPSPs")
        
    dEPSPchanges = get_EPSP_change(delta_ts, dPeakEPSPs)  # get the EPSP changes in percentage
    
    
    # load in in vitro data (kindly provided by Jose Guzman and converted to a simplified .json by us)
    fName = os.path.join(SWBasePath, "files", "original_STDP_data.json")
    with open(fName, "rb") as f_:
        dOrigData = json.load(f_)
    
    # fit exponential in additon to the Gaussian fit
    A, tau = fit_exponential(delta_ts, dOrigData, A_0=1.5, tau_0=70)  # A_0 and tau_0 is setted by hand...
    dOrigEFit = {"Ap": A, "Am": A, "taup": tau, "taum": tau}
    
    # fit exp function to the changes of simulated EPSPs
    A, tau = fit_exponential(delta_ts, dEPSPchanges, A_0=max_peakEPSP, tau_0=taup/ms)
    dFittedChanges = {"Ap": A, "Am": A, "taup": tau, "taum": tau}
        
    # plot them together!
    plot_compare_STDP_to_orig(dEPSPchanges, dOrigData, "compare_learning", dOrigEFit=dOrigEFit, dFittedChanges=dFittedChanges)    
    # plot STDP rule and fitted changes together
    dSTDPrule = {"Ap": Ap/1e-9, "Am": Am/1e-9, "taup": taup/ms, "taum": taum/ms}    
    plot_STDP2(dSTDPrule, dFittedChanges, mode_, "compare_STDP")
    
    plt.show()
    


    
