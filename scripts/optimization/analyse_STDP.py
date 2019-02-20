# -*- coding: utf8 -*-
"""
Analyse the "effect" of an STDP rule (with the given cell model and synapse parameters)
protocol based on Mishra et al. 2016 - 10.1038/ncomms11552
author: András Ecker last update: 11.2017
"""

import os, json, warnings
import numpy as np
import random as pyrandom
from brian2 import *
prefs.codegen.target = "numpy"
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from analyse_EPS import sym_paired_recording, get_peak_EPSP
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add "scripts" directory to the path (to import modules)
sys.path.insert(0, os.path.sep.join([base_path, "scripts"]))
from plots import plot_STDP_rule, plot_learned_EPSPs, plot_compare_STDP_to_orig, plot_STDP2
warnings.filterwarnings("ignore")


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

    pre_spike_train = np.arange(5, 305)
    # create 2 numpy arrays for Brian2's SpikeGeneratorGroup
    spike_times = pre_spike_train
    spiking_neurons = 0 * np.ones_like(pre_spike_train)
    for i in range(0, len(delta_ts)):
        post_spike_train = pre_spike_train + delta_ts[i]/1000.  # /1000 ms conversion
        spiking_neurons = np.concatenate((spiking_neurons, (i+1)*np.ones_like(post_spike_train)), axis=0)
        spike_times = np.concatenate((spike_times, post_spike_train), axis=0)

    PC = SpikeGeneratorGroup(1+len(delta_ts), spiking_neurons, spike_times*second)

    # mimics Brian1's exponentialSTPD class, with interactions='all', update='additive'
    # see more on conversion: http://brian2.readthedocs.io/en/stable/introduction/brian1_to_2/synapses.html
    STDP = Synapses(PC, PC,
            """
            w : 1
            dA_presyn/dt = -A_presyn/taup : 1 (event-driven)
            dA_postsyn/dt = -A_postsyn/taum : 1 (event-driven)
            """,
            on_pre="""
            A_presyn += Ap
            w = clip(w + A_postsyn, 0, wmax)
            """,
            on_post="""
            A_postsyn += Am
            w = clip(w + A_presyn, 0, wmax)
            """)

    STDP.connect(i=0, j=np.arange(1, len(delta_ts)+1).tolist())  # connect pre, to every post
    STDP.w = w_init

    # run simulation
    sm = SpikeMonitor(PC, record=True)
    run(310*second, report="text")

    return STDP.w[:]


def get_EPSP_change(delta_ts, peak_EPSPs):
    """
    Calculate the change (in percantage as in Mishra et al. 2016) of EPSP amplitudes
    :param delta_ts: delta ts used in the protocol
    :param peak_EPSPs: dict of peak EPSPs at every delta t
    :return: dict of percentages of EPSP changes at every delta t
    """

    EPSP_changes = {"time":[], "change":[]}
    baseline_EPSP = peak_EPSPs["baseline"]
    for delta_t in delta_ts:
        peak_EPSP = peak_EPSPs[delta_t]
        EPSP_changes["time"].append(delta_t)
        EPSP_changes["change"].append(((peak_EPSP*100.) / baseline_EPSP) - 100.)  # percentage change from 100%

    return EPSP_changes


def _f_exponential(x, A, tau):
    """
    Dummy helper function to pass to `scipy.optimize.curve_fit`
    :param x: independent variable
    :param A, tau: normalization and slope of the exponential
    :return: exponential function"""

    return A * np.exp(-x/tau)


def fit_exponential(delta_ts, EPSP_changes, A_0, tau_0):
    """
    Fit exponential curve to simulated EPSP changes (to compare with the STDP rule later)
    same form as STDP rule: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param delta_ts: delta ts used in the protocol
    :param EPSP_changes: dict of percentages returned by `get_EPSP_change()`
    :param A_0, tau_0: initial parameters for curve fitting
    :return: best parameters (of the fitted exponential function)
    """

    # preprocess data
    if len(EPSP_changes) == 2:  # simulated values
        xdata = []; ydata = [];
        delta_ts = EPSP_changes["time"]; changes = EPSP_changes["change"]
        for delta_t, change in zip(delta_ts, changes):
            if delta_t > 0:
                xdata.append(delta_t); ydata.append(change)

    else:  # the dict is the orig_data () -> in vitro values
        xdata = []; ydatap = []; ydatam = [];
        delta_ts = EPSP_changes["time(ms)"]; changes = EPSP_changes["mean(%)"]
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

    # load in in vitro data (kindly provided by Jose Guzman and converted to a simplified .json by us)
    fName = os.path.join(base_path, "files", "original_STDP_data.json")
    with open(fName, "rb") as f_:
        orig_data = json.load(f_)

    # fit exponential in additon to the Gaussian fit
    A, tau = fit_exponential(delta_ts, orig_data, A_0=1.5, tau_0=70)  # A_0 and tau_0 is setted by hand...
    orig_exp_fit = {"Ap": A, "Am": A, "taup": tau, "taum": tau}

    # STDP parameters (hand tuned to match the slope (tau) of the exponential fit above)
    taup = taum = 62.5 * ms
    Ap = Am = 4e-3
    wmax = 1e-8  # S
    Ap *= wmax  # needed to reproduce Brian1 results
    Am *= wmax  # needed to reproduce Brian1 results
    w_init = 1e-10  # S
    plot_STDP_rule(taup/ms, taum/ms, Ap/1e-9, Am/1e-9, "STDP_rule")

    # baseline EPS
    v_hold = -70  # mV
    i_hold = -43.638  # pA (calculated by `clamp_cell.py`)
    t_, EPSP, _ = sym_paired_recording(w_init*1e9, i_hold)  # * 1e9 nS conversion
    peak_EPSP = get_peak_EPSP(t_, EPSP, i_hold, v_hold)
    EPSPs = {"t":t_, "baseline":EPSP}; peak_EPSPs = {"baseline":peak_EPSP}

    # apply spike pairing LTP protocol
    weights = sim_pairing_prot(delta_ts, taup, taum, Ap, Am, wmax, w_init)

    # EPSPs after learning
    max_peak_EPSP = 0  # later used as initial guess of A, during fitting the exponential
    for delta_t, weight in zip(delta_ts, weights):
        t_, EPSP, _ = sym_paired_recording(weight*1e9, i_hold)  # get trace  * 1e9 nS conversion
        peak_EPSP = get_peak_EPSP(t_, EPSP, i_hold, v_hold)  # extract peak EPSP
        EPSPs[delta_t] = EPSP; peak_EPSPs[delta_t] = peak_EPSP  # store results
        max_peak_EPSP = peak_EPSP if peak_EPSP > max_peak_EPSP else max_peak_EPSP
    EPSP_changes = get_EPSP_change(delta_ts, peak_EPSPs)

    A, tau = fit_exponential(delta_ts, EPSP_changes, A_0=max_peak_EPSP, tau_0=taup/ms)
    sim_exp_fit = {"Ap": A, "Am": A, "taup": tau, "taum": tau}


    plot_learned_EPSPs(delta_ts, EPSPs, "learned_EPSPs")
    plot_compare_STDP_to_orig(EPSP_changes, orig_data, "compare_learning", orig_exp_fit=orig_exp_fit, sim_exp_fit=sim_exp_fit)
    STDP_params = {"Ap": Ap/1e-9, "Am": Am/1e-9, "taup": taup/ms, "taum": taum/ms}
    plot_STDP2(STDP_params, sim_exp_fit, "compare_STDP")
    plt.show()
