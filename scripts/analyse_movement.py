# -*- coding: utf8 -*-
"""
Analyse movement from maximum-likelihood estimate of position, similar to Pfeiffer and Foster 2015
author: András Ecker last update: 01.2019
"""

import os, pickle
import numpy as np
from detect_oscillations import slice_high_activity
from bayesian_decoding import load_tuning_curves, calc_posterior, extract_binspikecount
from plots import plot_step_sizes


base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])


def load_spikes(pklf_name):
    """
    Loads in saved spikes from simulations
    param pklf_name: name of saved file
    return: spike_times, spiking_neurons, rate
    """

    with open(pklf_name, "rb") as f:
        results = pickle.load(f)
    return results["spike_times"], results["spiking_neurons"], results["rate"]


def ML_est_trajectories(spike_times, spiking_neurons, slice_idx, delta_t, step_size, tuning_curves):
    """
    ML estimate of trajectories similar to Pfeiffer and Foster 2015 (see position decoding in `bayesian_decoding.py`)
    :param spike_times, spiking_neurons: see `load_spikes()`
    :param slice_idx: time idx used to slice out high activity states (see `detect_oscillations.py/slice_high_activity()`)
    :param tuning_curves: dictionary of tuning curves {neuronID: tuning curve} (see `bayesian_decoding.py/load_tuning_curves()`)
    return: ML_trajectories: trajectories calculated as maximum-likelihood estimates of position
    """

    ML_trajectories = []
    for bounds in slice_idx:  # iterate through sustained high activity periods
        lb = bounds[0]; ub = bounds[1]
        idx = np.where((lb <= spike_times) & (spike_times < ub))
        bin_spike_counts = extract_binspikecount(lb, ub, delta_t, step_size, spike_times[idx], spiking_neurons[idx], tuning_curves)

        # decode place of the animal and get ML estimate
        X_posterior = calc_posterior(bin_spike_counts, tuning_curves, delta_t)
        ML_trajectories.append(np.argmax(X_posterior, axis=0))

    return ML_trajectories


def analyse_step_sizes(ML_trajectories):
    """
    Analyse speed of the animal based on trajectories
    :param ML_trajectories: see `ML_est_trajectories()`
    return: speeds: step sizes within trajectories
    """

    step_sizes = []
    for trajectory in ML_trajectories:
        step_sizes.append(np.diff(trajectory))

    return step_sizes


if __name__ == "__main__":

    # 20ms 5ms is used in Pfeiffer and Foster 2015
    delta_t = 20  # ms
    t_incr = 5  # ms

    n_spatial_points = 50
    fig_dir = os.path.join(base_path, "figures", "1.00_replay_det_sym_0.5")

    spatial_points = np.linspace(0, 2*np.pi, n_spatial_points)
    pklf_name = os.path.join(base_path, "files", "PFstarts_0.5_linear.pkl")
    tuning_curves = load_tuning_curves(pklf_name, spatial_points)

    pklf_name = os.path.join(base_path, "files", "sim_spikes.pkl")
    spike_times, spiking_neurons, rate = load_spikes(pklf_name)

    slice_idx = slice_high_activity(rate)
    ML_trajectories = ML_est_trajectories(spike_times, spiking_neurons, slice_idx, delta_t, t_incr, tuning_curves)
    step_sizes = analyse_step_sizes(ML_trajectories)
    for i, step_size in enumerate(step_sizes):
        bounds = slice_idx[i]
        fig_name = os.path.join(fig_dir, "%i-%i_rmovement.png"%(bounds[0], bounds[1]))
        plot_step_sizes(step_size, fig_name, temporal_res=t_incr)
