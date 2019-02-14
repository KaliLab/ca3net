# -*- coding: utf8 -*-
"""
Analyse movement from maximum-likelihood estimate of position, similar to Pfeiffer and Foster 2015
author: András Ecker last update: 02.2019
"""

import os, pickle
import numpy as np
import random as pyrandom
from bayesian_decoding import load_tuning_curves, extract_binspikecount, calc_posterior
from detect_oscillations import bandpass_filter
from helper import load_spikes, slice_high_activity, save_step_sizes
from plots import plot_step_sizes, plot_step_size_distr


base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])

len_sim = 10000  # ms


def est_trajectories(spike_times, spiking_neurons, slice_idx, delta_t, step_size, tuning_curves):
    """
    Weighted mean estimate of trajectories similar to Pfeiffer and Foster 2015 (see position decoding in `bayesian_decoding.py`)
    :param spike_times, spiking_neurons: see `helper/load_spikes()`
    :param slice_idx: time idx used to slice out high activity states (see `helper.py/slice_high_activity()`)
    :param tuning_curves: dictionary of tuning curves {neuronID: tuning curve} (see `bayesian_decoding.py/load_tuning_curves()`)
    return: trajectories: trajectories calculated as maximum-likelihood estimates of position
    """

    n_spatial_points = tuning_curves[pyrandom.sample(tuning_curves, 1)[0]].shape[0]

    trajectories = []
    for bounds in slice_idx:  # iterate through sustained high activity periods
        lb = bounds[0]; ub = bounds[1]
        idx = np.where((lb <= spike_times) & (spike_times < ub))
        bin_spike_counts = extract_binspikecount(lb, ub, delta_t, step_size, spike_times[idx], spiking_neurons[idx], tuning_curves)

        # decode place of the animal and get ML estimate
        X_posterior = calc_posterior(bin_spike_counts, tuning_curves, delta_t)
        # calc weighted mean position
        x = np.arange(n_spatial_points)
        trajectories.append(np.dot(x, X_posterior))

    return trajectories


def analyse_step_sizes(trajectories):
    """
    Analyse speed of the animal based on trajectories
    :param trajectories: see `est_trajectories()`
    return: step_sizes: step sizes within trajectories
    """

    step_sizes = []
    for trajectory in trajectories:
        step_size = np.abs(np.diff(trajectory))
        # delete too big steps (probably decoding error)
        step_size[np.where(8. < step_size)] = 0.0
        step_sizes.append(step_size)

    return step_sizes


def filter_rate(rate, slice_idx):
    """
    Filters rate in gamma freq and returns sliced rates
    :param rate: PC pop firing rate
    :param slice_idx: time idx used to slice out high activity states (see `helper.py/slice_high_activity()`)
    :return: gamma_rates: list of sliced gamma filtered rates
    """

    gamma_rate = bandpass_filter(rate)
    t = np.linspace(0, len_sim, len(gamma_rate))
    gamma_rates = []
    for bounds in slice_idx:  # iterate through sustained high activity periods
        lb = bounds[0]; ub = bounds[1]
        idx = np.where((lb <= t) & (t < ub))
        gamma_rates.append(gamma_rate[idx])

    return gamma_rates


def analyse_multiple_seeds(seeds, tuning_curves, delta_t=20, t_incr=5):
    """
    Analyses step size distribution from many sims with different seeds
    :param seeds: list of random seeds used
    :param tuning_curves: see `est_trajectories()`
    :param delta_t, t_incr: time window and increment to use for decoding
    :return: step_sizes_seeds: dict of step sizes corresponding to different seeds
             avg_step_sizes: list of average step sizes (not grouped based on seed)
    """

    avg_step_sizes = []
    step_sizes_seeds = {}
    for seed in seeds:
        pklf_name = os.path.join(base_path, "files", "sim_vars_PC_%i.pkl"%seed)
        spike_times, spiking_neurons, rate = load_spikes(pklf_name)
        slice_idx = slice_high_activity(rate)
        gamma_rates = filter_rate(rate, slice_idx)
        trajectories = est_trajectories(spike_times, spiking_neurons, slice_idx, delta_t, t_incr, tuning_curves)
        step_sizes = analyse_step_sizes(trajectories)
        step_sizes_seeds[seed] = np.hstack(step_sizes)
        for i, step_size in enumerate(step_sizes):
            dist = np.abs(trajectories[i][-1] - trajectories[i][0])
            avg_step_sizes.append(float(dist)/len(step_size))

    return step_sizes_seeds, avg_step_sizes


if __name__ == "__main__":

    # 20ms 5ms is used in Pfeiffer and Foster 2015
    delta_t = 10  # ms
    t_incr = 10  # ms
    n_spatial_points = 50

    fig_dir = os.path.join(base_path, "figures", "1.00_replay_det_sym_0.5")

    spatial_points = np.linspace(0, 2*np.pi, n_spatial_points)
    pklf_name = os.path.join(base_path, "files", "PFstarts_0.5_linear.pkl")
    tuning_curves = load_tuning_curves(pklf_name, spatial_points)

    pklf_name = os.path.join(base_path, "files", "sim_vars_PC.pkl")
    spike_times, spiking_neurons, rate = load_spikes(pklf_name)

    slice_idx = slice_high_activity(rate)
    gamma_rates = filter_rate(rate, slice_idx)
    trajectories = est_trajectories(spike_times, spiking_neurons, slice_idx, delta_t, t_incr, tuning_curves)
    step_sizes = analyse_step_sizes(trajectories)

    avg_step_sizes = []
    for i, step_size in enumerate(step_sizes):
        bounds = slice_idx[i]
        dist = np.abs(trajectories[i][-1] - trajectories[i][0])
        avg_step_size = float(dist)/len(step_size)
        avg_step_sizes.append(avg_step_size)
        fig_name = os.path.join(fig_dir, "%i-%i_rmovement_wt%i_ti%i.png"%(bounds[0], bounds[1], delta_t, t_incr))
        plot_step_sizes(gamma_rates[i], step_size, avg_step_size, delta_t, fig_name)

    fig_name = os.path.join(fig_dir, "steps_distr.png")
    plot_step_size_distr(np.hstack(step_sizes), np.asarray(avg_step_sizes), fig_name)

    save_step_sizes(trajectories, step_sizes, avg_step_sizes, gamma_rates)


    #seeds = [1, 12, 1234, 12345, 1993]
    #step_sizes_seeds, avg_step_sizes = analyse_multiple_seeds(seeds, tuning_curves, delta_t=delta_t, t_incr=t_incr)
    #results = {"step_sizes":np.hstack(step_sizes_seeds.values()), "avg_step_sizes":np.asarray(avg_step_sizes)}
    #pklf_name = os.path.join(base_path, "files", "grand_avg_step_sizes.pkl")
    #with open(pklf_name, "wb") as f:
    #    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    #fig_name = os.path.join(base_path, "figures", "steps_distr.png")
    #plot_step_size_distr(results["step_sizes"], results["avg_step_sizes"], fig_name)
