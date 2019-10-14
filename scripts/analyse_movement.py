# -*- coding: utf8 -*-
"""
Analyse movement from maximum-likelihood estimate of position, similar to Pfeiffer and Foster 2015
author: András Ecker last update: 02.2019
"""

import os, pickle
import numpy as np
import random as pyrandom
from bayesian_decoding import load_tuning_curves, extract_binspikecount, calc_posterior
from detect_replay import slice_high_activity
from detect_oscillations import bandpass_filter, calc_phase
from helper import load_spikes, load_LFP, argmin_time_arrays, save_step_sizes, save_gavg_step_sizes
from plots import plot_step_sizes, plot_step_size_distr, plot_step_size_phases


base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])

len_sim = 10000  # ms


def _get_gamma_phase(t, LFP, slice_idx):
    """
    Filters LFP in gamma freq, extracts phase from HIlbert transform and returns sliced results
    :param LFP: estimated LFP
    :param slice_idx: time idx used to slice out high activity states (see `detect_replay.py/slice_high_activity()`)
    :return: gamma_filtered_LFPs, phases: dicts with sliced gamma filtered LFPs and corresponding phases
    """

    gamma_filtered_LFP = bandpass_filter(LFP)
    phase = calc_phase(gamma_filtered_LFP)
    t = np.linspace(0, len_sim, len(gamma_filtered_LFP))
    gamma_filtered_LFPs = {}; phases = {}
    for bounds in slice_idx:  # iterate through sustained high activity periods
        lb = bounds[0]; ub = bounds[1]
        idx = np.where((lb <= t) & (t < ub))
        gamma_filtered_LFPs[bounds] = gamma_filtered_LFP[idx]
        phases[bounds] = phase[idx]

    return gamma_filtered_LFPs, phases


def _est_trajectories(spike_times, spiking_neurons, slice_idx, delta_t, step_size, tuning_curves):
    """
    Weighted mean estimate of trajectories similar to Pfeiffer and Foster 2015 (see position decoding in `bayesian_decoding.py`)
    :param spike_times, spiking_neurons: see `helper/load_spikes()`
    :param slice_idx: time idx used to slice out high activity states (see `detect_replay.py/slice_high_activity()`)
    :param tuning_curves: dictionary of tuning curves {neuronID: tuning curve} (see `bayesian_decoding.py/load_tuning_curves()`)
    return: trajectories: trajectories calculated as maximum-likelihood estimates of position
    """

    n_spatial_points = tuning_curves[pyrandom.sample(tuning_curves, 1)[0]].shape[0]

    trajectories = {}
    for bounds in slice_idx:  # iterate through sustained high activity periods
        lb = bounds[0]; ub = bounds[1]
        idx = np.where((lb <= spike_times) & (spike_times < ub))
        bin_spike_counts = extract_binspikecount(lb, ub, delta_t, step_size, spike_times[idx], spiking_neurons[idx], tuning_curves)

        # decode place of the animal and get ML estimate
        X_posterior = calc_posterior(bin_spike_counts, tuning_curves, delta_t)
        # calc weighted mean position
        x = np.arange(n_spatial_points)
        trajectories[bounds] = np.dot(x, X_posterior)

    return trajectories


def _get_step_sizes(trajectories):
    """
    Gets step sizes from decoded trajectories
    :param trajectories: see `_est_trajectories()`
    return: step_sizes: step sizes within trajectories
    """

    step_sizes = {}
    for bounds, trajectory in trajectories.iteritems():
        step_size = np.abs(np.diff(trajectory))
        # delete too big steps (probably decoding error)
        step_size[np.where(8. < step_size)] = 0.0
        step_sizes[bounds] = step_size

    return step_sizes


def _analyse_phase_relationship(step_sizes, phases, delta_t):
    """
    Analyses the relationship between slow gamma phase and step sizes
    :param step_sizes: see `_get_step_sizes()`
    :param phases: see `_get_gamma_phase()`
    :return: all_step_sizes, corresponding_phases: lists of step sizes and corresponding phases
    """

    all_step_sizes = []
    corresponding_phases = []
    for bounds, step_size in step_sizes.iteritems():
        all_step_sizes.extend(step_size)
        len_ = bounds[1]-bounds[0]
        t_step_size = np.linspace(delta_t/2, len_-delta_t/2, len(step_size))
        phase = phases[bounds]
        t_phase = np.linspace(0, len_, len(phase))
        idx = argmin_time_arrays(t_step_size, t_phase)
        corresponding_phases.extend(phase[idx])

    return all_step_sizes, corresponding_phases


def analyse_single_seed(tuning_curves, delta_t, t_incr, fig_dir):
    """
    Analyses step sizes from decoded locations and their phase relation to slow gamma filtered LFP
    :param tuning_curves: place cell tuning curves - see `_est_trajectories()`
    :param delta_t, t_incr: time window and increment to use for decoding
    :param fig_dir: directory to save detailed and summary figures
    """

    pklf_name = os.path.join(base_path, "files", "sim_vars_PC_12345.pkl")
    spike_times, spiking_neurons, rate = load_spikes(pklf_name)
    slice_idx = slice_high_activity(rate, th=2, min_len=260)

    trajectories = _est_trajectories(spike_times, spiking_neurons, slice_idx, delta_t, t_incr, tuning_curves)
    step_sizes = _get_step_sizes(trajectories)

    pklf_name = os.path.join(base_path, "files", "LFP_12345.pkl")
    t, LFP = load_LFP(pklf_name)
    gamma_filtered_LFPs, phases = _get_gamma_phase(t, LFP, slice_idx)

    avg_step_sizes = {}
    for bounds, step_size in step_sizes.iteritems():
        dist = np.abs(trajectories[bounds][-1] - trajectories[bounds][0])
        avg_step_size = float(dist)/len(step_size)
        avg_step_sizes[bounds] = avg_step_size
        fig_name = os.path.join(fig_dir, "%i-%i_rmovement_wt%i_ti%i.png"%(bounds[0], bounds[1], delta_t, t_incr))
        plot_step_sizes(gamma_filtered_LFPs[bounds], step_size, avg_step_size, delta_t, fig_name)

    fig_name = os.path.join(fig_dir, "steps_distr.png")
    plot_step_size_distr(np.hstack(step_sizes.values()), np.asarray(avg_step_sizes.values()), fig_name)

    all_step_sizes, corresponding_phases = _analyse_phase_relationship(step_sizes, phases, delta_t)
    # remove below avg. step sizes and their phases from the grand avg. distributions, to better see large steps...
    idx = np.where(np.asarray(all_step_sizes) > np.mean(avg_step_sizes.values()))[0]
    plot_step_sizes_ = np.asarray(all_step_sizes)[idx]
    plot_corresponding_phases = np.asarray(corresponding_phases)[idx]
    fig_name = os.path.join(fig_dir, "steps_size_phases.png")
    plot_step_size_phases(plot_step_sizes_, plot_corresponding_phases, fig_name)

    save_step_sizes(trajectories, step_sizes, avg_step_sizes, gamma_filtered_LFPs)


def analyse_multiple_seeds(seeds, tuning_curves, delta_t, t_incr):
    """
    Analyses step size distribution and their phase relationship from many sims with different seeds
    :param seeds: list of random seeds used
    :param tuning_curves: place cell tuning curves - see `_est_trajectories()`
    :param delta_t, t_incr: time window and increment to use for decoding
    """

    gavg_step_sizes = []; gall_step_sizes = []; gall_corresponding_phases = []
    for seed in seeds:
        pklf_name = os.path.join(base_path, "files", "sim_vars_PC_%i.pkl"%seed)
        spike_times, spiking_neurons, rate = load_spikes(pklf_name)
        slice_idx = slice_high_activity(rate, th=2, min_len=260)

        trajectories = _est_trajectories(spike_times, spiking_neurons, slice_idx, delta_t, t_incr, tuning_curves)
        step_sizes = _get_step_sizes(trajectories)

        pklf_name = os.path.join(base_path, "files", "LFP_%i.pkl"%seed)
        t, LFP = load_LFP(pklf_name)
        gamma_filtered_LFPs, phases = _get_gamma_phase(t, LFP, slice_idx)

        avg_step_sizes = []
        for bounds, step_size in step_sizes.iteritems():
            dist = np.abs(trajectories[bounds][-1] - trajectories[bounds][0])
            avg_step_sizes.append(float(dist)/len(step_size))
        gavg_step_sizes.extend(avg_step_sizes)

        all_step_sizes, corresponding_phases = _analyse_phase_relationship(step_sizes, phases, delta_t)
        gall_step_sizes.extend(all_step_sizes); gall_corresponding_phases.extend(corresponding_phases)

    fig_name = os.path.join(base_path, "figures", "gavg_steps_distr.png")
    plot_step_size_distr(np.asarray(gall_step_sizes), np.asarray(gavg_step_sizes), fig_name)

    # remove below avg. step sizes and their phases from the grand avg. distributions, to better see large steps...
    idx = np.where(np.asarray(gall_step_sizes) > np.mean(gavg_step_sizes))[0]
    plot_step_sizes_ = np.asarray(gall_step_sizes)[idx]
    plot_corresponding_phases = np.asarray(gall_corresponding_phases)[idx]
    fig_name = os.path.join(base_path, "figures", "gavg_steps_size_phases.png")
    plot_step_size_phases(plot_step_sizes_, plot_corresponding_phases, fig_name)

    save_gavg_step_sizes(gall_step_sizes, gall_corresponding_phases, gavg_step_sizes, seeds)


if __name__ == "__main__":

    # 20ms 5ms is used in Pfeiffer and Foster 2015
    delta_t = 10  # ms
    t_incr = 10  # ms
    n_spatial_points = 50

    spatial_points = np.linspace(0, 2*np.pi, n_spatial_points)
    pklf_name = os.path.join(base_path, "files", "PFstarts_0.5_linear.pkl")
    tuning_curves = load_tuning_curves(pklf_name, spatial_points)

    #fig_dir = os.path.join(base_path, "figures", "1.00_replay_det_sym_0.5")
    #analyse_single_seed(tuning_curves, delta_t, t_incr, fig_dir)

    seeds = [1, 12, 1234, 12345, 1993]
    analyse_multiple_seeds(seeds, tuning_curves, delta_t, t_incr)
