# -*- coding: utf8 -*-
"""
Functions used to analyse sequence replay (see also `bayesian_decoding.py`)
author: András Ecker, last update: 06.2019
"""

import numpy as np
from tqdm import tqdm
from helper import _avg_rate, _get_consecutive_sublists, load_tuning_curves
from bayesian_decoding import extract_binspikecount, calc_posterior, fit_trajectory, test_significance


def slice_high_activity(rate, th, min_len, bin_=20):
    """
    Slices out high network activity - which will be candidates for replay detection
    :param rate: firing rate of the population
    :param th: rate threshold (above which is 'high activity')
    :param min_len: minimum length of continuous high activity (in ms)
    :param bin_: bin size for rate averaging (see `helper/_avg_rate()`)
    """

    assert min_len >= 128, "Spectral analysis won't work on sequences shorter than 128 ms"
    idx = np.where(_avg_rate(rate, bin_) >= th)[0]
    high_act = _get_consecutive_sublists(idx.tolist())
    slice_idx = []
    for tmp in high_act:
        if len(tmp) >= np.floor(min_len/bin_):
            slice_idx.append((tmp[0]*bin_, (tmp[-1]+1)*bin_))
    if not slice_idx:
        print("Sustained high network activity can't be detected"
              "(bin size:%i, min length:%.1f and threshold:%.2f)!" % (bin_, min_len, th))
    return slice_idx


def replay_linear(spike_times, spiking_neurons, slice_idx, pklf_name, N, delta_t=10, t_incr=10, n_spatial_points=50):
    """
    Checks if there is sequence replay, using methods originating from Davison et al. 2009 (see more in `bayesian_decoding.py`)
    :param spike_times, spiking_neurons: see `preprocess_monitors()`
    :param slice_idx: time idx used to slice out high activity states (see `slice_high_activity()`)
    :param pklf_name: filename of saved place fields (used for tuning curves, see `helper.py/load_tuning_curves()`)
    :param N: number of shuffled versions tested (significance test, see `bayesian_decoding/test_significance()`)
    :param delta_t: length of time bins used for decoding (in ms)
    :param t_incr: increment of time bins (see `bayesian_decoding/extract_binspikecount()`)
    :param n_spatial_points: number of spatial points to consider for decoding
    :return: significance: 1/nan for significant/non-significant replay detected
             results: dictinary of stored results
    """

    if slice_idx:
        spatial_points = np.linspace(0, 2*np.pi, n_spatial_points)
        tuning_curves = load_tuning_curves(pklf_name, spatial_points)

        sign_replays = []; results = {}
        for bounds in tqdm(slice_idx, desc="Detecting replay"):  # iterate through sustained high activity periods
            lb = bounds[0]; ub = bounds[1]
            idx = np.where((lb <= spike_times) & (spike_times < ub))
            bin_spike_counts = extract_binspikecount(lb, ub, delta_t, t_incr, spike_times[idx], spiking_neurons[idx],
                                                     tuning_curves)
            # decode place of the animal and try to fit path
            X_posterior = calc_posterior(bin_spike_counts, tuning_curves, delta_t)
            R, fitted_path, _ = fit_trajectory(X_posterior)
            sign, shuffled_Rs = test_significance(bin_spike_counts, tuning_curves, delta_t, R, N)
            sign_replays.append(sign)
            results[bounds] = {"X_posterior":X_posterior, "fitted_path":fitted_path,
                               "R":R, "shuffled_Rs":shuffled_Rs, "significance":sign}
        significance = 1 if not np.isnan(sign_replays).all() else np.nan
        return significance, results
    else:
        return np.nan, {}


def replay_circular(ISI_hist, th=0.7):
    """
    Checks if there is sequence replay in the circular case (simply based on repetition in ISIs)
    :param ISI_hist: inter spike intervals (see `preprocess_monitors()`)
    :param th: threshold for spike count in the highest and 2 nearest bins
    :return: replay: 1/nan for detected/non-detected replay
             avg_replay_interval: average replay interval calculated from ISIs (used only for tuning)
    """

    max_ID = np.argmax(ISI_hist)
    bins_3 = ISI_hist[max_ID-1:max_ID+2] if 1 <= max_ID <= len(ISI_hist)-2 else []
    replay = 1 if sum(int(i) for i in ISI_hist) * th < sum(int(i) for i in bins_3) else np.nan
    # this part is only used for (circular track) optimization...
    bin_means = np.arange(175, 826, 50)  # assumes that ISIs are binned into 20 intervals in `preprocess_monitors()`...
    if 1 <= max_ID <= len(ISI_hist)-2:
        tmp = ISI_hist[max_ID-1]*bin_means[max_ID-1] + ISI_hist[max_ID]*bin_means[max_ID]\
              + ISI_hist[max_ID+1]*bin_means[max_ID+1]
        avg_replay_interval = tmp / (ISI_hist[max_ID-1] + ISI_hist[max_ID] + ISI_hist[max_ID+1])
    elif max_ID == 0:
        avg_replay_interval = 175
    elif max_ID == len(ISI_hist)-1:
        avg_replay_interval = 825

    return replay, avg_replay_interval
