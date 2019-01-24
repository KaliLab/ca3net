# -*- coding: utf8 -*-
"""
Dummy script to averages weight matrix scaling results run with different random seeds
author: András Ecker, last update: 01.2019
"""

import os, pickle
import numpy as np
import pandas as pd


base_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
header = ["multiplier", "replay", "PC_rate", "BC_rate",
          "PC_ripple_freq", "PC_ripple_power", "BC_ripple_freq", "BC_ripple_power", "LFP_ripple_freq", "LFP_ripple_power",
          "PC_gamma_freq", "PC_gamma_power", "BC_gamma_freq", "BC_gamma_power", "LFP_gamma_freq", "LFP_gamma_power",
          "PC_max_autocorr", "PC_max_ripple_range_autocorr", "BC_max_autocorr", "BC_max_ripple_range_autocorr"]


def average_results(sim_version, seeds):
    """
    Loads in results obtained with different random seeds and averages them
    :param sim_version: tag of sim results used to load in files
    :param seeds: list of random seeds used
    :return: df: pandas dataframe with averaged results (with multipliers as indicies)
    """

    # just to get multipliers... (it's not checked if all versions were run with the same wmx multipliers)
    f_name = os.path.join(base_path, "files", "results", "%s_%s.txt"%(sim_version, seeds[0]))
    results_tmp = np.genfromtxt(f_name, comments='#')
    multipliers = results_tmp[:, 0]
    df = pd.DataFrame(index=multipliers)

    results = {name:np.zeros((len(multipliers), len(seeds))) for name in header[1:]}
    for i, seed in enumerate(seeds):
        f_name = os.path.join(base_path, "files", "results", "%s_%s.txt"%(sim_version, seed))
        results_tmp = np.genfromtxt(f_name, comments='#')
        for j, name in enumerate(header[1:]):
            results[name][:, i] = results_tmp[:, j+1]

    for name in header[1:]:
        df["mean_%s"%name] = np.nanmean(results[name], axis=1)
        df["std_%s"%name] = np.nanstd(results[name], axis=1)

    return df


if __name__ == "__main__":

    sim_version = "sym_0.5_shuffled_linear"
    seeds = [1, 12, 1234, 12345, 1993]

    df = average_results(sim_version, seeds)
    pklf_name = os.path.join(base_path, "files", "results", "%s_avg.pkl"%sim_version)
    df.to_pickle(pklf_name, protocol=pickle.HIGHEST_PROTOCOL)
