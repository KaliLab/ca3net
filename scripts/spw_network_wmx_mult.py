# -*- coding: utf8 -*-
"""
Looped version of `spw_network.py` -> checks the dynamics for different multipliers of the learned weight matrix
author: András Ecker last update: 06.2019
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from plots import plot_summary_replay, plot_summary_ripple, plot_summary_gamma, plot_summary_AC
from helper import load_wmx
from spw_network import run_simulation, analyse_results


base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
header = "multiplier, replay, PC rate, BC rate, " \
         "PC ripple freq, PC ripple power, BC ripple freq, BC ripple power, LFP ripple freq, LFP ripple power, " \
         "PC gamma freq, PC gamma power, BC gamma freq, BC gamma power, LFP gamma freq, LFP gamma power, " \
         "PC max autocorr, PC max ripple range autocorr, BC max autocorr, BC max ripple range autocorr"


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[1]
    except:
        STDP_mode = "sym"
    assert STDP_mode in ["sym", "asym"]

    place_cell_ratio = 0.5
    linear = True
    seed = 12345

    #f_in = "wmx_%s_%.1f_linear.pkl"%(STDP_mode, place_cell_ratio); multipliers = [0.8, 0.85, 0.9, 0.95, 1., 1.05, 1.1, 1.15, 1.2]
    #f_in = "wmx_%s_%.1f_2envs_linear.pkl"%(STDP_mode, place_cell_ratio); multipliers = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.05, 1.1]
    #f_in = "wmx_%s_%.1f_cshuffled_linear.pkl"%(STDP_mode, place_cell_ratio); multipliers = [1., 1.5, 2., 2.5, 3., 3.25, 3.5, 3.75, 4.]
    f_in = "wmx_%s_%.1f_binary_linear.pkl"%(STDP_mode, place_cell_ratio); multipliers = [0.9, 0.95, 1., 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
    print(f_in)
    PF_pklf_name = os.path.join(base_path, "files", "PFstarts_%s_linear.pkl"%place_cell_ratio) if linear else None
    f_out = "%s_%s.txt"%(f_in[4:-4], seed)

    verbose = False; TFR = False

    pklf_name = os.path.join(base_path, "files", f_in)
    wmx_PC_E = load_wmx(pklf_name) * 1e9 # *1e9 nS conversion

    results = np.zeros((len(multipliers), 20))
    for i, multiplier in enumerate(multipliers):
        print("multiplier: %.2f"%multiplier)
        dir_name = os.path.join(base_path, "figures", "%.2f_replay_det_%s_%.1f"%(multiplier, STDP_mode, place_cell_ratio)) if linear else None

        SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC = run_simulation(wmx_PC_E*multiplier, STDP_mode,
                                                                                     cue=False, save=False, seed=seed, verbose=verbose)
        results[i, :] = analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC, seed=seed,
                                        multiplier=multiplier, linear=linear, pklf_name=PF_pklf_name, dir_name=dir_name, TFR=TFR, save=False, verbose=verbose)
        del SM_PC; del SM_BC; del RM_PC; del RM_BC; del StateM_PC; del StateM_BC; plt.close("all")


    f_name = os.path.join(base_path, "files", "results", f_out)
    np.savetxt(f_name, results, fmt="%.3f", delimiter="\t", header=header)

    plot_summary_replay(multipliers, replay=results[:, 1], rates_PC=results[:, 2], rates_BC=results[:, 3])
    plot_summary_ripple(multipliers, ripple_freqs_PC=results[:, 4], ripple_freqs_BC=results[:, 6], ripple_freqs_LFP=results[:, 8],
                        ripple_powers_PC=results[:, 5], ripple_powers_BC=results[:, 7], ripple_powers_LFP=results[:, 9])
    plot_summary_gamma(multipliers, gamma_freqs_PC=results[:, 10], gamma_freqs_BC=results[:, 12], gamma_freqs_LFP=results[:, 14],
                       gamma_powers_PC=results[:, 11], gamma_powers_BC=results[:, 13], gamma_powers_LFP=results[:, 15])
    plot_summary_AC(multipliers, max_acs_PC=results[:, 16], max_acs_BC=results[:, 17],
                    max_acs_ripple_PC=results[:, 18], max_acs_ripple_BC=results[:, 19])
