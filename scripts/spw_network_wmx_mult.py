# -*- coding: utf8 -*-
"""
Looped version of `spw_network.py` -> checks the dynamics for different multipliers of the learned weight matrix
authors: Bence Bagi, András Ecker last update: 08.2018
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from plots import *
from spw_network import load_wmx, run_simulation, analyse_results


base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
header = "multiplier, replay interval, PC rate, BC rate, " \
         "PC ripple freq, PC ripple power, BC ripple freq, BC ripple power, " \
         "PC gamma freq, PC gamma power, BC gamma freq, BC gamma power, " \
         "PC max autocorr, PC max ripple range autocorr, BC max autocorr, BC max ripple range autocorr"


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[1]       
    except:
        STDP_mode = "asym"
    assert STDP_mode in ["sym", "asym"]
    
    place_cell_ratio = 0.5
    detailed = True; TFR = False; analyse_LFP = False
    que = False; save_spikes = False; verbose = False
    
    if not detailed:
        analyse_LFP = False
        print "Without `detailed` recording LFP can't be estimated..."
    
    f_in = "wmx_%s_%.1f.pkl"%(STDP_mode, place_cell_ratio); multipliers = np.arange(0.7, 1.4, 0.1)
    #f_in = "wmx_%s_%.1f_shuffled.pkl"%(STDP_mode, place_cell_ratio); multipliers = np.arange(0.7, 1.4, 0.1)
    #f_in = "wmx_%s_%.1f_binary.pkl"%(STDP_mode, place_cell_ratio); multipliers = np.arange(1.3, 2.0, 0.1) if STDP_mode == "sym" else np.arange(1.8, 2.5, 0.1)
    #f_in = "wmx_%s_%.1f_block_shuffled.pkl"%(STDP_mode, place_cell_ratio); multipliers = np.arange(0.7, 1.4, 0.1)
    #f_in = "wmx_%s_%.1f_shuffled_subpop_input.pkl"%(STDP_mode, place_cell_ratio); multipliers = np.arange(0.7, 1.4, 0.1)
    #f_in = "wmx_%s_%.1f_2envs.pkl"%(STDP_mode, place_cell_ratio); multipliers = np.arange(0.6, 1.3, 0.1)
    
    
    f_out = "%s.txt"%f_in[4:-4]
       
    pklf_name = os.path.join(base_path, "files", f_in)
    wmx_PC_E = load_wmx(pklf_name) * 1e9  # *1e9 nS conversion
    
    print f_in
    results = np.zeros((len(multipliers), 16))
    for i, multiplier in enumerate(multipliers):
        print "multiplier: %.2f"%multiplier
        
        if detailed:
            SM_PC, SM_BC, RM_PC, RM_BC, selection, StateM_PC, StateM_BC = run_simulation(wmx_PC_E*multiplier, STDP_mode, detailed=True,
                                                                                         LFP=analyse_LFP, que=que, save_spikes=save_spikes, verbose=verbose)
            results[i, :] = analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, multiplier=multiplier,
                                            detailed=True, selection=selection, StateM_PC=StateM_PC, StateM_BC=StateM_BC,
                                            TFR=TFR, analyse_LFP=analyse_LFP, verbose=verbose)
        else:
            SM_PC, SM_BC, RM_PC, RM_BC = run_simulation(wmx_PC_E*multiplier, STDP_mode, detailed=False, que=que, save_spikes=save_spikes, verbose=verbose)
            results[i, :] = analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, multiplier=multiplier, detailed=False, TFR=TFR, verbose=verbose)
        
        plt.close("all")
    
    
    f_name = os.path.join(base_path, "files", "results", f_out)
    np.savetxt(f_name, results, fmt="%.3f", delimiter="\t", header=header)
    
    plot_summary_replay(multipliers, avg_replay_intervals=results[:, 1], rates_PC=results[:, 2], rates_BC=results[:, 3])
    plot_summary_ripple(multipliers, ripple_freqs_PC=results[:, 4], ripple_freqs_BC=results[:, 6],
                        ripple_powers_PC=results[:, 5], ripple_powers_BC=results[:, 7])
    plot_summary_gamma(multipliers, gamma_freqs_PC=results[:, 8], gamma_freqs_BC=results[:, 10],
                       gamma_powers_PC=results[:, 9], gamma_powers_BC=results[:, 11])
    plot_summary_AC(multipliers, max_acs_PC=results[:, 12], max_acs_BC=results[:, 14],
                    max_acs_ripple_PC=results[:, 14], max_acs_ripple_BC=results[:, 15])
        
                 
              
