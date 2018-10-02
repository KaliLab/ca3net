# -*- coding: utf8 -*-
"""
Optimizes connection parameters (synaptic weights)
authors: Bence Bagi, András Ecker last update: 10.2018
"""

import os, sys, pickle, logging
import numpy as np
import bluepyopt as bpop
import multiprocessing as mp
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add "scripts" directory to the path (to import modules)
sys.path.insert(0, os.path.sep.join([base_path, "scripts"]))
from detect_oscillations import preprocess_monitors, replay_circular, analyse_rate, ripple, gamma
from plots import plot_evolution, plot_raster, plot_PSD, plot_zoomed
import sim_evaluator

# print info into console
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_wmx(pklf_name):
    """
    Dummy function to load in the excitatory weight matrix and make python clear the memory
    :param pklf_name: file name of the saved weight matrix
    :return: wmx_PC_E: excitatory weight matrix
    """
    
    with open(pklf_name, "rb") as f:
        wmx_PC_E = pickle.load(f)
        
    return wmx_PC_E


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[1]
    except:
        STDP_mode = "asym"
    assert STDP_mode in ["sym", "asym"]
    place_cell_ratio = 0.5
    f_in = "wmx_%s_%.1f.pkl"%(STDP_mode, place_cell_ratio)
        
    # parameters to be fitted as a list of: (name, lower bound, upper bound)
    optconf = [("w_PC_I_", 0.1, 1.0),
               ("w_BC_E_", 3.0, 5.5),
               ("w_BC_I_", 6.0, 8.0),
               ("wmx_mult_", 0.2, 3.0),
               ("w_PC_MF_", 20., 45.),
               ("rate_MF_", 10., 25.)]
               # the order matters! if you want to add more parameters - update `run_sim.py` too
    pnames = [name for name, _, _ in optconf]
    
    offspring_size = 25
    max_ngen = 20
    
    
    pklf_name = os.path.join(base_path, "files", f_in)
    wmx_PC_E = load_wmx(pklf_name)

    # Create multiprocessing pool for parallel evaluation of fitness function
    pool = mp.Pool(processes=mp.cpu_count())

    # Create BluePyOpt optimization and run
    evaluator = sim_evaluator.Brian2Evaluator(wmx_PC_E, optconf)
    opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=offspring_size, map_function=pool.map,
                                              eta=20, mutpb=0.3, cxpb=0.7)
    
    print "Started running %i simulations on %i cores..."%(offspring_size*max_ngen ,mp.cpu_count())                                   
    pop, hof, log, hist = opt.run(max_ngen=max_ngen,
                                  cp_filename=os.path.join(base_path, "scripts", "optimization", "checkpoints", "checkpoint_%.1f_%s_1.pkl"%(place_cell_ratio, STDP_mode)))
    del pool; del opt
    
    # ====================================== end of optimization ======================================

    # summary figure (about optimization)
    plot_evolution(log.select("gen"), np.array(log.select("min")), np.array(log.select("avg")),
                   np.array(log.select("std")), "fittnes_evolution")

    
    best = hof[0]  # get best individual
    for pname, value in zip(pnames, best):
        print "%s = %.3f"% (pname, value)

    # rerun with best parameters and save figures
    SM_PC, SM_BC, RM_PC, RM_BC = evaluator.generate_model(best, verbose=True)
    del evaluator

    if SM_PC.num_spikes > 0 and SM_BC.num_spikes > 0:  # check if there is any activity
    
        spike_times_PC, spiking_neurons_PC, rate_PC, ISI_hist_PC, bin_edges_PC = preprocess_monitors(SM_PC, RM_PC)
        spike_times_BC, spiking_neurons_BC, rate_BC = preprocess_monitors(SM_BC, RM_BC, calc_ISI=False)
        
        replay, avg_replay_interval = replay_circular(ISI_hist_PC[3:16])  # bins from 150 to 850 (range of interest)
        
        mean_rate_PC, rate_ac_PC, max_ac_PC, t_max_ac_PC, f_PC, Pxx_PC = analyse_rate(rate_PC, fs=1000.0, slice_idx=[], TFR=False)
        mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC = analyse_rate(rate_BC, fs=1000.0, slice_idx=[], TFR=False)
                
        max_ac_ripple_PC, t_max_ac_ripple_PC, avg_ripple_freq_PC, ripple_power_PC = ripple(rate_ac_PC, f_PC, Pxx_PC, slice_idx=[])
        max_ac_ripple_BC, t_max_ac_ripple_BC, avg_ripple_freq_BC, ripple_power_BC = ripple(rate_ac_BC, f_BC, Pxx_BC, slice_idx=[])
        avg_gamma_freq_PC, gamma_power_PC = gamma(f_PC, Pxx_PC, slice_idx=[])       
        avg_gamma_freq_BC, gamma_power_BC = gamma(f_BC, Pxx_BC, slice_idx=[])
    
        if not np.isnan(replay):
            print "Replay detected!"
        else:
            print "No replay... :("
        print "Mean excitatory rate: %.3f"%mean_rate_PC
        print "Mean inhibitory rate: %.3f"%mean_rate_BC
        print "Average exc. ripple freq: %.3f"%avg_ripple_freq_PC
        print "Exc. ripple power: %.3f"%ripple_power_PC
        print "Average inh. ripple freq: %.3f"%avg_ripple_freq_BC
        print "Inh. ripple power: %.3f"%ripple_power_BC
        
        plot_raster(spike_times_PC, spiking_neurons_PC, rate_PC, [ISI_hist_PC, bin_edges_PC], False, "blue", multiplier_=1)
        plot_PSD(rate_PC, rate_ac_PC, f_PC, Pxx_PC, "PC_population", "blue", multiplier_=1)
        plot_PSD(rate_BC, rate_ac_BC, f_BC, Pxx_BC, "BC_population", "green", multiplier_=1)
        plot_zoomed(spike_times_PC, spiking_neurons_PC, rate_PC, "PC_population", "blue", multiplier_=1)
        plot_zoomed(spike_times_BC, spiking_neurons_BC, rate_BC, "BC_population", "green", multiplier_=1, PC_pop=False)
        
    else:
    
        print "No activity !"


