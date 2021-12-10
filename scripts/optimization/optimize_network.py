# -*- coding: utf8 -*-
"""
Optimizes connection parameters (synaptic weights)
authors: Bence Bagi, András Ecker last update: 10.2018
"""

import os, sys, logging
import numpy as np
import pandas as pd
import bluepyopt as bpop
import multiprocessing as mp
import sim_evaluator
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add "scripts" directory to the path (to import modules)
sys.path.insert(0, os.path.sep.join([base_path, "scripts"]))
from helper import load_wmx, preprocess_monitors
from detect_oscillations import analyse_rate, ripple
from detect_replay import slice_high_activity
from plots import plot_evolution, plot_raster, plot_PSD, plot_zoomed

# print info into console
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_checkpoints(pklf_name):
    """
    Loads in saved checkpoints from pickle file (used e.g. to repeat analysis...)
    :param pklf_name: name of the saved pickle file
    :return: obejects saved by BluePyOpt"""
    import pickle
    with open(pklf_name, "rb") as f:
        cp = pickle.load(f)
    return cp["generation"], cp["halloffame"], cp["logbook"], cp["history"]


def hof2csv(pnames, hof, f_name):
    """
    Creates pandas DaataFrame from hall of fame and saves it to csv
    :param pnames: names of optimized parameters
    :param hof: BluePyOpt HallOfFame object
    :param f_name: name of the saved file
    """
    data = np.zeros((len(hof), len(pnames)))
    for i in range(len(hof)):
        data[i, :] = hof[i]
    df = pd.DataFrame(data=data, columns=pnames)
    df.to_csv(f_name)


def analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, linear):
    """
    Duplicated (simpler) version of `../spw_network.py/analyse_results()`
    :param SM_PC, SM_BC, RM_PC, RM_BC: Brian2 spike and rate monitors of PC and BC populations
    """

    if SM_PC.num_spikes > 0 and SM_BC.num_spikes > 0:  # check if there is any activity
        spike_times_PC, spiking_neurons_PC, rate_PC, ISI_hist_PC, bin_edges_PC = preprocess_monitors(SM_PC, RM_PC)
        spike_times_BC, spiking_neurons_BC, rate_BC = preprocess_monitors(SM_BC, RM_BC, calc_ISI=False)

        slice_idx = [] if not linear else slice_high_activity(rate_PC, th=2, min_len=260)
        mean_rate_PC, rate_ac_PC, max_ac_PC, t_max_ac_PC, f_PC, Pxx_PC = analyse_rate(rate_PC, 1000., slice_idx)
        mean_rate_BC, rate_ac_BC, max_ac_BC, t_max_ac_BC, f_BC, Pxx_BC = analyse_rate(rate_BC, 1000., slice_idx)

        avg_ripple_freq_PC, ripple_power_PC = ripple(f_PC, Pxx_PC, slice_idx)
        avg_ripple_freq_BC, ripple_power_BC = ripple(f_BC, Pxx_BC, slice_idx)

        print("Mean excitatory rate: %.3f" % mean_rate_PC)
        print("Mean inhibitory rate: %.3f" % mean_rate_BC)
        print("Average exc. ripple freq: %.3f" % avg_ripple_freq_PC)
        print("Exc. ripple power: %.3f" % ripple_power_PC)
        print("Average inh. ripple freq: %.3f" % avg_ripple_freq_BC)
        print("Inh. ripple power: %.3f" % ripple_power_BC)

        plot_raster(spike_times_PC, spiking_neurons_PC, rate_PC, [ISI_hist_PC, bin_edges_PC], slice_idx, "blue", multiplier_=1)
        plot_PSD(rate_PC, rate_ac_PC, f_PC, Pxx_PC, "PC_population", "blue", multiplier_=1)
        plot_PSD(rate_BC, rate_ac_BC, f_BC, Pxx_BC, "BC_population", "green", multiplier_=1)
        plot_zoomed(spike_times_PC, spiking_neurons_PC, rate_PC, "PC_population", "blue", multiplier_=1, PC_pop=False)
        plot_zoomed(spike_times_BC, spiking_neurons_BC, rate_BC, "BC_population", "green", multiplier_=1, PC_pop=False)

    else:
        print("No activity!")


if __name__ == "__main__":
    try:
        STDP_mode = sys.argv[1]
    except:
        STDP_mode = "sym"
    assert STDP_mode in ["sym", "asym"]
    linear = False
    place_cell_ratio = 0.5
    f_in = "wmx_%s_%.1f_linear.pkl" % (STDP_mode, place_cell_ratio) if linear else "wmx_%s_%.1f.pkl" % (STDP_mode, place_cell_ratio)
    cp_f_name = os.path.join(base_path, "scripts", "optimization", "checkpoints", "checkpoint_%s" % f_in[4:])
    hof_f_name = os.path.join(base_path, "scripts", "optimization", "checkpoints", "hof_%s" % f_in[4:])

    # parameters to be fitted as a list of: (name, lower bound, upper bound)
    # the order matters! if you want to add more parameters - update `run_sim.py` too
    optconf = [("w_PC_I_", 0.15, 0.4),
               ("w_BC_E_", 3.25, 4.25),
               ("w_BC_I_", 6.5, 8.0),
               ("wmx_mult_", 1.0, 3.0),
               ("w_PC_MF_", 24.0, 26.0),
               ("rate_MF_", 15.0, 17.0)]
    pnames = [name for name, _, _ in optconf]

    offspring_size = 35
    max_ngen = 10

    pklf_name = os.path.join(base_path, "files", f_in)
    wmx_PC_E = load_wmx(pklf_name) * 1e9  # *1e9 nS conversion

    # Create multiprocessing pool for parallel evaluation of fitness function
    pool = mp.Pool(processes=mp.cpu_count()-1)
    # Create BluePyOpt optimization and run
    evaluator = sim_evaluator.Brian2Evaluator(linear, wmx_PC_E, optconf)
    opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=offspring_size, map_function=pool.map,
                                              eta=20, mutpb=0.3, cxpb=0.7)

    print("Started running %i simulations on %i cores..." % (offspring_size*max_ngen, mp.cpu_count()-1))
    pop, hof, log, hist = opt.run(max_ngen=max_ngen, cp_filename=cp_f_name)
    del pool, opt
    # ====================================== end of optimization ======================================

    # summary figure (about optimization)
    plot_evolution(log.select("gen"), np.array(log.select("min")), np.array(log.select("avg")),
                   np.array(log.select("std")), "fittnes_evolution")

    # save hall of fame to csv, get best individual, and rerun with best parameters to save figures
    hof2csv(pnames, hof, hof_f_name)
    best = hof[0]
    for pname, value in zip(pnames, best):
        print("%s = %.3f" % (pname, value))
    SM_PC, SM_BC, RM_PC, RM_BC = evaluator.generate_model(best, verbose=True)
    analyse_results(SM_PC, SM_BC, RM_PC, RM_BC, linear)
