# -*- coding: utf8 -*-
"""
Load in and analyse results from BluePyOpt checkpoint
author: András Ecker last update: 10.2018
"""

import os, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
import sim_evaluator
from optimize_network import load_wmx, analyse_results
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add "scripts" directory to the path (to import modules)
sys.path.insert(0, os.path.sep.join([base_path, "scripts"]))
from plots import plot_evolution


# optimized parameters (the bounds might be difference, but it's necessary to have this here...)
optconf = [("w_PC_I_", 0.1, 0.3),
           ("w_BC_E_", 3.5, 4.5),
           ("w_BC_I_", 6.5, 8.0),
           ("wmx_mult_", 0.8, 3.0),
           ("w_PC_MF_", 20., 35.),
           ("rate_MF_", 10., 25.)]
pnames = [name for name, _, _ in optconf]


def load_checkpoints(pklf_name):
    """
    Loads in saved checkpoints from pickle file
    :param pklf_name: name of the saved pickle file
    :return: saved (BluePyOpt) dicts"""

    cp = pickle.load(open(pklf_name))

    pop = cp["generation"]
    hof = cp["halloffame"]
    log = cp["logbook"]
    hist = cp["history"]
    
    # summary figure (about optimization)
    plot_evolution(log.select("gen"), np.array(log.select("min")), np.array(log.select("avg")),
                   np.array(log.select("std")), "fittnes_evolution")
               
    return pop, hof, log, hist


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[1]
    except:
        STDP_mode = "asym"
    assert STDP_mode in ["sym", "asym"]
    place_cell_ratio = 0.5
    f_in = "wmx_%s_%.1f.pkl"%(STDP_mode, place_cell_ratio)
    cp_in = "checkpoint_%.1f_%s_1.pkl"%(place_cell_ratio, STDP_mode)
    
    pklf_name = os.path.join(base_path, "scripts", "optimization", "checkpoints", cp_in)
    _, hof, _, _ =  load_checkpoints(pklf_name)
    
    best = hof[0]  # get best individual
    for pname, value in zip(pnames, best):
        print "%s = %.3f"% (pname, value)
    
    pklf_name = os.path.join(base_path, "files", f_in)
    wmx_PC_E = load_wmx(pklf_name) * 1e9  # *1e9 nS conversion
    
    evaluator = sim_evaluator.Brian2Evaluator(wmx_PC_E, optconf)
    SM_PC, SM_BC, RM_PC, RM_BC = evaluator.generate_model(best, verbose=True)
    analyse_results(SM_PC, SM_BC, RM_PC, RM_BC)
    plt.show()
    

