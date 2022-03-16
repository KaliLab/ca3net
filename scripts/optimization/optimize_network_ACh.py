# -*- coding: utf8 -*-
"""
Mostly a copy-paste of `optimize_network.py` but this one is optimized for gamma oscillation in presence of ACh
(Optimizes connection parameters (synaptic weights))
author: András Ecker last update: 03.2022
"""

import os, sys, logging
import numpy as np
import pandas as pd
import bluepyopt as bpop
import multiprocessing as mp
import sim_evaluator_gamma as sim_evaluator
from optimize_network import load_checkpoints, hof2csv
base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-3])
# add "scripts" directory to the path (to import modules)
sys.path.insert(0, os.path.sep.join([base_path, "scripts"]))
from helper import load_wmx
from plots import plot_evolution

# print info into console
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


if __name__ == "__main__":

    f_in = "wmx_sym_0.5_linear.pkl"
    cp_f_name = os.path.join(base_path, "scripts", "optimization", "checkpoints", "gamma_checkpoint_%s" % f_in[4:])
    hof_f_name = os.path.join(base_path, "scripts", "optimization", "checkpoints", "gamma_hof_%s.csv" % f_in[4:-4])

    # ACh (actually carbachol) scale factors
    PC_E_ACh = 0.255  # PC EXC drops to 25.5% in Norbert's data
    PC_I_ACh = 0.28  # PC INH drops to 28% in Norbert's data
    BC_E_ACh = 0.4  # PVBC EXC drops to 40% in Norbert's data
    BC_I_ACh = np.mean([PC_E_ACh, PC_I_ACh, BC_E_ACh])  # no experimental data for this connection ...
    range_ACh = [-0.1, 0.1]  # % "wiggle room" for the above specified values...
    # parameters to be fitted as a list of: (name, lower bound, upper bound) - ripple optimize * ACh scale factors
    # the order matters! if you want to add more parameters - update `run_sim.py` too
    optconf = [("w_PC_I_", 0.65 * (PC_I_ACh + range_ACh[0]), 0.65 * (PC_I_ACh + range_ACh[1])),
               ("w_BC_E_", 0.85 * (BC_E_ACh + range_ACh[0]), 0.85 * (BC_E_ACh + range_ACh[1])),
               ("w_BC_I_", 5 * (BC_I_ACh + range_ACh[0]), 5 * (BC_I_ACh + range_ACh[1])),
               ("wmx_mult_", 0.62 * (PC_E_ACh + range_ACh[0]), 0.62 * (PC_E_ACh + range_ACh[1])),
               ("w_PC_MF_", 20.0, 30.0),
               ("rate_MF_", 10.0, 20.0)]
    pnames = [name for name, _, _ in optconf]

    offspring_size = 50
    max_ngen = 10

    pklf_name = os.path.join(base_path, "files", f_in)
    wmx_PC_E = load_wmx(pklf_name) * 1e9  # *1e9 nS conversion

    # Create multiprocessing pool for parallel evaluation of fitness function
    n_proc = np.max([offspring_size, mp.cpu_count()-1])
    pool = mp.Pool(processes=n_proc)
    # Create BluePyOpt optimization and run
    evaluator = sim_evaluator.Brian2Evaluator(True, wmx_PC_E, optconf)
    opt = bpop.optimisations.DEAPOptimisation(evaluator, offspring_size=offspring_size, map_function=pool.map,
                                              eta=20, mutpb=0.3, cxpb=0.7)

    print("Started running %i simulations on %i cores..." % (offspring_size*max_ngen, n_proc))
    pop, hof, log, hist = opt.run(max_ngen=max_ngen, cp_filename=cp_f_name)
    del pool, opt
    # ====================================== end of optimization ======================================

    # summary figure (about optimization)
    plot_evolution(log.select("gen"), np.array(log.select("min")), np.array(log.select("avg")),
                   np.array(log.select("std")), "fittnes_evolution")
    # pop, hof, log, hist = load_checkpoints(cp_f_name)

    # save hall of fame to csv, get best individual, and rerun with best parameters to save figures
    hof2csv(pnames, hof, hof_f_name)
    best = hof[0]
    for pname, value in zip(pnames, best):
        print("%s = %.3f" % (pname, value))
    _ = evaluator.evaluate_with_lists(best, verbose=True, plots=True)
