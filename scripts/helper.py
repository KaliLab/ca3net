# -*- coding: utf8 -*-
"""
Helper functions used here and there
author: András Ecker, last update: 06.2019
"""

import os
from shutil import rmtree
import pickle
from copy import deepcopy
import numpy as np
import pywt
from brian2.units import *
from poisson_proc import hom_poisson, get_tuning_curve_linear


base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
nPCs = 8000
nBCs = 150
Erev_E = 0.0*mV
Erev_I = -70.0*mV
len_sim = 10000  # ms
volume_cond = 1e9*nS / 3.54e5  # 1/3.54 S/m (`_estimate_LFP()` assumes 10 um distance...)


# ========== process Brian2 monitors ==========

def preprocess_monitors(SM, RM, calc_ISI=True):
    """
    preprocess Brian's SpikeMonitor and PopulationRateMonitor data for further analysis and plotting
    :param SM: Brian2 SpikeMonitor
    :param RM: Brian2 PopulationRateMonitor
    :param calc_ISI: bool for calculating ISIs
    :return spike_times, spiking_neurons: 2 lists: spike times and corresponding neuronIDs
            rate: firing rate of the population
            ISI_hist and ISI_bin_edges: bin heights and edges of the histogram of the ISI of the population
    """

    spike_times = np.array(SM.t_) * 1000.  # *1000 ms conversion
    spiking_neurons = np.array(SM.i_)
    tmp_spike_times = SM.spike_trains().items()
    rate = np.array(RM.rate_).reshape(-1, 10).mean(axis=1)

    if calc_ISI:
        ISIs = np.hstack([np.diff(spikes_i*1000) for i, spikes_i in tmp_spike_times])  # *1000 ms conversion
        ISI_hist, bin_edges = np.histogram(ISIs, bins=20, range=(0,1000))

        return spike_times, spiking_neurons, rate, ISI_hist, bin_edges
    else:
        return spike_times, spiking_neurons, rate


def _estimate_LFP(StateM, subset):
    """
    Estimates LFP by summing synaptic currents to PCs (assuming that all neurons are at equal distance (10 um) from the electrode)
    :param StateM: Brian2 StateMonitor object (of the PC population)
    :param subset: IDs of the recorded neurons
    :return: t, LFP: estimated LFP (in uV) and corresponding time points (in ms)
    """

    t = StateM.t_ * 1000.  # *1000 ms conversion
    LFP = np.zeros_like(t)*pA

    for i in subset:
        v = StateM[i].vm
        g_exc = StateM[i].g_ampa*nS + StateM[i].g_ampaMF*nS
        i_exc = g_exc * (v - (Erev_E * np.ones_like(v/mV)))  # pA
        g_inh = StateM[i].g_gaba*nS
        i_inh = g_inh * (v - (Erev_I * np.ones_like(v/mV)))  # pA
        LFP += -(i_exc + i_inh)  # (this is still in pA)

    LFP *= 1 / (4 * np.pi * volume_cond)

    return t, LFP/mV


def _avg_rate(rate, bin_, zoomed=False):
    """
    Averages rate (used also for bar plots)
    :param rate: np.array representing firing rates (hard coded for 10000 ms simulations)
    :param bin_: bin size
    :param zoomed: bool for zoomed in plots
    """

    t = np.linspace(0, len_sim, len(rate))
    t0 = 0 if not zoomed else 9900
    t1 = np.arange(t0, len_sim, bin_)
    t2 = t1 + bin_
    avg_rate = np.zeros_like(t1, dtype=np.float)
    for i, (t1_, t2_) in enumerate(zip(t1, t2)):
        avg_ = np.mean(rate[np.where((t1_ <= t) & (t < t2_))])
        if avg_ != 0.:
            avg_rate[i] = avg_

    return avg_rate


# ========== 2 environments ==========


# not used in the final version...
def merge_PF_starts():
    """Merges place field starting point generated for 2 different environments"""

    pklf_name = os.path.join(base_path, "files", "PFstarts_0.5_linear.pkl")
    with open(pklf_name, "rb") as f:
        place_fields = pickle.load(f, encoding="latin1")

    pklf_name = os.path.join(base_path, "files", "PFstarts_0.5_linear_no.pkl")
    with open(pklf_name, "rb") as f:
        place_fields_no = pickle.load(f, encoding="latin1")

    n = 0
    for i, PF_start_no in place_fields_no.items():
        if i in place_fields:
            PF_start = place_fields[i]
            place_fields[i] = [PF_start, PF_start_no]
            n += 1
        else:
            place_fields[i] = PF_start_no

    print("%i cells have place fields in both envs." % n)

    pklf_name = os.path.join(base_path, "files", "PFstarts_0.5_2envs_linear.pkl")
    with open(pklf_name, "wb") as f:
        pickle.dump(place_fields, f, protocol=pickle.HIGHEST_PROTOCOL)


def reorder_spiking_neurons(spiking_neurons, pklf_name_tuning_curves):
    """
    Reorders spiking neurons based on the intermediate (non-ordered) place fields
    :param spiking_neurons: list of spiking neurons (ordered)
    :param pklf_name_tuning_curves: file name of the tuning curves (in the non-ordered env.) - used only for idx
    :return: reordered_spiking_neurons: same spiking neurons list with neuron idx swapped to the non-ordered env. ones
    """

    with open(pklf_name_tuning_curves, "rb") as f:
        place_fields = pickle.load(f, encoding="latin1")

    # create a mapping between gids in the ordered env. and the non-ordered one
    PF_idx = np.asarray(list(place_fields.keys()))
    PF_starts = np.asarray(list(place_fields.values()))
    sort_idx = np.argsort(PF_starts, kind="mergesort")
    sorted_PF_idx = PF_idx[sort_idx]
    # key: ordered, val: non-ordered
    id_map_PF = {neuron_id: PF_idx[i] for i, neuron_id in enumerate(sorted_PF_idx)}
    assert np.sum(list(id_map_PF.keys())) == np.sum(list(id_map_PF.values()))

    # create a random mapping for gids which don't have place fields in the non-ordered env.
    # in order to get rid of "ghost" replays - replays in the other env. in the raster plot
    # TODO investigate why this is needed!
    non_PFs = np.array([neuron_id for neuron_id in range(nPCs) if neuron_id not in id_map_PF])
    tmp = deepcopy(non_PFs)
    np.random.shuffle(tmp)
    id_map_nonPF = {neuron_id: tmp[i] for i, neuron_id in enumerate(non_PFs)}
    assert np.sum(list(id_map_nonPF.keys())) == np.sum(list(id_map_nonPF.values()))

    reordered_spiking_neurons = np.zeros_like(spiking_neurons)
    for neuron_id in np.unique(spiking_neurons):
        if neuron_id in id_map_PF:  # place cells
            reordered_spiking_neurons[spiking_neurons == neuron_id] = id_map_PF[neuron_id]
        else:
            reordered_spiking_neurons[spiking_neurons == neuron_id] = id_map_nonPF[neuron_id]
    return reordered_spiking_neurons


# ========== saving & loading ==========


def create_dir(dir_name):
    """
    Deletes dir (if exists) and creates a new one with
    :param dir_name: string: full path of the directory to be created
    """
    if os.path.isdir(dir_name):
        rmtree(dir_name)
        os.mkdir(dir_name)
    else:
        os.mkdir(dir_name)


def save_place_fields(place_fields, pklf_name):
    """
    Save place field starts and corresponding neuron IDs for further analysis (see `bayesian_decoding.py`)
    :param place_fields: dict: neuron id:place field start
    :param pklf_name: name of saved file
    """

    with open(pklf_name, "wb") as f:
        pickle.dump(place_fields, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_vars(SM, RM, StateM, subset, seed, f_name="sim_vars_PC"):
    """
    Saves PC pop spikes, firing rate, membrane voltage, adaptation current and PSCs
    from a couple of recorded neurons after the simulation
    :param SM, RM: Brian2 SpikeMonitor, PopulationRateMonitor and StateMonitor
    :param subset: IDs of the recorded neurons
    :param seed: random seed used to run the simulation - here used only for naming
    :param f_name: name of saved file
    """

    spike_times, spiking_neurons, rate = preprocess_monitors(SM, RM, calc_ISI=False)
    # get PSCs from recorded voltage and conductances (and adaptation current)
    vs, PSCs, ws = {}, {}, {}
    for i in subset:
        v = StateM[i].vm
        vs[i] = v/mV
        g_exc = StateM[i].g_ampa*nS
        i_exc = -g_exc * (v - (Erev_E * np.ones_like(v/mV)))  # pA
        # separate outer (mossy fiber) input, from AMPA cond from local cells
        g_MF = StateM[i].g_ampaMF*nS
        i_MF = -g_MF * (v - (Erev_E * np.ones_like(v / mV)))  # pA
        g_inh = StateM[i].g_gaba*nS
        i_inh = -g_inh * (v - (Erev_I * np.ones_like(v/mV)))  # pA
        PSCs[i] = {"i_exc": i_exc/pA, "i_MF": i_MF/pA, "i_inh": i_inh/pA}
        ws[i] = StateM[i].w/pA
    # (shouldn't really be saved to PSCs only but keeping it for consistency)
    PSCs["t"] = StateM.t_ * 1000.  # *1000 ms conversion

    results = {"spike_times": spike_times, "spiking_neurons": spiking_neurons, "rate": rate,
               "vs": vs, "PSCs": PSCs, "ws": ws}
    pklf_name = os.path.join(base_path, "files", "%s_%s.pkl" % (f_name, seed))
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_PSD(f_PC, Pxx_PC, f_BC, Pxx_BC, f_LFP, Pxx_LFP, seed, f_name="PSD"):
    """
    Saves PSDs for PC and BC pop as well as LFP
    :params: f*, Pxx*: freqs and PSD (see `analyse_rate()` and `analyse_estimated_LFP()`)
    :param seed: random seed used to run the simulation - here used only for naming
    :param f_name: name of saved file
    """

    results = {"f_PC":f_PC, "Pxx_PC":Pxx_PC, "f_BC":f_BC, "Pxx_BC":Pxx_BC, "f_LFP":f_LFP, "Pxx_LFP":Pxx_LFP}
    pklf_name = os.path.join(base_path, "files", "%s_%s.pkl"%(f_name, seed))
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_TFR(f_PC, coefs_PC, f_BC, coefs_BC, f_LFP, coefs_LFP, seed, f_name="TFR"):
    """
    Saves TFR for PC and BC pop as well as LFP
    :params: f*, coefs*: freqs and coefficients (see `calc_TFR()`)
    :param seed: random seed used to run the simulation - here used only for naming
    :param f_name: name of saved file
    """

    results = {"f_PC":f_PC, "coefs_PC":coefs_PC, "f_BC":f_BC, "coefs_BC":coefs_BC, "f_LFP":f_LFP, "coefs_LFP":coefs_LFP}
    pklf_name = os.path.join(base_path, "files", "%s_%s.pkl"%(f_name, seed))
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_LFP(t, LFP, seed, f_name="LFP"):
    """
    Saves estimated LFP
    :params: t, LFP: time and LFP (see `analyse_estimated_LFP()`)
    :param seed: random seed used to run the simulation - here used only for naming
    :param f_name: name of saved file
    """

    results = {"t":t, "LFP":LFP}
    pklf_name = os.path.join(base_path, "files", "%s_%s.pkl"%(f_name, seed))
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_replay_analysis(replay, replay_results, seed, f_name="replay"):
    """
    Saves estimated LFP
    :params: replay: 1/nan for detected/undetected replay
    :param replay_results: saved matrices, fitted paths and sign. analysis from replay analysis (see `replay_linear()`)
    :param seed: random seed used to run the simulation - here used only for naming
    :param f_name: name of saved file
    """

    results = {"replay":replay, "replay_results":replay_results}
    pklf_name = os.path.join(base_path, "files", "%s_%s.pkl"%(f_name, seed))
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_step_sizes(trajectories, step_sizes, avg_step_sizes, gamma_filtered_LFPs, f_name="step_sizes"):
    """
    Saves estimated trajectory, calculated step sizes and slow gamma filtered LFP
    :param trajectories: estimated (from posterior matrix) trajectories
    :param step_sizes: event step sizes calculated from estimated trajectories
    :param avg_step_size: average step sizes calculated from distance and time of trajectories
    :param gamma_filtered_LFPs: gamma freq filtered and sliced LFP
    """

    results = {"trajectories":trajectories, "step_sizes":step_sizes,
               "avg_step_sizes":avg_step_sizes, "gamma_filtered_LFPs":gamma_filtered_LFPs}
    pklf_name = os.path.join(base_path, "files", "%s.pkl"%f_name)
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_gavg_step_sizes(step_sizes, phases, avg_step_sizes, seeds, f_name="gavg_step_sizes"):
    """
    Saves estimated step sizes and phases from sims with multiple seeds
    :param step_sizes: event step sizes calculated from estimated trajectories
    :param phases: calculated phases for every step size
    :param avg_step_size: average step sizes calculated from distance and time of trajectories
    :param seeds: seeds of different sims
    """

    results = {"step_sizes":step_sizes, "phases":phases,
               "avg_step_sizes":avg_step_sizes, "seeds":seeds}
    pklf_name = os.path.join(base_path, "files", "%s.pkl"%f_name)
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_wmx(weightmx, pklf_name):
    """
    Saves excitatory weight matrix
    :param weightmx: synaptic weight matrix to save
    :param pklf_name: file name of the saved weight matrix
    """

    np.fill_diagonal(weightmx, 0.0)
    with open(pklf_name, "wb") as f:
        pickle.dump(weightmx, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_wmx(pklf_name):
    """
    Dummy function to load in the excitatory weight matrix and make python clear the memory
    :param pklf_name: file name of the saved weight matrix
    :return: wmx_PC_E: excitatory weight matrix
    """

    with open(pklf_name, "rb") as f:
        wmx_PC_E = pickle.load(f, encoding="latin1")

    return wmx_PC_E


def load_spikes(pklf_name):
    """
    Loads in saved spikes from simulations
    param pklf_name: name of saved file
    return: spike_times, spiking_neurons, rate
    """

    with open(pklf_name, "rb") as f:
        tmp = pickle.load(f, encoding="latin1")
    return tmp["spike_times"], tmp["spiking_neurons"], tmp["rate"]


def load_LFP(pklf_name):
    """
    Loads in saved LFP from simulations
    param pklf_name: name of saved file
    return: t, LFP
    """

    with open(pklf_name, "rb") as f:
        tmp = pickle.load(f, encoding="latin1")
    return tmp["t"], tmp["LFP"]


def load_spike_trains(npzf_name):
    """
    Loads in spike trains and converts it to 2 np.arrays for Brian2's SpikeGeneratorGroup
    :param npzf_name: file name of saved spike trains
    :return spiking_neurons, spike_times: same spike trains converted into SpikeGeneratorGroup format
    """

    npz_f = np.load(npzf_name, allow_pickle=True)
    spike_trains = npz_f["spike_trains"]

    spiking_neurons = 0 * np.ones_like(spike_trains[0])
    spike_times = np.asarray(spike_trains[0])
    for neuron_id in range(1, nPCs):
        tmp = neuron_id * np.ones_like(spike_trains[neuron_id])
        spiking_neurons = np.concatenate((spiking_neurons, tmp), axis=0)
        spike_times = np.concatenate((spike_times, np.asarray(spike_trains[neuron_id])), axis=0)

    return spiking_neurons, spike_times


def _load_PF_starts(pklf_name):
    """
    Loads in saved place field starting points [rad]
    :param pklf_name: filename of saved place fields
    :return: place_fields: dict neuronID: place field start (saved in `generate_spike_trains.py`)
    """

    with open(pklf_name, "rb") as f:
        place_fields = pickle.load(f, encoding="latin1")

    return place_fields


def load_tuning_curves(pklf_name, spatial_points):
    """
    Loads in tau_i(x) tuning curves (used for generating 'teaching' spike train, see `poisson_proc.py`)
    (Can handle multiple place fields in different environments)
    :param pklf_name: see `_load_PF_starts`
    :param spatial_points: spatial coordinates to evaluate the tuning curves
    :return: tuning_curves: dict of tuning curves {neuronID: tuning curve}
    """

    place_fields = _load_PF_starts(pklf_name)
    #tuning_curves = {neuron_id: get_tuning_curve_linear(spatial_points, phi_start) for neuron_id, phi_start in place_fields.items()}
    tuning_curves = {}
    for neuron_id, phi_start in place_fields.items():
        if type(phi_start) != list:
            tuning_curves[neuron_id] = get_tuning_curve_linear(spatial_points, phi_start)
        else:  # multiple envs.
            tuning_curves_ = np.zeros((len(phi_start), len(spatial_points)))
            for i, phi_start_ in enumerate(phi_start):
                tuning_curves_[i, :] = get_tuning_curve_linear(spatial_points, phi_start_)
            tuning_curve = np.sum(tuning_curves_, axis=0)
            tuning_curve[np.where(tuning_curve > 1.)] = 1.
            tuning_curves[neuron_id] = tuning_curve

    return tuning_curves


# ========== misc. ==========


def refractoriness(spike_trains, ref_per=5e-3):
    """
    Delete spikes (from generated train) which are too close to each other
    :param spike_trains: list of lists representing individual spike trains
    :param ref_per: refractory period (in sec)
    :return spike_trains: same structure, but with some spikes deleted
    """

    spike_trains_updated = []; count = 0
    for single_spike_train in spike_trains:
        tmp = np.diff(single_spike_train)  # calculate ISIs
        idx = np.where(tmp < ref_per)[0] + 1
        if idx.size:
            count += idx.size
            single_spike_train_updated = np.delete(single_spike_train, idx).tolist()  # delete spikes which are too close
        else:
            single_spike_train_updated = single_spike_train
        spike_trains_updated.append(single_spike_train_updated)

    print("%i spikes deleted becuse of too short refractory period" % count)

    return spike_trains_updated


def _get_consecutive_sublists(list_):
    """
    Groups list into sublists of consecutive numbers
    :param list_: input list to group
    :return cons_lists: list of lists with consecutive numbers
    """

    # get upper bounds of consecutive sublists
    ubs = [x for x,y in zip(list_, list_[1:]) if y-x != 1]

    cons_lists = []; lb = 0
    for ub in ubs:
        tmp = [x for x in list_[lb:] if x <= ub]
        cons_lists.append(tmp)
        lb += len(tmp)
    cons_lists.append([x for x in list_[lb:]])

    return cons_lists


def argmin_time_arrays(time_short, time_long):
    """
    Finds closest elements in differently sampled time arrays (used for step size analysis...)
    TODO: add some error management here....
    :param time_short: time array with less elements
    :param time_long: time array with more elements (in the same range)
    :return: idx of long array, to get closest elements to short array
    """

    return [np.argmin(np.abs(time_long-t)) for t in time_short]


def generate_cue_spikes():
    """Generates short (200ms) Poisson spike train at 20Hz (with brian2's `PoissonGroup()` one can't specify the duration)"""

    spike_times = np.asarray(hom_poisson(20.0, 10, t_max=0.2, seed=12345))
    spiking_neurons = np.zeros_like(spike_times)
    for neuron_id in range(1, 100):
        spike_times_tmp = np.asarray(hom_poisson(20.0, 10, t_max=0.2, seed=12345+neuron_id))
        spike_times = np.concatenate((spike_times, spike_times_tmp), axis=0)
        spiking_neurons_tmp = neuron_id * np.ones_like(spike_times_tmp)
        spiking_neurons = np.concatenate((spiking_neurons, spiking_neurons_tmp), axis=0)

    return spike_times, spiking_neurons


def calc_spiketrain_ISIs():
    """Calculates inter spike intervals within the generated spike trains (separately for place cells, non-place cells)"""

    # just to get place cell idx
    pklf_name = os.path.join(base_path, "files", "PFstarts_0.5_linear.pkl")
    with open(pklf_name, "rb") as f:
        PFs = pickle.load(f, encoding="latin1")

    npzf_name = os.path.join(base_path, "files", "spike_trains_0.5_linear.npz")
    npz_f = np.load(npzf_name)
    spike_trains = npz_f["spike_trains"]

    place_cell_ISIs = []
    nplace_cell_ISIs = []
    for i in range(nPCs):
        if i in PFs:
            place_cell_ISIs.extend(np.diff(spike_trains[i]).tolist())
        else:
            nplace_cell_ISIs.extend(np.diff(spike_trains[i]).tolist())

    results = {"PCs":np.asarray(place_cell_ISIs), "nPCs":np.asarray(nplace_cell_ISIs)}
    pklf_name = os.path.join(base_path, "files", "spiketrain_ISIs.pkl")
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def calc_single_cell_rates(seed):
    """Calculates single cell firing rates for cells (separately for place cells, non-place cells and BCs)"""

    # just to get place cell idx
    pklf_name = os.path.join(base_path, "files", "PFstarts_0.5_linear.pkl")
    with open(pklf_name, "rb") as f:
        PFs = pickle.load(f, encoding="latin1")

    pklf_name = os.path.join(base_path, "files", "sim_vars_PC_%s.pkl" % seed)
    spike_times, spiking_neurons, rate = load_spikes(pklf_name)

    place_cell_rates = []
    nplace_cell_rates = []
    for i in range(nPCs):
        spikes = spike_times[spiking_neurons == i]
        if i in PFs:
            place_cell_rates.append(len(spikes)/(len_sim/1000.))
        else:
            nplace_cell_rates.append(len(spikes)/(len_sim/1000.))

    pklf_name = os.path.join(base_path, "files", "sim_vars_BC_%s.pkl" % seed)
    spike_times, spiking_neurons, _ = load_spikes(pklf_name)

    BC_rates = []
    for i in range(nBCs):
        spikes = spike_times[spiking_neurons == i]
        BC_rates.append(len(spikes)/(len_sim/1000.))

    results = {"PCs": np.asarray(place_cell_rates), "nPCs": np.asarray(nplace_cell_rates), "BCs": np.asarray(BC_rates)}
    pklf_name = os.path.join(base_path, "files", "single_cell_rates_%s.pkl" % seed)
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def calc_ISIs(seed):
    """Calculates inter spike intervals for cells (separately for place cells, non-place cells and BCs)"""

    # just to get place cell idx
    pklf_name = os.path.join(base_path, "files", "PFstarts_0.5_linear.pkl")
    with open(pklf_name, "rb") as f:
        PFs = pickle.load(f, encoding="latin1")

    pklf_name = os.path.join(base_path, "files", "sim_vars_PC_%s.pkl"%seed)
    spike_times, spiking_neurons, _ = load_spikes(pklf_name)

    place_cell_ISIs = []
    nplace_cell_ISIs = []
    for i in range(nPCs):
        idx = np.where(spiking_neurons == i)[0]
        spikes = spike_times[idx]
        if i in PFs:
            place_cell_ISIs.extend(np.diff(spikes).tolist())
        else:
            nplace_cell_ISIs.extend(np.diff(spikes).tolist())

    pklf_name = os.path.join(base_path, "files", "sim_vars_BC_%s.pkl"%seed)
    spike_times, spiking_neurons, _ = load_spikes(pklf_name)

    BC_ISIs = []
    for i in range(nBCs):
        idx = np.where(spiking_neurons == i)[0]
        spikes = spike_times[idx]
        BC_ISIs.extend(np.diff(spikes).tolist())

    results = {"PCs":np.asarray(place_cell_ISIs), "nPCs":np.asarray(nplace_cell_ISIs), "BCs":np.asarray(BC_ISIs)}
    pklf_name = os.path.join(base_path, "files", "ISIs_%s.pkl"%seed)
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


def calc_LFP_TFR(seed):
    """Calculates TFR of the full LFP (not sliced, not downsampled)"""

    pklf_name = os.path.join(base_path, "files", "LFP_%s.pkl"%seed)
    t, LFP = load_LFP(pklf_name)
    fs = 10000.

    scales = np.concatenate((np.linspace(25, 80, 250), np.linspace(80, 300, 250)[1:]))  # 27-325 Hz  pywt.scale2frequency("morl", scale) / (1/fs)
    coefs, freqs = pywt.cwt(LFP, scales, "morl", 1/fs)

    results = {"coefs": coefs, "freqs": freqs}
    pklf_name = os.path.join(base_path, "files", "LFP_TFR_%s.pkl" % seed)
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)


#if __name__ == "__main__":
#    calc_spiketrain_ISIs()
#    seed = 12345
#    calc_single_cell_rates(seed)
#    calc_ISIs(seed)
#    calc_LFP_TFR(seed)
