# -*- coding: utf8 -*-
"""
Helper functions to plot dynamics, weight matrix and couple of other things
authors: András Ecker, Bence Bagi last update: 02.2019
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from helper import _avg_rate


sns.set(style="ticks", context="notebook")
base_path = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
fig_dir = os.path.join(base_path, "figures")

nPCs = 8000
nBCs = 150
len_sim = 10000  # ms
spike_th_PC = -3.25524288  # (re-optimized by Szabolcs)
spike_th_BC = -34.78853881  # (re-optimized by Szabolcs)


def plot_raster(spike_times, spiking_neurons, rate, hist, slice_idx, color_, multiplier_):
    """
    Saves figure with raster plot and NEST like rate below
    :param spike_times, spiking_neurons: used for raster plot (see `detect_oscillation.py/preprocess_monitors()`)
    :param rate: firing rate of the population
    :param hist: used for plotting ISI histogram (see `detect_oscillations.py/preprocess_monitors()`)
    :param slice_idx: boundaries of detected high activity periods
    :param color_, multiplier_: outline and naming parameters
    """

    fig = plt.figure(figsize=(10, 8))
    if slice_idx is not None:
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
    else:
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 2])

    ax = fig.add_subplot(gs[0])
    ax.scatter(spike_times, spiking_neurons, c=color_, marker=".", s=12)
    ax.set_title("PC_population raster")
    ax.set_xlim([0, len_sim])
    ax.set_ylim([0, nPCs])
    ax.set_ylabel("Neuron ID")

    bin_ = 20
    avg_rate = _avg_rate(rate, bin_)

    ax2 = fig.add_subplot(gs[1])
    sns.despine(ax=ax2)
    ax2.bar(np.linspace(0, len_sim, len(avg_rate)), avg_rate, width=bin_, align="edge", color=color_, edgecolor="black", linewidth=0.5, alpha=0.9)
    if slice_idx is not None:
        for bounds in slice_idx:
            ax2.axvline(bounds[0], color="gray", ls="--")
            ax2.axvline(bounds[1], color="gray", ls="--")
    ax2.set_xlim([0, len_sim])
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Rate (Hz)")

    ax3 = fig.add_subplot(gs[2])
    sns.despine(ax=ax3)
    ax3.bar(hist[1][:-1], hist[0], width=50, align="edge", color=color_, edgecolor="black", linewidth=0.5, alpha=0.9)  # width=50 comes from bins=20
    if slice_idx is None:
        ax3.axvline(150, ls="--", c="gray", label="ROI for replay analysis")
        ax3.axvline(850, ls="--", c="gray")
        ax3.legend()
    ax3.set_title("ISI distribution")
    ax3.set_xlabel("$\Delta t$ (ms)")
    ax3.set_xlim([0, 1000])
    ax3.set_ylabel("Count")
    plt.yscale("log")

    fig.tight_layout()

    fig_name = os.path.join(fig_dir, "%.2f*.png"%multiplier_)
    fig.savefig(fig_name)


def plot_posterior_trajectory(X_posterior, fitted_path, R, fig_name, temporal_res=10):
    """
    Saves plot with the posterior distribution Pr(x|spikes) and fitted trajectory
    :param X_posterior: posterior matrix (see `bayesian_decoding.py`)
    :param fitted_path: fitted trajectory (see `bayesian_decoding.py`)
    :param R: godness of fit (see `bayesian_decoding.py`)
    :param fig_name: name of saved img
    :param temporal_res: temporal resolution used for binning spikes (used only for xlabel scaling)
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    i = ax.imshow(X_posterior, cmap="hot", aspect="auto", interpolation="hermite", origin="lower")
    fig.colorbar(i)
    ax.autoscale(False)
    ax.plot(fitted_path-3, color="white", lw=2)
    ax.plot(fitted_path+3, color="white", lw=2)
    ax.set_title("Posterior matrix and fitted path (R = %.2f)"%R)
    ax.set_xticklabels(ax.get_xticks().astype(int)*temporal_res)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Sampled position")

    fig.savefig(fig_name)
    plt.close(fig)


def plot_PSD(rate, rate_ac, f, Pxx, title_, color_, multiplier_):
    """
    Saves figure with rate, its autocorrelation and PSD
    :param rate: firing rate - precalculated by `detect_oscillation.py/preprocess_spikes()`
    :param rate_ac: autocorrelation function of the rate (see `detect_oscillation.py/analyse_rate()`)
    :param f, Pxx: estimated PSD and frequencies used (see `detect_oscillation.py/analyse_rate()`)
    :param title_, color_, multiplier: outline and naming parameters
    """

    bin_ = 20
    avg_rate = _avg_rate(rate, bin_)

    try:
        Pxx_plot = np.zeros_like(Pxx); rate_ac_plot = np.zeros((Pxx.shape[0], 199))
        for i in range(Pxx.shape[0]):
            Pxx_plot[i, :] = 10 * np.log10(Pxx[i, :] / max(Pxx[i, :]))
            if rate_ac[i].shape[0] >= 201:
                rate_ac_plot[i, :] = rate_ac[i][2:201]
            else:
                rate_ac_plot[i, 0:rate_ac[i].shape[0]-2] = rate_ac[i][2:201]
        Pxx_plot_mean = np.mean(Pxx_plot, axis=0)
        rate_ac_plot_mean = np.mean(rate_ac_plot, axis=0)
    except:
        Pxx_plot_mean = 10 * np.log10(Pxx / max(Pxx))
        rate_ac_plot_mean = rate_ac[2:201]
    f = np.asarray(f)
    f_ripple = f[np.where((150 < f) & (f < 220))]; Pxx_ripple_plot = Pxx_plot_mean[np.where((150 < f) & (f < 220))]
    f_gamma = f[np.where((30 < f) & (f < 100))]; Pxx_gamma_plot = Pxx_plot_mean[np.where((30 < f) & (f < 100))]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(3, 1, 1)

    ax.bar(np.linspace(0, len_sim, len(avg_rate)), avg_rate, width=bin_, align="edge", color=color_, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.set_xlim([0, len_sim])
    ax.set_title("%s rate"%title_)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Rate (Hz)")

    ax2 = fig.add_subplot(3, 1, 2)
    try:
        for rate_ac_plot_tmp in rate_ac_plot:
            ax2.plot(np.linspace(2, 200, len(rate_ac_plot_tmp)), rate_ac_plot_tmp, lw=0.5, color="gray", alpha=0.5)
    except:
        pass
    ax2.plot(np.linspace(2, 200, len(rate_ac_plot_mean)), rate_ac_plot_mean, color=color_)
    ax2.set_title("Autocorrelogram (500-5 Hz)")
    ax2.set_xlabel("Time shift (ms)")
    ax2.set_xlim([2, 200])
    ax2.set_ylabel("Autocorrelation")

    ax3 = fig.add_subplot(3, 1, 3)
    try:
        for Pxx_plot_tmp in Pxx_plot:
            ax3.plot(f, Pxx_plot_tmp, lw=0.5, color="gray", alpha=0.5)
    except:
        pass
    ax3.plot(f, Pxx_plot_mean, color=color_, marker="o")
    ax3.plot(f_ripple, Pxx_ripple_plot, "r-", marker="o", linewidth=1.5, label="ripple (150-220 Hz)")
    ax3.plot(f_gamma, Pxx_gamma_plot, "k-", marker="o", linewidth=1.5, label="gamma (30-100 Hz)")
    ax3.set_title("Power Spectrum Density")
    ax3.set_xlim([0, 500])
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("PSD (dB)")
    ax3.legend()

    sns.despine()
    fig.tight_layout()
    fig_name = os.path.join(fig_dir, "%.2f_%s.png"%(multiplier_, title_))
    fig.savefig(fig_name)


def plot_TFR(coefs, freqs, title_, fig_name):
    """
    Saves figure with time frequency representation
    :param coefs, freqs: coefficients from wavelet transform and frequencies used (see `pywt.cwt()` in `detect_oscillations/analyse_rate()`)
    :param title_, fig_name: outline and naming parameters
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    i = ax.imshow(np.abs(coefs), cmap=plt.get_cmap("jet"), aspect="auto", interpolation=None,
                  vmax=np.max(np.abs(coefs)), vmin=np.max(-np.abs(coefs)))
    fig.colorbar(i)
    ax.set_title("Wavlet transform of %s"%title_)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_yticks(np.arange(0, 300, 20)); ax.set_yticklabels(["%.1f"%i for i in freqs[::20].copy()])

    fig.savefig(fig_name)
    plt.close(fig)


def _select_subset(selection, ymin, ymax):
    """
    Helper function to select a subset of neurons for plotting more detailes (the subset is from the ones spiking in the last 100ms - see `plot_zoomed()`)
    param selection: np.array of recorded neurons
    param ymin, ymax: lower and upper bound for the selection
    return subset: list of selected subset
    """
    try:
        np.random.shuffle(selection)
        subset = []
        counter = 5
        for i in selection:
            if i >= ymin and i <= ymax:
                subset.append(i)
                counter -= 1
            if counter == 0:
                break
    except:  # if there isn't any cell firing
        subset = [500, 2000, 4000, 6000, 7500]
    return subset


def plot_zoomed(spike_times, spiking_neurons, rate, title_, color_, multiplier_, PC_pop=True, StateM=None, selection=None):
    """
    Saves figure with zoomed in raster, rate and optionally a trace (last 100ms)
    :param spike_times, spiking_neurons: used for raster plot - precalculated by `detect_oscillation.py/preprocess_spikes()`
    :param rate: firing rate - precalculated by detect_oscillation.py/preprocess_spikes
    :param title_, color_, linespec_, multiplier_: outline and naming parameters
    :param PC_pop: flag for calculating and returning ymin and ymax (and zooming in the plot)
    :param StateM: Brian2 StateMonitor object (could be more elegant...)
    :param selection: np.array of recorded neurons (used only if PC_pop is true)
    return subset: see `_select_subset()`
    """

    zoom_from = len_sim - 100  # ms

    # get last 100ms of raster
    idx = np.where(spike_times > zoom_from)
    spike_times = spike_times[idx]; spiking_neurons = spiking_neurons[idx]

    bin_ = 2
    avg_rate = _avg_rate(rate, bin_, zoomed=True)

    # set boundaries and marker size
    if PC_pop:
        ymin = spiking_neurons.min()-5 if spiking_neurons.min()-5 > 0 else 0
        ymax = spiking_neurons.max()+5 if spiking_neurons.max()+5 < nPCs else nPCs
        subset = _select_subset(selection, ymin, ymax)
        size_ = 12
    else:
        ymin = 0; ymax = nBCs
        size_ = 20

    if StateM:  # select trace to plot if StateMonitor is passed
        if PC_pop:
            id_ = subset[0]
            for i in subset:
                idx = np.where(np.asarray(spiking_neurons)==i)[0]  # spike times of given neuron (used for red dots on scatter)
                if len(idx) != 0:  # the randomly selected neuron spikes...
                    id_ = i
                    break
        else:  # for Bas. pop we always plot the same cell
            id_ = int(nBCs/2)  # fixed in simulations
            idx = np.where(np.asarray(spiking_neurons)==id_)[0]  # spike times of given neuron (used for red dots on scatter)
        t = StateM.t_ * 1000.  # *1000 ms convertion
        v = StateM[id_].vm*1000  # *1000 mV conversion

    fig = plt.figure(figsize=(10, 8))
    if StateM:
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 2])
    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    ax = fig.add_subplot(gs[0])
    ax.scatter(spike_times, spiking_neurons, c=color_, marker=".", s=size_)
    if StateM:
        if len(idx) != 0:
            if PC_pop:
                ax.scatter(spike_times[idx], spiking_neurons[idx], c="red", marker=".", s=size_, label=id_)
            else:
                ax.scatter(spike_times[idx], spiking_neurons[idx], c="red", marker=".", s=size_)
    ax.set_title("%s raster (last 100 ms)"%title_)
    ax.set_xlim([zoom_from, len_sim])
    ax.set_ylim([ymin, ymax])
    ax.set_ylabel("Neuron ID")
    if StateM and PC_pop:
        ax.legend()

    ax2 = fig.add_subplot(gs[1])
    sns.despine(ax=ax2)
    ax2.bar(np.linspace(zoom_from, len_sim, len(avg_rate)), avg_rate, width=bin_, align="edge", color=color_, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax2.set_xlim([zoom_from, len_sim])
    ax2.set_ylabel("Rate (Hz)")

    if StateM:
        ax3 = fig.add_subplot(gs[2])
        sns.despine(ax=ax3)
        if len(idx) != 0:
            ax3.plot(t[np.where((zoom_from <= t) & (t < len_sim))], v[np.where((zoom_from <= t) & (t < len_sim))], linewidth=2, c=color_,)
            tmp = spike_th_PC * np.ones_like(idx, dtype=np.float) if PC_pop else spike_th_BC * np.ones_like(idx, dtype=np.float)
            ax3.plot(spike_times[idx], tmp, c="red", marker=".", linewidth=0, label=id_)
        else:
            ax3.plot(t[np.where((zoom_from <= t) & (t < len_sim))], v[np.where((zoom_from <= t) & (t < len_sim))], linewidth=2, c=color_, label=id_)
        ax3.set_xlim([zoom_from, len_sim])
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Vm (mV)")
        ax3.legend()

    fig.tight_layout()
    fig_name = os.path.join(fig_dir, "%.2f_%s_zoomed.png"%(multiplier_, title_))
    fig.savefig(fig_name)

    if PC_pop:
         return subset


def plot_detailed(StateM, subset, multiplier_, plot_adaptation=True):
    """
    Saves figure with more detailes about some selected neurons
    :param StateM: Brian2 StateMonitor object
    :param subset: selected neurons to plot (max 5)
    :param multiplier_: naming parameter
    :param plot_adaptation: boolean flag for plotting adaptation var.
    """

    zoom_from = len_sim - 100  # ms

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    t = StateM.t_ * 1000.  # *1000 ms convertion
    for i in subset:
        ax.plot(t, StateM[i].vm*1000, linewidth=1.5, label="%i"%i)  # *1000 mV conversion
        if plot_adaptation:
            ax2.plot(t, StateM[i].w*1e12, linewidth=1.5, label="%i"%i)  # *1e12 pA conversion
        ax3.plot(t, (StateM[i].g_ampa + StateM[i].g_ampaMF), linewidth=1.5, label="%i"%i)
        ax4.plot(t, StateM[i].g_gaba, linewidth=1.5, label="%i"%i)

    ax.set_title("Membrane potential (last 100 ms)")
    ax.set_ylabel("V (mV)")
    ax.set_xlim([zoom_from, len_sim])
    ax.legend()

    ax2.set_title("Adaptation variable (last 100 ms)")
    ax2.set_ylabel("w (pA)")
    ax2.set_xlim([zoom_from, len_sim])
    if plot_adaptation:
        ax2.legend()

    ax3.set_title("Exc. inputs (last 100 ms)")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("g_AMPA (nS)")
    ax3.set_xlim([zoom_from, len_sim])
    ax3.legend()

    ax4.set_title("Inh. inputs (last 100 ms)")
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("g_GABA (nS)")
    ax4.set_xlim([zoom_from, len_sim])
    ax4.legend()

    sns.despine()
    fig.tight_layout()
    fig_name = os.path.join(fig_dir, "%.2f_PC_population_zoomed_detailed.png"%multiplier_)
    fig.savefig(fig_name)


def plot_LFP(t, LFP, f, Pxx, multiplier_):
    """
    Saves plot of the estimated LFP and it's power spectrum (see `detect_oscillations.py/analyse_estimated_LFP()`)
    :param t: time vector used for the plot
    :param LFP: estimated LFP
    :param f, Pxx:
    """

    try:
        Pxx_plot = np.zeros_like(Pxx)
        for i in range(Pxx.shape[0]):
            Pxx_plot[i, :] = 10 * np.log10(Pxx[i, :] / max(Pxx[i, :]))
        Pxx_plot_mean = np.mean(Pxx_plot, axis=0)
    except:
        Pxx_plot_mean = 10 * np.log10(Pxx / max(Pxx))

    f_ripple = f[np.where((150 < f) & (f < 220))]; Pxx_ripple_plot = Pxx_plot_mean[np.where((150 < f) & (f < 220))]
    f_gamma = f[np.where((30 < f) & (f < 100))]; Pxx_gamma_plot = Pxx_plot_mean[np.where((30 < f) & (f < 100))]

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])

    ax = fig.add_subplot(gs[0])
    ax.plot(t, LFP, color=(0.35, 0.35, 0.35))
    ax.set_title("Estimated LFP")
    ax.set_xlabel("Time (ms)")
    ax.set_xlim([t[0], t[-1]])
    ax.set_ylabel('"LFP (mV)" - currents summed from 400 PCs')

    ax2 = fig.add_subplot(gs[1])
    try:
        for Pxx_plot_tmp in Pxx_plot:
            ax2.plot(f, Pxx_plot_tmp, lw=0.5, color="gray", alpha=0.5)
    except:
        pass
    ax2.plot(f, Pxx_plot_mean, color="purple", marker="o")
    ax2.plot(f_ripple, Pxx_ripple_plot, "r-", marker="o", linewidth=1.5, label="ripple (150-220 Hz)")
    ax2.plot(f_gamma, Pxx_gamma_plot, "k-", marker="o", linewidth=1.5, label="gamma (30-100 Hz)")
    ax2.set_title("Power Spectrum Density")
    ax2.set_xlim([0, 500])
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("PSD (dB)")
    ax2.legend()

    sns.despine()
    fig.tight_layout()
    fig_name = os.path.join(fig_dir, "%.2f_LFP.png"%multiplier_)
    fig.savefig(fig_name)


def plot_step_sizes(gamma_LFP, step_sizes, avg_step_size, delta_t, fig_name):
    """
    Saves figure with step sizes within estimated trajectories
    :param gama_rate: gamma freq filtered LFP
    :param step_sizes: event step sized calculated from estimated trajectories
    :param avg_step_size: average step sizes calculated from distance and time of trajectories
    :param delta_t: width of time windows used (only for xlabel)
    :param fig_name: name of saved img
    """

    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(2, 1, 1)

    t_end = len(gamma_LFP)/10.  # hard coded for 10 kHz sampling rate...
    t_gamma = np.linspace(0, t_end, len(gamma_LFP))
    ax.plot(t_gamma, gamma_LFP, color="blue")
    ax.set_title("Background gamma")
    ax.set_xlim([0, t_end])
    ax.set_ylabel("PC rate in the gamma band")
    #ax.set_yticks([]); ax.set_yticklabels([])

    t = np.linspace(delta_t/2, t_end-delta_t/2, len(step_sizes))
    ax2 = fig.add_subplot(2, 1, 2)
    sns.despine()
    ax2.axhline(avg_step_size, color="gray", ls="--", zorder=1)
    ax2.plot(t, step_sizes, "k-", lw=1.5, zorder=1)
    ax2.scatter(t, step_sizes, color="red", zorder=2)
    ax2.set_title("Relative movement based on MLE trajectory")
    ax2.set_xlim([0, t_end]); ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("(Relative) Movement")

    fig.tight_layout()
    fig.align_ylabels([ax, ax2])
    fig.savefig(fig_name)
    plt.close(fig)


def plot_step_size_distr(step_sizes, avg_step_sizes, fig_name):
    """
    Saves figure with observed and predicted step sizes based on estimated trajectories
    :param step_sizes: event step sized calculated from estimated trajectories
    :param avg_step_sizes: average step sizes calculated from distance and time of trajectories
    :param fig_name: name of saved img
    """

    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()

    sns.distplot(step_sizes, ax=ax, kde=False, rug=False, norm_hist=True,
                 hist_kws={"color":"black", "alpha":0.8}, label="observed")
    ax.errorbar(np.mean(avg_step_sizes), 0.5, xerr=np.std(avg_step_sizes), color="red",
                fmt="o", capthick=2, capsize=5, label="predicted")
    ax.set_title("Distribution of step sizes")
    ax.set_xlabel("Step size")
    ax.set_ylabel("Prob")
    ax.legend()

    fig.savefig(fig_name)
    plt.close(fig)


def plot_step_size_phases(all_step_sizes, corresponding_phases, fig_name):
    """
    Plots 2D hist of step sizes and corresponding phases
    :params all_step_sizes, corresponding_phases: step sizes and corresponding_phases
    :param fig_name: name of saved img
    """

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.hexbin(corresponding_phases, all_step_sizes, gridsize=10, cmap="gray_r")
    fig.colorbar(i)
    ax.set_title("Slow gamma phase of step sizes")
    ax.set_xlabel("Phase (rad)")
    ax.set_ylabel("Step size")

    fig.savefig(fig_name)
    plt.close(fig)


def plot_STDP_rule(taup, taum, Ap, Am, save_name):
    """
    Saves plot of the STDP rule used for learning
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param taup, taum: time constant of weight change
    :param Ap, Am: max amplitude of weight change
    """

    delta_t = np.linspace(-150, 150, 1000)
    delta_w = np.where(delta_t>0, Ap*np.exp(-delta_t/taup), Am*np.exp(delta_t/taum))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()

    ax.plot(delta_t, delta_w, "b-", linewidth=2, label="STDP rule taup:%s(ms), Ap:%s"%(taup, Ap))
    ax.set_title("STDP curve")
    ax.set_xlabel("$\Delta t$ /post-pre/ (ms)")
    ax.set_ylabel("$\Delta w$ (nS)")
    if Ap == Am:
        ax.set_ylim([-Ap*0.05, Ap*1.05])
    else:
        ax.set_ylim([-Ap*1.05, Ap*1.05])
    ax.set_xlim([-150, 150])
    ax.axhline(0, ls="-", c="k")
    ax.legend()

    fig_name = os.path.join(fig_dir, "%s.png"%save_name)
    fig.savefig(fig_name)


def plot_wmx(wmx, save_name):
    """
    Saves figure with the weight matrix
    :param wmx: numpy array representing the weight matrix
    :param save_name: name of saved img
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(wmx*1e9, cmap="cividis", origin="lower", interpolation="nearest")  # nS conversion
    fig.colorbar(i)
    ax.set_title("Learned synaptic weights (nS)")
    ax.set_xlabel("Target neuron")
    ax.set_ylabel("Source neuron")

    fig_name = os.path.join(fig_dir, "%s.png"%save_name)
    fig.savefig(fig_name)


def plot_wmx_avg(wmx, n_pops, save_name):
    """
    Saves figure with the averaged weight matrix (better view as a whole)
    :param wmx: numpy array representing the weight matrix
    :param n_pops: number of populations
    :param save_name: name of saved img
    """

    assert nPCs % n_pops == 0

    pop_size = int(nPCs / n_pops)
    mean_wmx = np.zeros((n_pops, n_pops))
    for i in range(n_pops):
        for j in range(n_pops):
            tmp = wmx[int(i*pop_size):int((i+1)*pop_size), int(j*pop_size):int((j+1)*pop_size)]
            mean_wmx[i, j] = np.mean(tmp)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    i = ax.imshow(mean_wmx*1e9, cmap="cividis", origin="lower", interpolation="nearest")
    fig.colorbar(i)
    ax.set_title("Learned avg. synaptic weights (nS)")
    ax.set_xlabel("Target neuron")
    ax.set_ylabel("Source neuron")

    fig_name = os.path.join(fig_dir, "%s.png"%save_name)
    fig.savefig(fig_name)


def plot_w_distr(wmx, save_name):
    """
    Saves figure with the distribution of the weights
    :param wmx: numpy array representing the weight matrix
    :param save_name: name of saved img
    """

    # deleting nulls from wmx to plot the distribution of the weights
    wmx_nonzero = wmx[np.nonzero(wmx)]*1e9  # nS conversion
    log10wmx_nonzero = np.log10(wmx_nonzero)
    print("mean(nonzero weights): %s (nS)" % np.mean(wmx_nonzero))

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 1, 1)
    ax.hist(wmx_nonzero, bins=150)
    ax.set_title("Distribution of synaptic weights")
    ax.set_xlabel("Synaptic weights (nS)")
    ax.set_ylabel("Count")
    plt.yscale("log")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.hist(log10wmx_nonzero, bins=150, color="red")
    ax2.set_title("Distribution of synaptic weights")
    ax2.set_xlabel("log10(synaptic weights(nS))")
    ax2.set_ylabel("Count")
    plt.yscale("log")

    sns.despine()
    fig.tight_layout()
    fig_name = os.path.join(fig_dir, "%s.png"%save_name)
    fig.savefig(fig_name)


def save_selected_w(wmx, selection):
    """
    Saves the incomming weights of some selected neurons into a dictionary
    :param wmx: numpy array representing the weight matrix
    :param selection: numpa array of selected neuron IDs
    """

    return {i:wmx[:, i] for i in selection}


def plot_weights(incomming_weights, save_name):
    """
    Saves figure with some selected weights
    :param incomming_weights: dictionary storing the input weights of some neurons (see `save_selected_w()`)
    :param save_name: name of saved img
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()

    for i, val in incomming_weights.items():
        ax.plot(val*1e9, alpha=0.5, label="%i"%i)  # nS conversion
    ax.set_title("Incomming exc. weights")
    ax.set_xlabel("Neuron ID")
    ax.set_ylabel("Weight (nS)")
    ax.set_xlim([0, nPCs])
    ax.legend()

    fig_name = os.path.join(fig_dir, "%s.png"%save_name)
    fig.savefig(fig_name)


def plot_summary_replay(multipliers, replay, rates_PC, rates_BC):
    """
    Saves summary figure with replay (1/non) and avg. firing rates
    :param multipliers: wmx multipliers
    :param replay: replay list of 1/np.nans (has to be the same size as multipliers)
    :param rates_PC, rates_BC: avg. exc. and inh. firing rates (have to be the same size as multipliers)
    """

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 1, 1)
    sns.despine(ax=ax)
    ax.plot(multipliers, replay, "b-", linewidth=2, marker="*")
    ax.set_title("Replay")
    ax.set_xlim([multipliers[0], multipliers[-1]])
    ax.set_xticks(multipliers); ax.set_xticklabels(multipliers)
    ax.set_ylabel("Time (ms)")

    ax2 = fig.add_subplot(2, 1, 2)
    ax3 = ax2.twinx()
    sns.despine(ax=ax2)
    sns.despine(ax=ax3, right=False)
    ax2.plot(multipliers, rates_PC, "b-", linewidth=2, marker="*", label="PC rate")
    ax2.set_ylabel(ylabel="PC rate (Hz)", color="blue")
    ax3.plot(multipliers, rates_BC, "g-", linewidth=2, marker="*", label="BC rate")
    ax3.set_ylabel(ylabel="BC rate (Hz)", color="green")
    ax2.set_xlabel("Scale factors")
    ax2.set_xlim([multipliers[0], multipliers[-1]])
    ax2.set_xticks(multipliers); ax2.set_xticklabels(multipliers)
    ax2.set_title("Mean firing rates")
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax2.legend(h2+h3, l2+l3)

    fig.tight_layout()
    fig_name = os.path.join(fig_dir, "replay_rate.png")
    fig.savefig(fig_name)
    plt.close(fig)


def plot_summary_ripple(multipliers, ripple_freqs_PC, ripple_freqs_BC, ripple_freqs_LFP, ripple_powers_PC, ripple_powers_BC, ripple_powers_LFP):
    """
    Saves summary figure with ripple freq. and power
    :param multipliers: wmx multipliers
    :param ripple_freqs_PC, ripple_freqs_BC, ripple_freqs_LFP: exc., inh. and LFP ripple frequency (have to be the same size as multipliers)
    :param ripple_powers_PC, ripple_powers_BC, ripple_powers_LFP: exc., inh. and LFP ripple power (have to be the same size as multipliers)
    """

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(multipliers, ripple_freqs_PC, "b-", linewidth=2, marker="o", label="ripple freq (PC)")
    ax2 = ax.twinx()
    ax2.plot(multipliers, ripple_powers_PC, "r-", linewidth=2, marker="*", label="ripple power (PC)")
    ax.set_xlim([multipliers[0], multipliers[-1]])
    ax.set_xticks(multipliers); ax.set_xticklabels(multipliers)
    ax.set_ylabel(ylabel="Frequency (Hz)", color="blue")
    ax2.set_ylabel(ylabel="Power (%)", color="red")
    ax2.set_ylim([0, 100])
    ax.set_title("Ripple oscillation")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2)


    ax3 = fig.add_subplot(3, 1, 2)
    ax3.plot(multipliers, ripple_freqs_BC, "g-", linewidth=2, marker="o", label="ripple freq (BC)")
    ax4 = ax3.twinx()
    ax4.plot(multipliers, ripple_powers_BC, "r-", linewidth=2, marker="*", label="ripple power (BC)")
    ax3.set_xlim([multipliers[0], multipliers[-1]])
    ax3.set_xticks(multipliers); ax3.set_xticklabels(multipliers)
    ax3.set_ylabel(ylabel="Frequency (Hz)", color="green")
    ax4.set_ylabel(ylabel="Power (%)", color="red")
    ax4.set_ylim([0, 100])
    h3, l3 = ax3.get_legend_handles_labels()
    h4, l4 = ax4.get_legend_handles_labels()
    ax3.legend(h3+h4, l3+l4)

    ax5 = fig.add_subplot(3, 1, 3)
    ax5.plot(multipliers, ripple_freqs_LFP, color="purple", linewidth=2, marker="o", label="ripple freq (LFP)")
    ax6 = ax5.twinx()
    ax6.plot(multipliers, ripple_powers_LFP,  "r-", linewidth=2, marker="*", label="ripple power (LFP)")
    ax5.set_xlim([multipliers[0], multipliers[-1]])
    ax5.set_xticks(multipliers); ax5.set_xticklabels(multipliers)
    ax5.set_ylabel(ylabel="Frequency (Hz)", color="purple")
    ax6.set_ylabel(ylabel="Power (%)", color="red")
    ax6.set_ylim([0, 100])
    ax5.set_xlabel("Scale factors")
    h5, l5 = ax5.get_legend_handles_labels()
    h6, l6 = ax6.get_legend_handles_labels()
    ax5.legend(h5+h6, l5+l6)

    sns.despine(right=False)
    fig.tight_layout()
    fig_name = os.path.join(fig_dir, "ripple.png")
    fig.savefig(fig_name)
    plt.close(fig)


def plot_summary_gamma(multipliers, gamma_freqs_PC, gamma_freqs_BC, gamma_freqs_LFP, gamma_powers_PC, gamma_powers_BC, gamma_powers_LFP):
    """
    Saves summary figure with ripple freq. and power
    :param multipliers: wmx multipliers
    :param gamma_freqs_PC, gamma_freqs_BC, gamma_freqs_LFP: exc., inh. and LFP gamma frequency (have to be the same size as multipliers)
    :param gamma_powers_PC, gamma_powers_BC, gamma_powers_LFP: exc., inh. and LFP gamma power (have to be the same size as multipliers)
    """

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(3, 1, 1)
    ax.plot(multipliers, gamma_freqs_PC, "b-", linewidth=2, marker="o", label="gamma freq (PC)")
    ax2 = ax.twinx()
    ax2.plot(multipliers, gamma_powers_PC, "r-", linewidth=2, marker="*", label="gamma power (PC)")
    ax.set_xlim([multipliers[0], multipliers[-1]])
    ax.set_xticks(multipliers); ax.set_xticklabels(multipliers)
    ax.set_ylabel(ylabel="Frequency (Hz)", color="blue")
    ax2.set_ylabel(ylabel="Power (%)", color="red")
    ax2.set_ylim([0, 100])
    ax.set_title("Gamma oscillation")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2)


    ax3 = fig.add_subplot(3, 1, 2)
    ax3.plot(multipliers, gamma_freqs_BC, "g-", linewidth=2, marker="o", label="gamma freq (BC)")
    ax4 = ax3.twinx()
    ax4.plot(multipliers, gamma_powers_BC, "r-", linewidth=2, marker="*", label="gamma power (BC)")
    ax3.set_xlim([multipliers[0], multipliers[-1]])
    ax3.set_xticks(multipliers); ax3.set_xticklabels(multipliers)
    ax3.set_ylabel(ylabel="Frequency (Hz)", color="green")
    ax4.set_ylabel(ylabel="Power (%)", color="red")
    ax4.set_ylim([0, 100])
    h3, l3 = ax3.get_legend_handles_labels()
    h4, l4 = ax4.get_legend_handles_labels()
    ax3.legend(h3+h4, l3+l4)

    ax5 = fig.add_subplot(3, 1, 3)
    ax5.plot(multipliers, gamma_freqs_LFP, color="purple", linewidth=2, marker="o", label="gamma freq (LFP)")
    ax6 = ax5.twinx()
    ax6.plot(multipliers, gamma_powers_LFP, "r-", linewidth=2, marker="*", label="gamma power (LFP)")
    ax5.set_xlim([multipliers[0], multipliers[-1]])
    ax5.set_xticks(multipliers); ax5.set_xticklabels(multipliers)
    ax5.set_ylabel(ylabel="Frequency (Hz)", color="purple")
    ax6.set_ylabel(ylabel="Power (%)", color="red")
    ax6.set_ylim([0, 100])
    ax5.set_xlabel("Scale factors")
    h5, l5 = ax5.get_legend_handles_labels()
    h6, l6 = ax6.get_legend_handles_labels()
    ax5.legend(h5+h6, l5+l6)

    sns.despine(right=False)
    fig.tight_layout()
    fig_name = os.path.join(fig_dir, "gamma.png")
    fig.savefig(fig_name)
    plt.close(fig)


def plot_summary_AC(multipliers, max_acs_PC, max_acs_BC, max_acs_ripple_PC, max_acs_ripple_BC):
    """
    Saves summary figure with maximum autocorrelations
    :param multipliers: wmx multipliers
    :param max_acs_PC, max_acs_BC: max. exc. and inh. ACs (have to be the same size as multipliers)
    :param max_acs_ripple_PC, max_acs_ripple_BC: max. exc. and inh. ACs in ripple range (have to be the same size as multipliers)
    """

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(multipliers, max_acs_PC, "b-", linewidth=2, marker="*", label="PC")
    ax.plot(multipliers, max_acs_BC, "g-", linewidth=2, marker="*", label="BC")
    ax.set_xlim([multipliers[0], multipliers[-1]])
    ax.set_xticks(multipliers); ax.set_xticklabels(multipliers)
    ax.set_title("Maximum autocerrelations")
    ax.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(multipliers, max_acs_ripple_PC, "b-", linewidth=2, marker="*", label="PC")
    ax2.plot(multipliers, max_acs_ripple_BC, "g-", linewidth=2, marker="*", label="BC")
    ax2.set_xlim([multipliers[0], multipliers[-1]])
    ax2.set_xticks(multipliers); ax2.set_xticklabels(multipliers)
    ax2.set_title("Maximum autocerrelations in ripple range")
    ax2.set_xlabel("Scale factors")
    ax2.legend()

    sns.despine()
    fig.tight_layout()
    fig_name = os.path.join(fig_dir, "autocorrelations.png")
    fig.savefig(fig_name)
    plt.close(fig)


def plot_summary_BC(freqs, powers, xlabel, xticklabels, ylabel, yticklabels, save_name):
    """
    Saves figure with parameter sweeps from BC pop's
    :params freqs, powers: matrices with ripple frequencies and powers (see `optimization/analyse_BC_network`)
    :params xticklabels, yticklabels: params used to create the matrices (see `optimization/analyse_BC_network`)
    :params xlabel, ylabel: axlables
    :param save_name: name of saved img
    """

    fig = plt.figure(figsize=(14, 7))

    ax = fig.add_subplot(1, 2, 1)
    i = ax.imshow(freqs, cmap=plt.get_cmap("jet"), origin="lower", aspect="auto", interpolation=None)
    fig.colorbar(i)
    ax.set_title("Ripple oscillation freq. in BC network (Hz)")
    ax.set_xlabel(xlabel)
    ax.set_xticks(np.arange(0, len(xticklabels))); ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.set_yticks(np.arange(0, len(yticklabels))); ax.set_yticklabels(yticklabels)

    ax2 = fig.add_subplot(1, 2, 2)
    i2 = ax2.imshow(powers, cmap=plt.get_cmap("jet"), origin="lower", aspect="auto", interpolation=None)
    fig.colorbar(i2); i2.set_clim(0, 100)
    ax2.set_title("Ripple oscillation power in BC network")
    ax2.set_xlabel(xlabel)
    ax2.set_xticks(np.arange(0, len(xticklabels))); ax2.set_xticklabels(xticklabels)
    ax2.set_ylabel(ylabel)
    ax2.set_yticks(np.arange(0, len(yticklabels))); ax2.set_yticklabels(yticklabels)

    fig.tight_layout()
    fig_name = os.path.join(fig_dir, "%s.png"%save_name)
    fig.savefig(fig_name)


def plot_evolution(ngen, min_fit, mean_fit, std_fit, save_name):
    """
    Saves figure with the evolution of fittnes error (see: `optimization/optimize_network`)
    :param ngen: number of generations
    :param min_fit: minimum of fitting errors (see bpop: _,_,log,_ = opt.run())
    :param mean_fit: mean of fitting errors (see bpop: _,_,log,_ = opt.run())
    :param std_fit: standard deviation of fitting errors (see bpop: _,_,log,_ = opt.run())
    :param save_name: name of saved img
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()

    ax.plot(ngen, mean_fit, "k-", linewidth=2, label="pop. average")
    ax.fill_between(ngen, mean_fit - std_fit, mean_fit + std_fit, color="lightgray", linewidth=1.5, label="pop. std")
    ax.plot(ngen, min_fit, "r-", linewidth=2, label="pop. minimum")
    ax.set_xlabel("#Generation")
    ax.set_xlim([1, max(ngen)])
    ax.set_ylabel("Fittnes score")
    ax.legend()

    fig_name = os.path.join(fig_dir, "%s.png"%save_name)
    fig.savefig(fig_name)


def plot_SS_voltage(t, v, SS_voltage, current):
    """
    Saves figure with SS voltage after current injection
    :param t,v: time and voltage
    :param SS_voltage: steady state voltage reached
    :param current: input current
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine()

    ax.plot(t, v, linewidth=1.5, label="V_m")
    ax.plot(np.linspace(800, 1000, 200), SS_voltage*np.ones(200), linewidth=1.5, label="V_SS: %.3f mV"%SS_voltage)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Memb. pot. (mV)")
    ax.set_title("Cell with %s pA input"%current)
    ax.legend()

    fig_name = os.path.join(fig_dir, "clamp_%s.png"%current)
    fig.savefig(fig_name)


def plot_avg_EPS(t, EPSPs, EPSP, EPSCs, EPSC, mean_weight, save_name):
    """
    Saves figure with average EPSP and EPSC
    :param t: time
    :param EPSPs, EPSP: array of EPSPs from random weights and EPSP for the average weight
    :param EPSCs, EPSC: array of EPSCs from random weights and EPSC for the average weight
    :param mean_weight: mean of all nonzero weights
    :param save_name: name of saved img
    """

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(t, np.mean(EPSPs, axis=0), "b-", label="mean of %i random weights"%EPSPs.shape[0])
    ax.plot(t, EPSP, "g-", label="mean of all weights (%f nS)"%mean_weight)
    ax.set_title("average EPSP")
    ax.set_xlim([0, 400])
    ax.set_ylabel("EPSP (mV)")
    ax.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    sns.despine()
    ax2.plot(t, np.mean(EPSCs, axis=0), "b-", label="mean of %i random weights"%EPSCs.shape[0])
    ax2.plot(t, EPSC, "g-", label="mean of all weights (%f nS)"%mean_weight)
    ax2.set_title("average EPSC")
    ax2.set_xlabel("Time (ms)")
    ax2.set_xlim([0, 400])
    ax2.set_ylabel("EPSC (pA)")
    ax2.legend()

    fig.tight_layout()
    fig_name = os.path.join(fig_dir, "%s.png"%save_name)
    fig.savefig(fig_name)


def plot_EPS_dist(peak_EPSPs, peak_EPSCs, save_name):
    """
    Saves violin plots with EPSP and EPSC distributions
    :param peak_EPSPs, peak_EPSCs: vectors of peakEPSP & EPSC values
    :param save_name: name of saved img
    """

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 1, 1)
    ax.violinplot(peak_EPSPs, vert=False, showmeans=True, showextrema=False, showmedians=False,
                  points=peak_EPSPs.shape[0], bw_method="silverman")
    ax.set_title("%i random EPSPs (mean: %f mV)"%(peak_EPSPs.shape[0], np.mean(peak_EPSPs)))
    ax.set_xlabel("EPSP (mV)")
    ax.set_yticks([])

    ax2 = fig.add_subplot(2, 1, 2)
    sns.despine()
    ax2.violinplot(peak_EPSCs, vert=False, showmeans=True, showextrema=False, showmedians=False,
                   points=peak_EPSCs.shape[0], bw_method="silverman")
    ax2.set_title("%i random EPSCs (mean: %f pA)"%(peak_EPSCs.shape[0], np.mean(peak_EPSCs)))
    ax2.set_xlabel("EPSC (pA)")
    ax2.set_yticks([])

    fig.tight_layout()
    fig_name = os.path.join(fig_dir, "%s.png"%save_name)
    fig.savefig(fig_name)


# not used in the final version...
def plot_learned_EPSPs(delta_ts, EPSPs, save_name):
    """
    Saves plot with EPSPs after learning (via STDP)
    :param delta_ts: list of $delta$t intervals between pre and post spikes (in ms)
    :param EPSPs: dictonary with keys of delta_ts and EPSP traces as values
    :param save_name: name of saved img
    """

    assert len(delta_ts) == 8, "plot works only for 2*4 delta_ts..."
    t = EPSPs["t"]

    fig = plt.figure(figsize=(20, 8))

    for i, delta_t in enumerate(delta_ts):
        ax = fig.add_subplot(2, 4, i+1)
        ax.plot(t, EPSPs["baseline"], "k--", linewidth=1.5)
        col = "b" if delta_t > 0 else "r"
        ax.plot(t, EPSPs[delta_t], color=col, linewidth=2)
        ax.set_title("$\Delta t$ /post-pre/ = %s (ms)"%delta_t)
        ax.set_xlim([0, 400])
        ax.set_ylabel("EPSP (mV)")
        if i >= 4:
            ax.set_xlabel("Time (ms)")

    sns.despine()
    fig.tight_layout()
    fig_name = os.path.join(fig_dir, "%s.png"%save_name)
    fig.savefig(fig_name)


# not used in the final version...
def plot_compare_STDP_to_orig(EPSP_changes, orig_data, save_name, orig_exp_fit=None, sim_exp_fit=None):
    """
    Saves plot based on Figure 1 i) in Mishra et al. 2016 - 10.1038/ncomms11552, extended with the in silico EPSP changes (with fixed cell model and STDP rule)
    :param EPSP_changes: dictonary with keys of delta_ts and values of EPSP changes in percentage
    :param orig_data: dictionary of original values (might not be straightforwad to understand)
    :param save_name: name of saved img
    :param orig_exp_fit: optional dictionary with taup, taum: time constant of weight change & Ap, Am: max amplitude of weight change (fitted to original data)
    :param sim_exp_fit: same dictionary as above (fitted to simulated data)
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine(right=False)

    # plot original data
    ax.plot(orig_data["time(ms)"], orig_data["mean(%)"], "ko", markersize=6, label="original (in vitro) data")
    ax.errorbar(orig_data["time(ms)"], orig_data["mean(%)"], yerr=orig_data["sem"], fmt="none", ecolor="k")
    # plot fitted Gaussian (to original data)
    t_ = np.arange(-150, 150, 0.1)
    fit_G = orig_data["gauss_a"] * np.exp(-((t_ - orig_data["gauss_b"])**2) / (2*orig_data["gauss_c"]**2))
    ax.plot(t_, fit_G, "r-", linewidth=2, label="Gaussian fit (orig. data)")
    if orig_exp_fit:  # plot exponential fit (to original data)
        fit_E = np.where(t_>0, orig_exp_fit["Ap"]*np.exp(-t_/orig_exp_fit["taup"]), orig_exp_fit["Am"]*np.exp(t_/orig_exp_fit["taum"]))
        ax.plot(t_, fit_E, color="orange", linewidth=2, label="exponential fit (orig. data) taup:%.3f(ms)"%orig_exp_fit["taup"])
    # plot in silico data
    ax2 = ax.twinx()
    ax2.plot(EPSP_changes["time"], EPSP_changes["change"], "go", markersize=6, label="simulated data")
    if sim_exp_fit:  # plot exponential fit (to simulated data)
        fit_E = np.where(t_>0, sim_exp_fit["Ap"]*np.exp(-t_/sim_exp_fit["taup"]), sim_exp_fit["Am"]*np.exp(t_/sim_exp_fit["taum"]))
        ax2.plot(t_, fit_E, "g-", linewidth=2, label="exponential fit (sim data) taup:%.3f(ms)"%sim_exp_fit["taup"])

    ax.set_title("Compare in vitro and in silico EPSP changes")
    ax.set_xlabel("$\Delta t$ /post-pre/ (ms)")
    ax.set_xlim([-150, 150])
    ax.set_ylabel("Change in EPSP amplitude (%)", color="black")
    ax2.set_ylabel("Change in EPSP amplitude (%)", color="green")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2)

    fig_name = os.path.join(fig_dir, "%s.png"%save_name)
    fig.savefig(fig_name)


# not used in the final version...
def plot_STDP2(STDP_params, sim_exp_fit, save_name):
    """
    Saves plot of the STDP rule used for learning
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param STDP_params: dictionary with taup, taum: time constant of weight change & Ap, Am: max amplitude of weight change (used to specify STDP rule for learning)
    :param sim_exp_fit: same dictionary as above (fitted to EPSP % changes)
    """

    delta_t = np.linspace(-150, 150, 1000)
    delta_w_rule = np.where(delta_t<0, STDP_params["Am"]*np.exp(delta_t/STDP_params["taum"]), None)
    delta_w_fitted = np.where(delta_t>0, sim_exp_fit["Ap"]*np.exp(-delta_t/sim_exp_fit["taup"]), None)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    sns.despine(right=False)

    ax.plot(delta_t, delta_w_rule, "b-", linewidth=2, label="STDP rule taup:%s(ms), Ap:%s"%(STDP_params["taup"], STDP_params["Ap"]))
    ax.axhline(0, ls="-", c="k")
    ax.set_title("STDP")
    ax.set_xlabel("$\Delta t$ /post-pre/ (ms)")
    ax.set_xlim([-150, 150])
    ax.set_ylabel("$\Delta w$ (nS)", color="blue")
    ax2 = ax.twinx()
    ax2.plot(delta_t, delta_w_fitted, "g-", linewidth=2, label="fitted 'STDP' taup:%.3f(ms), Ap:%.3f"%(sim_exp_fit["taup"], sim_exp_fit["Ap"]))
    ax2.set_ylabel("Change in EPSP amplitude (%)", color="green")
    ax.set_ylim([-1*STDP_params["Ap"]*0.05, STDP_params["Ap"]*1.05])
    ax2.set_ylim([-1*sim_exp_fit["Ap"]*0.05, sim_exp_fit["Ap"]*1.05])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2)

    fig_name = os.path.join(fig_dir, "%s_sym.png"%save_name)
    fig.savefig(fig_name)
