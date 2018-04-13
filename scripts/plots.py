#!/usr/bin/python
# -*- coding: utf8 -*-
"""
helper file to plot dynamics (and the weight matrix)... and couple of other things
authors: Bence Bagi, András Ecker, last update: 09.2017
"""

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import brian2.monitors.statemonitor

#sns.set_context("paper")
sns.set_style("white")

SWBasePath = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-2])
figFolder = os.path.join(SWBasePath, "figures")

# spike thresholds
v_spike_Pyr = 19.85800072  # (optimized by Bence)
v_spike_Bas = -17.48690645  # (optimized by Bence)

nPC = 8000
nBC = 150
len_sim = 10000  # ms


def _avg_rate(rate, bin_, zoomed=False):
    """
    helper function to bin rate for bar plots
    :param rate: np.array representing firing rates (hard coded for 10000ms simulations)
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


def plot_raster_ISI(spikeTimes, spikingNeurons, rate, hist, color_, multiplier_):
    """
    saves figure with raster plot and ISI distribution
    (note: the main reason of this function is that Brian2 doesn't have ISIHistogramMonitor and the corresponding plot)
    :param spikeTimes, spikingNeurons: used for raster plot - precalculated by `detect_oscillation.py/preprocess_spikes()`
    :param hist: used for plotting InterSpikeInterval histogram
                 result of a numpy.histogram call: [hist, bin_edges] (see `detect_oscillations.py/preprocess_monitors()`)
    :param color_, multiplier_: outline and naming parameters
    """

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 2])
    
    ax = fig.add_subplot(gs[0])
    ax.scatter(spikeTimes, spikingNeurons, c=color_, marker='.', linewidth=0)
    ax.set_title("Pyr_population raster")
    ax.set_xlim([0, len_sim])    
    ax.set_ylim([0, nPC])
    ax.set_ylabel("Neuron number")
    
    bin_ = 20
    avg_rate = _avg_rate(rate, bin_)
     
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(np.linspace(0, len_sim, len(avg_rate)), avg_rate, width=bin_, align="edge", color=color_, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax2.set_xlim([0, len_sim])
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Rate (Hz)")

    ax3 = fig.add_subplot(gs[2])
    ax3.bar(hist[1][:-1], hist[0], width=50, align="edge", color=color_, edgecolor="black", linewidth=0.5, alpha=0.9)  # width=50 comes from bins=20 
    ax3.axvline(150, ls='--', c="gray", label="ROI for replay analysis")
    ax3.axvline(850, ls='--', c="gray")
    ax3.set_title("Pyr_population ISI distribution")
    ax3.set_xlabel("$\Delta t$ (ms)")
    ax3.set_xlim([0, 1000])
    ax3.set_ylabel("Count")
    ax3.legend()

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s*.png"%(multiplier_))
    fig.savefig(figName)


def plot_PSD(rate, rAC, f, Pxx, title_, color_, multiplier_, TFR=False, tfr=None, t=None, freqs=None, fs=None):
    """
    saves figure with rate, auto-correlation plot, PSD and optionally TFR
    :param rate: firing rate - precalculated by `detect_oscillation.py/preprocess_spikes()`
    :param rAC: auto-correlation function of the rate (returned by `detect_oscillation.py/analyse_rate()`)
    :param f, Pxx (returned by `detect_oscillation.py/analyse_rate()`) see more: http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.signal.welch.html
    :param title_, color_, multiplier: outline and naming parameters
    :param TFR: bool - to plot time frequency representation
    :param tfr, t, freqs: calculated TRF and time points and frequencies used - returned by `tftb.processing.Scalogram()`
    :param fs: sampling freq to scale normalized freqs from `tftb.processing.Scalogram()`
    """

    # bin rate
    bin_ = 20
    avg_rate = _avg_rate(rate, bin_)
    # get AC in 'interesting' range
    rEACPlot = rAC[2:201] # 500 - 5 Hz interval
    # get gamma and ripple range
    f = np.asarray(f)    
    fRipple = f[np.where((160 < f) & (f < 230))]; PxxRipple = Pxx[np.where((160 < f) & (f < 230))]
    fGamma = f[np.where((30 < f) & (f < 100))]; PxxGamma = Pxx[np.where((30 < f) & (f < 100))]
    PxxPlot = 10 * np.log10(Pxx / max(Pxx))
    PxxRipplePlot = 10 * np.log10(PxxRipple / max(Pxx))
    PxxGammaPlot = 10 * np.log10(PxxGamma / max(Pxx))
    
    if TFR:  # 4 subplots with Scalogram
    
        tfr = np.abs(tfr) ** 2
        tfr[tfr <= np.amax(tfr) * 0.05] = 0.0
        t_, f_ = np.meshgrid(t, freqs*fs)
        
        fig = plt.figure(figsize=(15, 8))
        
        ax = fig.add_subplot(2, 2, 1)    
        ax2 = fig.add_subplot(2, 2, 2)        
        ax3 = fig.add_subplot(2, 2, 4)  # name changed to match 3 subplot version
        
        ax4 = fig.add_subplot(2, 2, 3)  # name changed       
        ax4.contour(t_, f_, tfr, 20, cmap=plt.get_cmap("jet"))
        ax4.grid(True)
        ax4.set_title("TFR (Morlet scalogram)")
        ax4.set_xlabel("Time (ms)")
        ax4.set_xlim([0, len_sim])
        ax4.set_ylabel("Frequency (Hz)")
        ax4.set_ylim([2, 250])
    
    else:  # 3 subplots as previously
    
        fig = plt.figure(figsize=(10, 8))
        
        ax = fig.add_subplot(3, 1, 1)
        ax.set_xlabel("Time (ms)")
        ax2 = fig.add_subplot(3, 1, 2)
        ax3 = fig.add_subplot(3, 1, 3)
       
    ax.bar(np.linspace(0, len_sim, len(avg_rate)), avg_rate, width=bin_, align="edge", color=color_, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax.set_xlim([0, len_sim])
    ax.set_title("%s rate"%title_)
    ax.set_ylabel("Rate (Hz)")
    
    ax2.plot(np.linspace(2, 200, len(rEACPlot)), rEACPlot, color=color_)
    ax2.set_title("Autocorrelogram 2-200 ms")
    ax2.set_xlabel("Time (ms)")
    ax2.set_xlim([2, 200])
    ax2.set_ylabel("AutoCorrelation")
    
    ax3.plot(f, PxxPlot, color=color_, marker='o')
    ax3.plot(fRipple, PxxRipplePlot, 'r-', marker='o', linewidth=1.5, label="ripple (160-230Hz)")
    ax3.plot(fGamma, PxxGammaPlot, 'k-', marker='o', linewidth=1.5, label="gamma (30-100Hz)")
    ax3.set_title("Power Spectrum Density")
    ax3.set_xlim([0, 500])
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("PSD (dB)")
    ax3.legend()      

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s_%s.png"%(multiplier_, title_))
    fig.savefig(figName)


def _select_subset(selection, ymin, ymax):
    """
    helper function to select a subset of neurons for plotting more detailes (the subset is from the ones spiking in the last 100ms - see `plot_zoomed()`)
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
        subset = [500, 1999, 4000, 6000, 7498]
    return subset


def plot_zoomed(spikeTimes, spikingNeurons, rate, title_, color_, multiplier_, Pyr_pop=True, sm=None, selection=None):
    """
    saves figure with zoomed in raster, rate and optionally a trace (last 100ms)
    :param spikeTimes, spikingNeurons: used for raster plot - precalculated by `detect_oscillation.py/preprocess_spikes()`
    :param rate: firing rate - precalculated by detect_oscillation.py/preprocess_spikes
    :param title_, color_, linespec_, multiplier_: outline and naming parameters
    :param Pyr_pop: flag for calculating and returning ymin and ymax (and zooming in the plot)
    :param sm: Brian2 StateMonitor object (could be more elegant...)
    :param selection: np.array of recorded neurons (used only if Pyr_pop is true)
    return subset: see `_select_subset()`
    """
    
    zoom_from = len_sim - 100  # ms

    # get last 100ms of raster
    ROI = [np.where(spikeTimes > zoom_from)[0]]  # hard coded for 10000ms...
    spikeTimes = spikeTimes[ROI]; spikingNeurons = spikingNeurons[ROI]
    
    # average rate 
    bin_ = 1.5
    avg_rate = _avg_rate(rate, bin_, zoomed=True)
    
    # set boundaries
    if Pyr_pop:        
        ymin = spikingNeurons.min()-5 if spikingNeurons.min()-5 > 0 else 0
        ymax = spikingNeurons.max()+5 if spikingNeurons.max()+5 < nPC else nPC
        subset = _select_subset(selection, ymin, ymax)
    else:
        ymin = 0; ymax = nBC
    
    # select trace to plot
    if sm:
        if Pyr_pop:
            id_ = subset[0]
            for i in subset:
                idx = np.where(np.asarray(spikingNeurons)==i)[0]  # spike times of given neuron (used for red dots on scatter)
                if len(idx) != 0:  # the randomly selected neuron spikes...
                    id_ = i
                    break             
        else:  # for Bas. pop we always plot the same
            id_ = nBC/2  # fixed in simulations
            idx = np.where(np.asarray(spikingNeurons)==id_)[0]  # spike times of given neuron (used for red dots on scatter)
    
        # get trace from monitor
        if type(sm) is brian2.monitors.statemonitor.StateMonitor:
            t = sm.t_ * 1000.  # *1000 ms convertion
            v = sm[id_].vm*1000  # *1000 mV conversion        
        else:
            t = sm.times*1000.  # *1000 ms convertion
            v = sm["vm", id_]*1000  # *1000 mV conversion

    fig = plt.figure(figsize=(10, 8))
    if sm:
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 2])
    else:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    
    ax = fig.add_subplot(gs[0])
    ax.scatter(spikeTimes, spikingNeurons, c=color_, marker='.', linewidth=0)
    if sm:
        if len(idx) != 0:
            if Pyr_pop:
                ax.scatter(spikeTimes[idx], spikingNeurons[idx], c="red", marker='.', linewidth=0, label=id_)
            else:
                ax.scatter(spikeTimes[idx], spikingNeurons[idx], c="red", marker='.', linewidth=0)
    ax.set_title("%s raster (last 100 ms)"%title_)
    ax.set_xlim([zoom_from, len_sim])
    ax.set_ylim([ymin, ymax])
    ax.set_ylabel("Neuron number")
    if sm and Pyr_pop:
        ax.legend()
 
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(np.linspace(zoom_from, len_sim, len(avg_rate)), avg_rate, width=bin_, align="edge", color=color_, edgecolor="black", linewidth=0.5, alpha=0.9)
    ax2.set_xlim([zoom_from, len_sim]) 
    ax2.set_ylabel("Rate (Hz)")
    
    if sm:   
        ax3 = fig.add_subplot(gs[2])
        if len(idx) != 0:
            ax3.plot(t[np.where((zoom_from <= t) & (t < len_sim))], v[np.where((zoom_from <= t) & (t < len_sim))], linewidth=2, c=color_,)
            tmp = v_spike_Pyr * np.ones_like(idx, dtype=np.float) if Pyr_pop else v_spike_Bas * np.ones_like(idx, dtype=np.float)
            ax3.plot(spikeTimes[idx], tmp, c="red", marker='.', linewidth=0, label=id_)
        else:
            ax3.plot(t[np.where((zoom_from <= t) & (t < len_sim))], v[np.where((zoom_from <= t) & (t < len_sim))], linewidth=2, c=color_, label=id_)
        ax3.set_xlim([zoom_from, len_sim])
        ax3.set_xlabel("Time (ms)")
        ax3.set_ylabel("Vm (mV)")
        ax3.legend()

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s_%s_zoomed.png"%(multiplier_, title_))
    fig.savefig(figName)

    if Pyr_pop:
         return subset


def plot_detailed(msM, subset, multiplier_, plot_adaptation=True, new_network=False):
    """
    saves figure with more detailes about some selected neurons
    :param msM: Brian MultiStateMonitor object or Brian2 StateMonitor object (could be more elegant...)
    :param subset: selected neurons to plot (max 5)
    :param multiplier_: naming parameter
    :param plot_adaptation: boolean flag for plotting adaptation var.
    :param new_network: boolean flag for plotting AMPA conductance (in the new network it's a sum)
    """
    
    zoom_from = len_sim - 100  # ms

    fig = plt.figure(figsize=(15, 8))
    #fig.suptitle("Detailed plots of selected vars. (Pyr. pop)")
    ax = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    

    if type(msM) is brian2.monitors.statemonitor.StateMonitor:
        t = msM.t_ * 1000.  # *1000 ms convertion
        for i in subset:
            ax.plot(t, msM[i].vm*1000, linewidth=1.5, label="%i"%i)  # *1000 mV conversion
            if plot_adaptation:
                ax2.plot(t, msM[i].w*1e12, linewidth=1.5, label="%i"%i)  # *1e12 pA conversion
            if new_network:  # diff exc->exc synapses (g_ampa is a sum of them in the new network)
                ax3.plot(t, (msM[i].g_ampa + msM[i].g_ampaMF), linewidth=1.5, label="%i"%i)
            else:
                ax3.plot(t, msM[i].g_ampa, linewidth=1.5, label="%i"%i)
            ax4.plot(t, msM[i].g_gaba, linewidth=1.5, label="%i"%i)
    else:
        t = msM.times*1000.  # *1000 ms convertion
        for i in subset:
            ax.plot(t, msM["vm", i]*1000, linewidth=1.5, label="%i"%i)  # *1000 mV conversion
            if plot_adaptation:
                ax2.plot(t, msM["w", i]*1e12, linewidth=1.5, label="%i"%i)  # *1e12 pA conversion
            if new_network:  # diff exc->exc synapses (g_ampa is a sum of them in the new network)
                ax3.plot(t, (msM["g_ampa", i] + msM["g_ampaMF", i]), linewidth=1.5, label="%i"%i)
            else:
                ax3.plot(t, msM["g_ampa", i], linewidth=1.5, label="%i"%i)
            ax4.plot(t, msM["g_gaba", i], linewidth=1.5, label="%i"%i)

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

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s_Pyr_population_zoomed_detailed.png"%(multiplier_))
    fig.savefig(figName)


def plot_STDP_rule(taup, taum, Ap, Am, saveName_):
    """
    Plots the STDP rule used for learning
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param taup, taum: time constant of weight change
    :param Ap, Am: max amplitude of weight change
    :return mode: just for saving conventions (see other wmx figures)
    """

    # automate naming
    if Ap == Am:
        mode = "sym"
    elif Ap == Am*-1:
        mode = "asym"
    elif np.abs(Ap) != np.abs(Am):
        print "naming conventions won't work!"
        mode = "tmp"
    print "========== STDP rule: %s =========="%mode

    delta_t = np.linspace(-150, 150, 1000)
    delta_w = np.where(delta_t>0, Ap*np.exp(-delta_t/taup), Am*np.exp(delta_t/taum))

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(delta_t, delta_w, "b-", linewidth=2, label="STDP rule taup:%s(ms), Ap:%s"%(taup, Ap))
    ax.set_title("STDP curve")
    ax.set_xlabel("$\Delta t$ /post-pre/ (ms)")
    ax.set_ylabel("$\Delta w$ (nS)")
    if mode == "asym":
        ax.set_ylim([-Ap*1.05, Ap*1.05])
    elif mode == "sym":
        ax.set_ylim([-Ap*0.05, Ap*1.05])
    ax.set_xlim([-150, 150])
    ax.axhline(0, ls='-', c='k')
    ax.legend()

    figName = os.path.join(figFolder, "%s_%s.png"%(saveName_, mode))
    fig.savefig(figName)

    return mode


def plot_wmx(wmx, saveName_):
    """
    saves figure with the weight matrix
    :param wmx: ndarray representing the weight matrix
    :param saveName_: name of saved img
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    i = ax.imshow(wmx, cmap=plt.get_cmap("jet"))
    i.set_interpolation("nearest")  # set to "None" to less pixels and smooth, nicer figure
    fig.colorbar(i)
    ax.set_title("Learned synaptic weights")
    ax.set_xlabel("target neuron")
    ax.set_ylabel("source neuron")

    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)


def plot_wmx_avg(wmx, nPop, saveName_):
    """
    saves figure with the averaged weight matrix (better view as a whole)
    :param wmx: ndarray representing the weight matrix
    :param nPop: number of populations
    :param saveName_: name of saved img
    """

    assert nPC % nPop == 0

    popSize = int(nPC / nPop)
    wmxM = np.zeros((100, 100))
    for i in range(nPop):
        for j in range(nPop):
            tmp = wmx[int(i*popSize):int((i+1)*popSize), int(j*popSize):int((j+1)*popSize)]
            wmxM[i, j] = np.mean(tmp)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    
    i = ax.imshow(wmxM, cmap=plt.get_cmap("jet"))
    i.set_interpolation("nearest")  # set to "None" to less pixels and smooth, nicer figure
    fig.colorbar(i)
    ax.set_title("Learned synaptic weights (avg.)")
    ax.set_xlabel("target neuron")
    ax.set_ylabel("source neuron")

    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)


def plot_w_distr(wmx, saveName_):
    """
    saves figure with the distribution of the weights
    :param wmx: ndarray representing the weight matrix
    :param saveName_: name of saved img
    """

    # deleting nulls from wmx to plot the distribution of the weights
    wmx = wmx[np.nonzero(wmx)]*1e9  # nS conversion
    log10wmx = np.log10(wmx)
    print "mean(nonzero weights): %s (nS)"%np.mean(wmx)

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 1, 1)
    ax.hist(wmx, bins=150)
    ax.set_title('Distribution of synaptic weights')
    ax.set_xlabel('pyr-pyr synaptic weight strength (nS)')
    ax.set_ylabel('# of synapses (on logarithmic scale)')
    plt.yscale('log')

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.hist(log10wmx, bins=150, color='red')
    ax2.set_title('Distribution of synaptic weights')
    ax2.set_xlabel('log10(pyr-pyr synaptic weight strength) (nS)')
    ax2.set_ylabel('# of synapses (on logarithmic scale)')
    plt.yscale('log')

    fig.tight_layout()

    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)


def save_selected_w(Wee, selection):
    """saves the incomming weights of some selected neurons"""
    w = {}
    for i in selection:
        w[i] = Wee[:, i]
    return w


def plot_weights(dWee, saveName_):
    """
    saves figure with some selected weights
    :param dW: dictionary storing the input weights of some neurons (see save_selected_w())
    :param saveName_: name of saved img
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    for i, val in dWee.items():
        ax.plot(val, alpha=0.5, label="%i"%i)

    ax.set_title("Incomming exc. weights")
    ax.set_xlabel("#Neuron")
    ax.set_ylabel("Weight (nS)")
    ax.set_xlim([0, nPC])
    ax.legend()

    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)


def plot_summary_replay(multipliers, replay_interval, rateE, rateI):
    """
    saves summary figure with avg. replay interval and avg. firing rates
    :param multipliers: wmx multipliers
    :param replay _interval: replay intervals (has to be the same size as multipliers)
    :param rateE, rateI: avg. exc. and inh. firing rates (have to be the same size as multipliers)
    """

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(multipliers, replay_interval, linewidth=2, marker='|')
    ax.set_title("Average replay interval")
    ax.set_xlim([multipliers[0], multipliers[-1]])
    ax.set_ylabel("Time (ms)")

    ax2 = fig.add_subplot(2, 1, 2)
    ax3 = ax2.twinx()
    ax2.plot(multipliers, rateE, "b-", linewidth=2, marker="|", label="PC rate")
    ax2.set_ylabel(ylabel="Exc. rate (Hz)", color="blue")
    ax3.plot(multipliers, rateI, "g-", linewidth=2, marker="|", label="BC rate")
    ax3.set_ylabel(ylabel="Inh rate (Hz)", color="green")
    ax2.set_xlabel("scale factors")
    ax2.set_xlim([multipliers[0], multipliers[-1]])
    ax2.set_title("Mean firing rates")
    h2, l2 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax2.legend(h2+h3, l2+l3)

    fig.tight_layout()
    figName = os.path.join(figFolder, "replay_rate.png")
    fig.savefig(figName)   


def plot_summary_AC(multipliers, maxACE, maxACI, maxRACE, maxRACI):
    """
    saves summary figure with maximum autocorrelations
    :param multipliers: wmx multipliers
    :param maxACE, maxACI: max. exc. and inh. ACs (have to be the same size as multipliers)
    :param maxRACE, maxRACI: max. exc. and inh. ACs in ripple range (have to be the same size as multipliers)
    """

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(multipliers, maxACE, "b-", linewidth=2, marker="|", label="PC (exc.)")
    ax.plot(multipliers, maxACI, "g-", linewidth=2, marker="|", label="BC (inh.)")
    ax.set_xlim([multipliers[0], multipliers[-1]])
    ax.set_title("Maximum autocerrelations")
    ax.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(multipliers, maxRACE, "b-", linewidth=2, marker="|", label="PC (exc.)")
    ax2.plot(multipliers, maxRACI, "g-", linewidth=2, marker="|", label="BC (inh.)")
    ax2.set_xlim([multipliers[0], multipliers[-1]])
    ax2.set_title("Maximum autocerrelations in ripple range")
    ax2.set_xlabel("scale factors")
    ax2.legend()

    fig.tight_layout()
    figName = os.path.join(figFolder, "autocorrelations.png")
    fig.savefig(figName)
    

def plot_summary_ripple(multipliers, rippleFE, rippleFI, ripplePE, ripplePI):
    """
    saves summary figure with ripple freq. and power
    :param multipliers: wmx multipliers
    :param rippleFE, rippleFI: exc. and inh. ripple frequency (have to be the same size as multipliers)
    :param ripplePE, ripplePI: exc. and inh. ripple power (have to be the same size as multipliers)
    """

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(multipliers, rippleFE, "b-", linewidth=2, marker="o", label="ripple freq (exc.)")
    ax2 = ax.twinx()
    ax2.plot(multipliers, ripplePE, "r-", linewidth=2, marker="|", label="ripple power (exc.)")
    ax.set_xlim([multipliers[0], multipliers[-1]])
    ax.set_ylabel(ylabel="freq (Hz)", color="blue")
    ax.set_ylim([np.nanmin(rippleFE)-5, np.nanmax(rippleFE)+8])
    ax2.set_ylabel(ylabel="power %", color="red")
    ax2.set_ylim([0, 100])
    ax.set_title("Ripple oscillation")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2)


    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(multipliers, rippleFI,  "g-", linewidth=2, marker="o", label="ripple freq (inh.)")
    ax4 = ax3.twinx()
    ax4.plot(multipliers, ripplePI,  "r-", linewidth=2, marker="|", label="ripple power (inh.)")
    ax3.set_xlim([multipliers[0], multipliers[-1]])
    ax3.set_ylabel(ylabel="freq (Hz)", color="green")
    ax3.set_ylim([np.nanmin(rippleFI)-5, np.nanmax(rippleFI)+8])
    ax4.set_ylabel(ylabel="power %", color="red")
    ax4.set_ylim([0, 100])
    ax3.set_xlabel("scale factors")
    h3, l3 = ax3.get_legend_handles_labels()
    h4, l4 = ax4.get_legend_handles_labels()
    ax3.legend(h3+h4, l3+l4)

    fig.tight_layout()
    figName = os.path.join(figFolder, "ripple.png")
    fig.savefig(figName)
    
    
def plot_summary_gamma(multipliers, gammaFE, gammaFI, gammaPE, gammaPI):
    """
    saves summary figure with ripple freq. and power
    :param multipliers: wmx multipliers
    :param gammaFE, gammaFI: exc. and inh. gamma frequency (have to be the same size as multipliers)
    :param gammaPE, gammaPI: exc. and inh. gamma power (have to be the same size as multipliers)
    """

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(multipliers, gammaFE, "b-", linewidth=2, marker="o", label="gamma freq (exc.)")
    ax2 = ax.twinx()
    ax2.plot(multipliers, gammaPE, "r-", linewidth=2, marker="|", label="gamma power (exc.)")
    ax.set_xlim([multipliers[0], multipliers[-1]])
    ax.set_ylabel(ylabel="freq (Hz)", color="blue")
    ax.set_ylim([np.nanmin(gammaFE)-5, np.nanmax(gammaFE)+8])
    ax2.set_ylabel(ylabel="power %", color="red")
    ax2.set_ylim([0, 100])
    ax.set_title("Gamma oscillation")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2)


    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot(multipliers, gammaFI,  "g-", linewidth=2, marker="o", label="gamma freq (inh.)")
    ax4 = ax3.twinx()
    ax4.plot(multipliers, gammaPI,  "r-", linewidth=2, marker="|", label="gamma power (inh.)")
    ax3.set_xlim([multipliers[0], multipliers[-1]])
    ax3.set_ylabel(ylabel="freq (Hz)", color="green")
    ax3.set_ylim([np.nanmin(gammaFI)-5, np.nanmax(gammaFI)+8])
    ax4.set_ylabel(ylabel="power %", color="red")
    ax4.set_ylim([0, 100])
    ax3.set_xlabel("scale factors")
    h3, l3 = ax3.get_legend_handles_labels()
    h4, l4 = ax4.get_legend_handles_labels()
    ax3.legend(h3+h4, l3+l4)

    fig.tight_layout()
    figName = os.path.join(figFolder, "gamma.png")
    fig.savefig(figName)
    
    
def plot_evolution(ngen, min_fit, mean_fit, std_fit, saveName_):
    """
    saves figure with the evolution of fittnes error (see: optimization/)
    :param ngen: number of generations
    :param min_fit: minimum of fitting errors (see bpop: _,_,log,_ = opt.run())
    :param mean_fit: mean of fitting errors (see bpop: _,_,log,_ = opt.run())
    :param std_fit: standard deviation of fitting errors (see bpop: _,_,log,_ = opt.run())
    :param saveName_: name of saved img
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(ngen, mean_fit, 'k-', linewidth=2, label="pop. average")
    ax.fill_between(ngen, mean_fit - std_fit, mean_fit + std_fit, color='lightgray', linewidth=1.5, label=r"pop. std")
    ax.plot(ngen, min_fit, "r-", linewidth=2, label="pop. minimum")
    ax.set_xlabel("#Generation")
    ax.set_xlim([1, max(ngen)])                                                         
    ax.set_ylabel("Fittnes score")                                                                                
    ax.legend()
    
    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)
    
    
def plot_SS_voltage(t, v, SS_voltage, current):
    """
    saves figure with SS voltage after current injection
    :param t,v: time and voltage
    :param SS_voltage: steady state voltage reached
    :param current: input current
    """
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(t, v, linewidth=1.5, label="V_m")
    ax.plot(np.linspace(800, 1000, 200), SS_voltage*np.ones(200), linewidth=1.5, label="V_SS: %.3f mV"%SS_voltage)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Memb. pot. (mV)")
    ax.set_title("Pyramidal cell with %s pA input"%current)
    ax.legend() 
    
    figName = os.path.join(figFolder, "clampPC_%s.png"%current)
    fig.savefig(figName)
  

def plot_avg_EPS(t, EPSPs, EPSP, EPSCs, EPSC, mean_weight, saveName_):
    """
    saves figure with average EPSP and EPSC
    :param t: time
    :param EPSPs, EPSP: array of EPSPs from random weights and EPSP for the average weight
    :param EPSCs, EPSC: array of EPSCs from random weights and EPSC for the average weight
    :param mean_weight: mean of all nonzero weights
    :param saveName_: name of saved img
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
    ax2.plot(t, np.mean(EPSCs, axis=0), "b-", label="mean of %i random weights"%EPSCs.shape[0])
    ax2.plot(t, EPSC, "g-", label="mean of all weights (%f nS)"%mean_weight)
    ax2.set_title("average EPSC")
    ax2.set_xlabel("Time (ms)")
    ax2.set_xlim([0, 400])
    ax2.set_ylabel("EPSC (pA)")
    ax2.legend()
    
    fig.tight_layout()    
    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)


def plot_EPS_dist(peakEPSPs, peakEPSCs, saveName_):
    """
    saves violin plots with EPSP and EPSC distributions
    :param peakEPSPs, peakEPSCs: vectors of peakEPSP & EPSC values
    :param saveName_: name of saved img
    """

    fig = plt.figure(figsize=(10, 8))
    
    ax = fig.add_subplot(2, 1, 1)
    ax.violinplot(peakEPSPs, vert=False, showmeans=True, showextrema=False, showmedians=False,
                  points=peakEPSPs.shape[0], bw_method='silverman')
    ax.set_title("%i random EPSPs (mean: %f mV)"%(peakEPSPs.shape[0], np.mean(peakEPSPs)))
    ax.set_xlabel("EPSP (mV)")
    ax.set_yticks([])
    
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.violinplot(peakEPSCs, vert=False, showmeans=True, showextrema=False, showmedians=False,
                   points=peakEPSCs.shape[0], bw_method='silverman')
    ax2.set_title("%i random EPSCs (mean: %f pA)"%(peakEPSCs.shape[0], np.mean(peakEPSCs)))
    ax2.set_xlabel("EPSC (pA)")
    ax2.set_yticks([])
    
    fig.tight_layout()
    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)


def plot_learned_EPSPs(delta_ts, dEPSPs, saveName_):
    """
    saves plot with EPSPs after learning (via STDP)
    :param delta_ts: list of $delta$t intervals between pre and post spikes (in ms)
    :param dEPSPs: dictonary with keys of delta_ts and EPSP traces as values
    :param saveName_: name of saved img
    """
    
    assert len(delta_ts) == 8, "plot works only for 2*4 delta_ts..."
    t = dEPSPs["t"]
    
    fig = plt.figure(figsize=(20, 8))
    
    for i, delta_t in enumerate(delta_ts):
        ax = fig.add_subplot(2, 4, i+1)
        ax.plot(t, dEPSPs["baseline"], "k--", linewidth=1.5)
        col = "b" if delta_t > 0 else "r"
        ax.plot(t, dEPSPs[delta_t], color=col, linewidth=2)
        ax.set_title("$\Delta t$ /post-pre/ = %s (ms)"%delta_t)
        ax.set_xlim([0, 400])
        ax.set_ylabel("EPSP (mV)")
        if i >= 4:
            ax.set_xlabel("Time (ms)")
        
    fig.tight_layout()
    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)
       
    
def plot_compare_STDP_to_orig(dEPSPchanges, dOrigData, saveName_, dOrigEFit=None, dFittedChanges=None):
    """
    saves plot based on Figure 1 i) in Mishra et al. 2016 - 10.1038/ncomms11552, extended with the in silico EPSP changes (with fixed cell model and STDP rule)
    :param dEPSPchanges: dictonary with keys of delta_ts and values of EPSP changes in percentage
    :param dOrigData: dictionary of original values (might not be straightforwad to understand)
    :param saveName_: name of saved img
    :param dOrigEFit: optional dictionary with taup, taum: time constant of weight change & Ap, Am: max amplitude of weight change (fitted to original data)
    :param dFittedChanges: same dictionary as above (fitted to simulated data)
    """
    
    fig = plt.figure(figsize=(10, 8))    
    ax = fig.add_subplot(1, 1, 1)
    
    # plot original data
    ax.plot(dOrigData["time(ms)"], dOrigData["mean(%)"], "ko", markersize=6, label="original (in vitro) data")
    ax.errorbar(dOrigData["time(ms)"], dOrigData["mean(%)"], yerr=dOrigData["sem"], fmt="none", ecolor="k")
    # plot fitted Gaussian (to original data)
    t_ = np.arange(-150, 150, 0.1)
    fit_G = dOrigData["gauss_a"] * np.exp(-((t_ - dOrigData["gauss_b"])**2) / (2*dOrigData["gauss_c"]**2))
    ax.plot(t_, fit_G, "r-", linewidth=2, label="Gaussian fit (orig. data)")
    if dOrigEFit:  # plot exponential fit (to original data)
        fit_E = np.where(t_>0, dOrigEFit["Ap"]*np.exp(-t_/dOrigEFit["taup"]), dOrigEFit["Am"]*np.exp(t_/dOrigEFit["taum"]))
        ax.plot(t_, fit_E, color="orange", linewidth=2, label="exponential fit (orig. data) taup:%.3f(ms)"%dOrigEFit["taup"])
    # plot in silico data
    ax2 = ax.twinx()
    ax2.plot(dEPSPchanges["time"], dEPSPchanges["change"], "go", markersize=6, label="simulated data")    
    if dFittedChanges:  # plot exponential fit (to simulated data)
        fit_E = np.where(t_>0, dFittedChanges["Ap"]*np.exp(-t_/dFittedChanges["taup"]), dFittedChanges["Am"]*np.exp(t_/dFittedChanges["taum"]))
        ax2.plot(t_, fit_E, "g-", linewidth=2, label="exponential fit (sim data) taup:%.3f(ms)"%dFittedChanges["taup"])
    
    ax.set_title("Compare in vitro and in silico EPSP changes")
    ax.set_xlabel("$\Delta t$ /post-pre/ (ms)")
    ax.set_xlim([-150, 150])
    ax.set_ylabel("Change in EPSP amplitude (%)", color="black")
    ax2.set_ylabel("Change in EPSP amplitude (%)", color="green")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2)
    
    figName = os.path.join(figFolder, "%s.png"%saveName_)
    fig.savefig(figName)


def plot_STDP2(dSTDPrule, dFittedChanges, mode, saveName_):
    """
    Plots the STDP rule used for learning
    exponential STDP: f(s) = A_p * exp(-s/tau_p) (if s > 0), where s=tpost_{spike}-tpre_{spike}
    :param dSTDPrule: dictionary with taup, taum: time constant of weight change & Ap, Am: max amplitude of weight change (used to specify STDP rule for learning)
    :param dFittedChanges: same dictionary as above (fitted to EPSP % changes)
    :param mode: returned by `plot_STDP_rule` (used only for ylim and naming)
    :return mode: just for saving conventions
    """

    delta_t = np.linspace(-150, 150, 1000)
    delta_w_rule = np.where(delta_t<0, dSTDPrule["Am"]*np.exp(delta_t/dSTDPrule["taum"]), None)
    delta_w_fitted = np.where(delta_t>0, dFittedChanges["Ap"]*np.exp(-delta_t/dFittedChanges["taup"]), None)

    fig = plt.figure(figsize=(10, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(delta_t, delta_w_rule, "b-", linewidth=2, label="STDP rule taup:%s(ms), Ap:%s"%(dSTDPrule["taup"], dSTDPrule["Ap"]))
    ax.axhline(0, ls='-', c='k')  
    ax.set_title("STDP")
    ax.set_xlabel("$\Delta t$ /post-pre/ (ms)")
    ax.set_xlim([-150, 150])
    ax.set_ylabel("$\Delta w$ (nS)", color="blue")
    
    ax2 = ax.twinx()
    ax2.plot(delta_t, delta_w_fitted, "g-", linewidth=2, label="fitted 'STDP' taup:%.3f(ms), Ap:%.3f"%(dFittedChanges["taup"], dFittedChanges["Ap"]))
    ax2.set_ylabel("Change in EPSP amplitude (%)", color="green")
    
    if mode == "asym":
        ax.set_ylim([-1*dSTDPrule["Ap"]*1.05, dSTDPrule["Ap"]*1.05])
        ax2.set_ylim([-1*dFittedChanges["Ap"]*1.05, dFittedChanges["Ap"]*1.05])
    elif mode == "sym":
        ax.set_ylim([-1*dSTDPrule["Ap"]*0.05, dSTDPrule["Ap"]*1.05])
        ax2.set_ylim([-1*dFittedChanges["Ap"]*0.05, dFittedChanges["Ap"]*1.05])
  
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2)

    figName = os.path.join(figFolder, "%s_%s.png"%(saveName_, mode))
    fig.savefig(figName)
    




