#!/usr/bin/python
# -*- coding: utf8 -*-
"""
Helper functions for generating hippocampal like spike trains (absolutely not the most efficient way)
authors: András Ecker, Eszter Vértes, Szabolcs Káli last update: 02.2018
"""

import numpy as np


theta = 7.0  # theta osc. freq. [Hz]
v_mice = 32.43567842  # [cm/s]
l_route = 300.0  # circumference [cm]
l_place_field = 30.0  # [cm]
t_max = 405.0  # [s]
s = 2.0  # param of phase-locking (von Misses distribution)

r = l_route / (2*np.pi)  # [cm]
phi_PF_rad = l_place_field / r  # [rad]
t_route = l_route / v_mice  # [s]
w_mice = 2*np.pi / t_route  # angular velocity


def _generate_exp_rand_number(lambda_, seed):
    """
    MATLAB's random exponential number
    :param lambda_: rate (of the Poisson process)
    :param seed: seed for random number generation
    :return: random number (same as MATLAB's exprnd(mu))
    """
    
    np.random.seed(seed)
    mu = 1.0 / lambda_
    return -mu * np.log(np.random.rand(1))[0]


def hom_poisson(lambda_, seed):
    """
    Generates Poisson process (interval times X_i = -ln(U_i)/lambda_, where lambda_ is the rate and U_i ~ Uniform(0,1))
    :param lambda_: rate of the Poisson process
    :param seed: seed for random number generation
    :return: poisson_proc: list which represent a homogenos Poisson process
    #TODO: optimize this or change to `NeuroTools.stgen()`
    """
    
    poisson_proc = [_generate_exp_rand_number(lambda_, seed)]; i = 0     
    while poisson_proc[i] < t_max:
        isi = _generate_exp_rand_number(lambda_, seed+i+1)
        poisson_proc.append(poisson_proc[-1] + isi)
        i += 1
    del poisson_proc[-1]  # delete the last element which is higher than t_max
    
    return poisson_proc


def _get_lambda(x, y, mid_PF, m):
    """
    Calculates lambda parameter of distr. (cos(): PF prefference * exp(): vonMisses distr AKA. circular normal distr. = tuning curve on circle)
    :param y: phase precision
    :param mid_PF: mean of the (current) place field
    :param m: prefered phase: f(current position within the place field)
    :return: lambda: calculated lambda parameter of the Poisson process
    """
    
    lambda1 = np.cos((2*np.pi) / (2 * phi_PF_rad) * (x - mid_PF))
    lambda2 = np.exp(s * np.cos(y - m)) / np.exp(s)
    
    return lambda1 * lambda2


def calc_lambda(t, phi_start, phase0):
    """
    Calculates the lambda parameter of the Poisson process, that represent the firing rate of CA3 pyr. cells
    (takes preferred place and phase precession into account)
    #TODO: vectorize this (instead of calling it for every t in the hom. Poisson - it could be called for the whole hom. Poisson once, but beware of PFs!)
    :param t: time (used for calculating the current position of the mice)
    :param phi_start: starting point of the place field
    :param phase0: initial phase (used for modeling phase precession)
    :return: lambda: calculated lambda parameter of the Poisson process (see: `_get_lambda()`)
    """
    
    phi_end = np.mod(phi_start + phi_PF_rad, 2*np.pi)
    
    x = np.mod(w_mice * t, 2*np.pi)  # position of the mice [rad]

    # first if-else is needed because of the circular tract...
    if phi_start < phi_end:
        if phi_start <= x and x < phi_end:  # if the mice is in the place field
            mid_PF = phi_start + phi_PF_rad/2.0
            y = phase0 + 2*np.pi * theta * t  # phase prec...
            m = - (x - phi_start) * 2*np.pi / phi_PF_rad
            lambdaP = _get_lambda(x, y, mid_PF, m)
        else:
            lambdaP = 0
    else:
        if phi_start <= x or x < phi_end:  # if the mice is in the place field
            mid_PF = np.mod(phi_start + phi_PF_rad/2.0, 2*np.pi)
            y = phase0 + 2*np.pi * theta * t  # phase prec...
            m = - (x - phi_start) * 2*np.pi / phi_PF_rad
            lambdaP = _get_lambda(x, y, mid_PF, m)
        else:
            lambdaP = 0

    return lambdaP


def inhom_poisson(lambda_, phi_start, seed, phase0=0.0):
    """
    Generates a homogenous Poisson process and converts it to inhomogenous
    via keeping only a subset of spikes based on the rate of the place cell (see: `calc_lambda()`)
    :param lambda_: rate of the Poisson process (see `calc_lambda()`)
    :param phi_start: starting point of the place field (see `calc_lambda()`)
    :param seed: seed for random number generation
    :param phase0: initial phase (see `calc_lambda()`)
    :return: inhom_poisson_proc: list which represent an inhomogenos Poisson process
    """

    poisson_proc = hom_poisson(lambda_, seed)

    inhom_poisson_proc = []
    for i, t in enumerate(poisson_proc):
        np.random.seed(seed+i+1)
        if calc_lambda(t, phi_start, phase0) >= np.random.rand(1):
            inhom_poisson_proc.append(t)

    return inhom_poisson_proc


def refractoriness(spike_trains, ref_per=5e-3):
    """
    Delete spikes which are too close to each other
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
            spikes_updated = np.delete(single_spike_train, idx).tolist()  # delete spikes which are too close
        else:
            single_spike_train_updated = single_spike_train
        spike_trains_updated.append(single_spike_train_updated)
    
    print "%i spikes deleted becuse of too short refractory period"%count
      
    return spike_trains_updated
     
