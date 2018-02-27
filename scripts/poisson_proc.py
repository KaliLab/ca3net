#!/usr/bin/python
# -*- coding: utf8 -*-
"""
helper functions for generating hippocampal like spike trains
authors: András Ecker, Eszter Vértes, Szabolcs Káli last update: 02.2018
"""

import numpy as np

theta = 7.0  # theta frequence [Hz]
vMice = 32.43567842  # velocity of the mice [cm/s]
lRoute = 300.0  # circumference [cm]
lPlaceField = 30.0  # length of the place field [cm]
tMax = 500.0  # [s]
s = 2  # param of phase-locking (von Misses distribution)

r = lRoute / (2*np.pi)  # radius [cm]
phiPFRad = lPlaceField / r  # (angle of) place field [rad]
tRoute = lRoute / vMice  # [s]
wMice = 2*np.pi / tRoute  # angular velocity


def get_lambda(x, y, mPF, m):
    """
    calculates lambda parameter of distr. (cos() - PF prefference * exp() - vonMisses distr)
    :param y: phase precision
    :param mPF: mean of the (current) place field
    :param m: prefered phase: f(current position within the place field)
    :return: lambda: calculated lambda parameter of the Poisson process
    """
    
    lambda1 = np.cos((2*np.pi) / (2 * phiPFRad) * (x - mPF))
    lambda2 = np.exp(s * np.cos(y - m)) / np.exp(s)
    
    return lambda1 * lambda2


def calc_lambda(t, phiStart, phase0):
    """
    calculates the lambda parameter of the Poisson process, that represent the firing rate of CA3 pyr. cells
    (takes preferred place and phase precession into account)
    :param t: time (used for calculating the current position of the mice)
    :param phiStart: starting point of the place field
    :param phase0: initial phase (used for modeling phase precession)
    :return: lambda: calculated lambda parameter of the Poisson process (see: get_lambda)
    """
    
    phiEnd = np.mod(phiStart + phiPFRad, 2*np.pi)
    
    x = np.mod(wMice * t, 2*np.pi)  # position of the mice [rad]

    if phiStart < phiEnd:
        if phiStart <= x and x < phiEnd:  # if the mice is in the place field
            y = phase0 + 2*np.pi * theta * t
            mPF = phiStart + phiPFRad / 2
            m = - (x - phiStart) * 2*np.pi / phiPFRad
            lambdaP = get_lambda(x, y, mPF, m)
        else:
            lambdaP = 0
    else:
        if phiStart <= x or x < phiEnd:  # if the mice is in the place field
            y = phase0 + 2*np.pi * theta * t
            mPF = phiStart + phiPFRad / 2
            m = - (x - phiStart) * 2*np.pi / phiPFRad
            lambdaP = get_lambda(x, y, mPF, m)
        else:
            lambdaP = 0

    return lambdaP
    
    
def generate_exp_rand_number(lambda_, seed):
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
    generates Poisson process (interval times X_i = -ln(U_i)/lambda_, where lambda_ is the rate and U_i ~ Uniform(0,1))
    :param lambda_: rate of the Poisson process
    :param seed: seed for random number generation
    :return: homP: list which represent a homogenos Poisson process
    """
    
    homP = []
    homP.append(generate_exp_rand_number(lambda_, seed))
    i = 0
    while homP[i] < tMax:
        exprnd = generate_exp_rand_number(lambda_, seed+i+1)
        homP.append(homP[-1] + exprnd)
        i += 1
    del homP[-1]  # delete the last element which is higher than tMax
    
    return homP


def inhom_poisson(lambda_, phiStart, seed, phase0=0.0):
    """
    makes a homogenous Poisson process and converts it to inhomogenous
    via keeping only a subset of spikes based on the rate of the place cell (see: calc_lambda)
    :param lambda_: rate of the Poisson process (see calc_lambda)
    :param phiStart: starting point of the place field (see calc_lambda)
    :param seed: seed for random number generation
    :param phase0: initial phase (see calc_lambda)
    :return: inhP: list which represent an inhomogenos Poisson process
    """

    homP = hom_poisson(lambda_, seed)

    inhP = []  # inhomogeneous Poisson process
    for i, t in enumerate(homP):
        np.random.seed(seed+i+1)
        if calc_lambda(t, phiStart, phase0) >= np.random.rand(1):
            inhP.append(t)

    return inhP


def refractoriness(spikeTrains, ref_per=5e-3):  # added only in 05.2017
    """
    delete spikes which are too close to each other
    :param spikeTrains: list of lists representing individual spike trains
    :param ref_per: refractory period (in sec)
    :return spikeTrains: same structure, but with some spikes deleted
    """
    
    spikeTrains_updated = []
    count = 0
    for spikes in spikeTrains:  # iterates over single spike trains (from indiv. neurons)
        tmp = np.diff(spikes)  # calculate ISIs
        idx = np.where(tmp < ref_per)[0] + 1 # 2ms refractoriness
        if idx.size:
            count += idx.size
            spikes_updated = np.delete(spikes, idx).tolist()  # delete spikes which are too close
        else:
            spikes_updated = spikes
        spikeTrains_updated.append(spikes_updated)
    
    print "%i spikes deleted becuse of too short refractory period"%count
      
    return spikeTrains_updated
     
