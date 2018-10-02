# -*- coding: utf8 -*-
"""
Functions used for generating hippocampal like spike trains
Setup: many repetitions on a circular track or linear [0, 2pi] track
authors: András Ecker, Eszter Vértes, Szabolcs Káli last update: 10.2018
"""

import numpy as np


f_theta = 7.0  # theta osc. freq. [Hz]
v_mice = 32.43567842  # [cm/s]
l_route = 300.0  # circumference [cm]
l_place_field = 30.0  # [cm]
r = l_route / (2*np.pi)  # [cm]
phi_PF_rad = l_place_field / r  # [rad]
t_route = l_route / v_mice  # [s]
w_mice = 2*np.pi / t_route  # angular velocity

# hard coded s and std of Gaussians: 1/10th of the total route is PF (used with the parameters above and 10% rate def. from Dragoi and Buzsáki 2006)
#TODO: make these parameters calculated based on PF parameters (otherwise DON'T touch this!)
s = 47.0  # phase-locking (param of circular Gaussian)
std = 0.146  # std (param of Gaussian, defined in [0,2*np.pi])


def _generate_exp_rand_numbers(lambda_, n_rnds, seed):
    """
    MATLAB's random exponential number
    :param lambda_: normalization (will be the rate of Poisson proc - see `hom_poisson()`)
    :param n_rnds: number of random numbers to gerenerate
    :param seed: seed for random number generation
    :return: exponential random numbers
    """
    
    np.random.seed(seed)  
    return -1.0 / lambda_ * np.log(np.random.rand(n_rnds))


def hom_poisson(lambda_, n_rnds, t_max, seed):
    """
    Generates Poisson process (interval times X_i = -ln(U_i)/lambda_, where lambda_ is the rate and U_i ~ Uniform(0,1))
    :param lambda_: rate of the Poisson process
    :param n_rnds: see `_generate_exp_rand_numbers()`
    :param t_max: length of the generate Poisson process
    :param seed: seed for random number generation (see `_generate_exp_rand_numbers()`)
    :return: poisson_proc: np.array which represent a homogenos Poisson process
    """
    
    rnd_isis = _generate_exp_rand_numbers(lambda_, n_rnds, seed)    
    poisson_proc = np.cumsum(rnd_isis)
    
    return poisson_proc[np.where(poisson_proc <= t_max)]
    #TODO: this doesn't make sure that the Poisson proc is t_max long... but generates longer (based on n_rnds) and deletes the end


def get_tuning_curve_circular(spatial_points, phi_start):
    """
    Calculates (not estimates) tuning curve (on a circle -> circular Gaussian function)
    :param spatial_points: spatial points along the circle (in rad)
    :param phi_start: starting point of the place field (in rad)
    :return: tau: tuning curve of the place cell
    """

    mid_PF = np.mod(phi_start + phi_PF_rad/2.0, 2*np.pi)    
    tau = 1.0/np.exp(s) * np.exp(s*np.cos(spatial_points - mid_PF))  # circular Gaussian
        
    return tau


def get_tuning_curve_linear(spatial_points, phi_start):
    """
    Calculates (not estimates) tuning curve (Gaussian function)
    :param spatial_points: spatial points along the track
    :param phi_start: starting point of the place field
    :return: tau: tuning curve of the place cell
    """

    mid_PF = phi_start + phi_PF_rad/2.0
    tau = np.exp(-np.power(spatial_points-mid_PF, 2)/(2*std**2))
        
    return tau
    
    
def evaluate_lambda_t(t, phi_start, linear, phase0):
    """
    Evaluates firing rate(t, x) = tuning_curve(x) * theta_modulation(t, x) at given time points
    :param t: sample time points
    :param phi_start: starting point of the place field (in rad)
    :param linear: flag for circular vs. linear track -> slightly diff tuning curves
    :param phase0: init. phase (used to calc. phase precession)
    :return: lambda_t sampled at the given time points
    """
    
    x = np.mod(w_mice * t, 2*np.pi)  # positions of the mice [rad]
    
    if not linear:
        tau_x = get_tuning_curve_circular(x, phi_start)
    else:
        tau_x = get_tuning_curve_linear(x, phi_start)
            
    # theta modulation of firing rate + phase precession
    phase = phase0 + 2*np.pi * f_theta * t
    phase_shift = -np.pi / phi_PF_rad * (x - phi_start)
    theta_mod = np.cos(phase - phase_shift)
        
    lambda_t = tau_x * theta_mod
    lambda_t[np.where(lambda_t < 0.0)] = 0.0
    
    return lambda_t
    

def inhom_poisson(lambda_, t_max, phi_start, linear, seed, phase0=0.0):
    """
    Generates a homogenous Poisson process and converts it to inhomogenous
    via keeping only a subset of spikes based on the (time and space dependent) rate of the place cell (see `evaluate_lambda_t()`)
    :param lambda_: rate of the hom. Poisson process (see `hom_poisson()`)
    :param t_max: length of the generate Poisson process
    :param phi_start: starting point of the place field (see `evaluate_lambda_t()`)
    :param linear: flag for circular vs. linear track (see `evaluate_lambda_t()`)
    :param seed: seed for random number generation
    :param phase0: initial phase (see `evaluate_lambda_t()`)
    :return: inhom_poisson_proc: inhomogenos Poisson process representing the spike train of a place cell
    """

    poisson_proc = hom_poisson(lambda_, 10000, t_max, seed)  # hard coded 10000 works with 20Hz rate and 405sec spike train
    
    # keep only a subset of spikes
    lambda_t = evaluate_lambda_t(poisson_proc, phi_start, linear, phase0)
    np.random.seed(seed)
    inhom_poisson_proc = poisson_proc[np.where(lambda_t >= np.random.rand(poisson_proc.shape[0]))]
    
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
            single_spike_train_updated = np.delete(single_spike_train, idx).tolist()  # delete spikes which are too close
        else:
            single_spike_train_updated = single_spike_train
        spike_trains_updated.append(single_spike_train_updated)
    
    print "%i spikes deleted becuse of too short refractory period"%count
      
    return spike_trains_updated
     
