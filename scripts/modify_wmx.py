# -*- coding: utf8 -*-
"""
Weight matrix modifications
author: András Ecker last update: 08.2018
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from plots import *
from stdp import save_wmx
from spw_network import load_wmx


base_path = os.path.sep.join(os.path.abspath("__file__").split(os.path.sep)[:-2])
nPCs = 8000


def shuffle(wmx_orig):
    """
    Randomly shuffles the weight matrix (keeps weight distribution, but no spatial pattern)
    :param wmx_orig: original weight matrix
    :return: wmx_modified: modified weight matrix
    """
    
    np.random.seed(12345)
    
    wmx_modified = wmx_orig  # stupid numpy...
    np.random.shuffle(wmx_modified)  # shuffle's only rows (keeps output weights)
    np.random.shuffle(wmx_modified.T)  # transpose and shuffle rows -> shuffle columns

    return wmx_modified
    
   
def binarize(wmx_orig, ratio=0.05):
    """
    Makes the matrix binary by averaging the highest x and the lowest 1-x part of the nonzero weights
    :param wmx_orig: original weight matrix
    :param ratio: highest x part of the matrix
    :return: wmx_modified: modified weight matrix
    """

    # sort non-zero values, get avg of min and max weights
    wmx_orig_nonzero = wmx_orig[np.nonzero(wmx_orig)].tolist()
    n_nonzero = len(wmx_orig_nonzero)
    
    th_idx = int(n_nonzero * ratio)
    wmx_orig_nonzero.sort(reverse=True)
    max_ = np.mean(wmx_orig_nonzero[:th_idx])
    min_ = np.mean(wmx_orig_nonzero[th_idx:])
    print min_, max_

    # create binary weight matrix
    wmx_modified = np.zeros((nPCs, nPCs))
    
    for i in range(0, nPCs):
        row_orig = wmx_orig[i, :]
        row_modified = np.zeros_like(row_orig)
        
        idx_nonzero = np.nonzero(row_orig)[0]  
        row_modified[idx_nonzero] = min_
        
        n_row_nonzero = idx_nonzero.shape[0]
        th_row_idx = int(n_row_nonzero * ratio)
             
        max_idx = np.argpartition(row_orig, -th_row_idx)[-th_row_idx:]  # numpy magic to get out the idx of the n highest values
        row_modified[max_idx] = max_
        
        wmx_modified[i, :] = row_modified

    return wmx_modified
    
    
def shuffle_blocks(wmx_orig, pop_size=800):
    """
    Shuffles pop_size*pop_size blocks within the martrix   
    :param wmx_orig: original weight matrix
    :param pop_size: size of the blocks kept together
    :return: wmx_modified: modified weight matrix
    """
    
    assert nPCs % pop_size == 0
    np.random.seed(12345)
    
    # get blocks
    n_pops = nPCs / pop_size
    blocks = {}
    for i in range(n_pops):
        for j in range(n_pops):
            blocks[i, j] = wmx_orig[i*pop_size:(i+1)*pop_size, j*pop_size:(j+1)*pop_size]
    
    # generate shuffling idx      
    x = np.linspace(0, n_pops-1, n_pops)
    y = np.linspace(0, n_pops-1, n_pops)
    np.random.shuffle(x)
    np.random.shuffle(y)

    # create block shuffled weight matrix
    wmx_modified = np.zeros((nPCs, nPCs))
    for i, id_i in enumerate(x):
        for j, id_j in enumerate(y):
            wmx_modified[i*pop_size:(i+1)*pop_size, j*pop_size:(j+1)*pop_size] = blocks[id_i, id_j]

    return wmx_modified


if __name__ == "__main__":

    try:
        STDP_mode = sys.argv[1]       
    except:
        STDP_mode = "asym"
    assert STDP_mode in ["sym", "asym"]
    
    place_cell_ratio = 0.5
    linear = True
    f_in = "wmx_%s_%.1f_linear.pkl"%(STDP_mode, place_cell_ratio) if linear else "wmx_%s_%.1f.pkl"%(STDP_mode, place_cell_ratio)
    
    pklf_name = os.path.join(base_path, "files", f_in)
    wmx_orig = load_wmx(pklf_name)
    
    #wmx_modified = shuffle(wmx_orig); f_out = "%s_shuffled_linear.pkl"%f_in[:-11] if linear else "%s_shuffled.pkl"%f_in[:-4]
    #wmx_modified = binarize(wmx_orig); f_out = "%s_binary_linear.pkl"%f_in[:-11] if linear else "%s_binary.pkl"%f_in[:-4]
    wmx_modified = shuffle_blocks(wmx_orig); f_out = "%s_block_shuffled_linear.pkl"%f_in[:-11] if linear else "%s_block_shuffled.pkl"%f_in[:-4]
    
    assert np.shape(wmx_modified) == (nPCs, nPCs), "Output shape is not %i*%i"%(nPCs, nPCs)
    assert (wmx_modified >= 0.0).all(), "Negative weights in the modified matrix!"
    
    
    pklf_name = os.path.join(base_path, "files", f_out)
    save_wmx(wmx_modified, pklf_name)
    
    plot_wmx(wmx_modified, save_name=f_out[:-4])
    plot_wmx_avg(wmx_modified, n_pops=100, save_name="%s_avg"%f_out[:-4])
    plot_w_distr(wmx_modified, save_name="%s_distr"%f_out[:-4])
    selection = np.array([500, 2400, 4000, 5500, 7015])
    plot_weights(save_selected_w(wmx_modified, selection), save_name="%s_sel_weights"%f_out[:-4])
    plt.show()
    
    
 
