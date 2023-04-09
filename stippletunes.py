import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

def make_stipple_tune(X, indx, y, n_angles, n_perwin, theta_win, n_threads=1):
    """
    Use the Viterbi algorithm to optimally rotate a path and trace through 
    a given stipple pattern

    Parameters
    ----------
    X: ndarray(N, 2)
        Unordered points in a stipple pattern
    y: ndarray(M) or ndarray(M, 2)
        A 1 or 2 channel audio stream
    n_angles: int
        Number of discrete angles to consider
    n_perwin: int
        Number of points per rotation interval
    theta_win: int
        Number of angles to consider
    n_threads: int
        Number of threads to use

    Returns
    -------
    ndarray(M, 2)
        Stippled traced 2-channel audio

    """
    ## Step 1: Initialize variables
    d_theta = 2*np.pi/(n_angles+1)
    Y = np.array(y)
    if len(Y.shape) == 1:
        Y = np.array([Y, Y]).T
    Y = Y[:, 0] + 1j*Y[:, 1]

    M = int(np.ceil(len(Y)/n_perwin)) # Total number of windows
        
    theta_angles = {}
    for j in range(n_angles):
        for dk in range(-theta_win, theta_win+1):
            k = (j+dk)%n_angles # Last state
            theta_angles[(k, j)] = np.exp(1j*np.linspace(d_theta*(j+dk), d_theta*j, n_perwin))
        
    ## Step 2: Fill in dynamic programming matrix and state matrix
    C = np.zeros((n_angles, M))
    I = np.zeros((n_angles, M), dtype=int)
    Yj = np.zeros(n_perwin*theta_win, dtype=Y.dtype)
    for i in range(1, M):
        print(i, end='.')
        # Try each pair of angles
        for j in range(n_angles): # Current state
            ks = np.inf*np.ones(n_angles)
            ks_idxs = np.zeros(theta_win, dtype=int)
            for dk in range(1, theta_win+1):
                k = (j+dk)%n_angles
                ks_idxs[dk-1] = k
                Yj[(dk-1)*n_perwin:dk*n_perwin] = Y[(i-1)*n_perwin:i*n_perwin]*theta_angles[(k, j)]
            #dd, _ = tree.query(np.array([np.real(Yj), np.imag(Yj)]).T, workers=n_threads)
            #dd = np.reshape(dd, (theta_win, n_perwin))
            #transition_costs = np.sum(dd, 1)
            ks[ks_idxs] = transition_costs + C[ks_idxs, i-1]
            I[j, i] = np.argmin(ks)
            C[j, i] = ks[I[j, i]]
    
    ## Step 3: Do backtracing
    path = np.zeros(M, dtype=int)
    idx = np.argmin(C[:, -1])
    i = M-1
    while i >= 0:
        path[i] = idx
        idx = I[idx, i]
        i -= 1
    
    ## Step 4: Fill in final 2-channel audio
    Z = 1*Y
    Z_res = np.zeros((len(Z), 2))
    errs = np.array([])
    for i in range(1, len(path)):
        k = path[i-1]
        j = path[i]
        Z[(i-1)*n_perwin:i*n_perwin] *= theta_angles[(k, j)]
        Zi = Z[(i-1)*n_perwin:i*n_perwin]
        errsi, idxs = tree.query(np.array([np.real(Zi), np.imag(Zi)]).T)
        errs = np.concatenate((errs, errsi))
        Z_res[(i-1)*n_perwin:i*n_perwin, :] = X[idxs[0:len(Zi)], :]
    
    return Z_res, path, errs