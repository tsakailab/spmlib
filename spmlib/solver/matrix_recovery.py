# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:19:19 2017

@author: tsakai
"""
#from __future__ import print_function

from math import sqrt
import numpy as np
from scipy import linalg
from scipy import sparse as sp

import spmlib.proxop as prox


def low_rank_matrix_completion(Y, R=None, l=1., rho=1., maxiter=300, rtol=1e-12, tol=1e-5, verbose=False, nesterovs_momentum=False, restart_every = np.nan, prox_rank=lambda Z,l: prox.nuclear(Z,l)):
    """
    Low-rank matrix completion by ADMM
    solves
    Yest = arg min_X f(X) + g(X)   where f(X)=0.5||(Y-X)[R]||_F^2, g(X)=l*||X||_*
    
    Parameters
    ----------
    Y : array_like, shape (`m`, `n`)
        `m` x `n` matrix to be completed, some of whose entries can be np.nan, meaning "masked (not ovserved)".
    R : array_like, optional, default None
        `m` x `nc` bool matrix of a mask of Y. False indicates "masked".
    l : scalar, optional, default 1.
        Barancing parameter lambda.
    rho : scalar, optional, default 1.
        Augmented Lagrangian parameter.
    maxiter : int, optional, default 300
        Maximum iterations.
    rtol : scalar, optional, default 1e-12
        Relative convergence torelance of `x` and `z` in ADMM.
    tol : scalar, optional, default 1e-5
        Convergence torelance for residual.
    verbose: int or bool, optional, default False
        Print the costs f(X) and g(X) every this number.
    nesterovs_momentum : bool, optional, default False
        Nesterov acceleration.
    restart_every : int, optional, default `np.nan`
        Restart the Nesterov acceleration every `restart_every` iterations. If `np.nan`, this is disabled.
    prox_rank: function, optinal, default `spmlib.proxop.nuclear`
        Proximity operator as a Python function of regularizer g for `Z`. By default, `prox_rank` is `lambda Z,l: prox.nuclear(Z,l)`, i.e., the soft thresholding of singular values of Z.

    Returns
    -------
    Yest : (`m`, `n`) ndarray, x.reshape(`m`,`n`) = np.dot(U, np.dot(s, Vh))
        Low-rank matrix estimate.
    U : ndarray, shape (`M`, `r`) with ``r = min(m, n)``
        Unitary matrix having left singular vectors as columns.
    s : ndarray, shape (`r`,)
        The singular values, sorted in non-increasing order.
    Vh : ndarray, shape (`r`, `n`)
        Unitary matrix having right singular vectors as rows.
    count : int
        Loop count at termination.
        
    Example
    -------
    >>> Yest = low_rank_matrix_completion(Y, R=R, l=1., tol=1e-4*linalg.norm(Y[R]), maxit=100, Nesterov=True, restart_every=100)[0]
    See demo_spmlib_solvers_matrix.py    
    """
    
    Y = Y.copy()
    # if a bool matrix R (observation) is given, mask Y with False of R
    if R is not None:
        Y[~R] = np.nan
    #   NaN in Y is masked
    Y = np.ma.masked_invalid(Y)
    numObsY = Y.count()
    # Y and R are modified to data and mask arrays, respectively.
    Y, R = Y.data, ~Y.mask

    #l=linalg.norm(Y)/sqrt(Y.size)
    #scale = linalg.norm(Y[R].ravel())
    #Y = Y / scale

    m, n = Y.shape
    G = sp.vstack((sp.eye(m*n, format='csr', dtype=Y.dtype)[R.ravel()], sp.eye(m*n, dtype=Y.dtype)))
    #pinvG = linalg.pinv(G.toarray())
    # Pseudo inverse of G is explicitly described as 
    pinvG = np.ones(m*n, dtype=Y.dtype)
    pinvG[R.ravel()] = 0.5
    pinvG = sp.diags(pinvG, format='csr') # sp.dia_matrix((pinvG,np.array([0])), shape=())
    pinvG = sp.hstack((0.5*sp.eye(m*n, format='csr', dtype=Y.dtype)[R.ravel()].T, pinvG))

    # initialize
    x = np.zeros(m*n, dtype=Y.dtype)
    z = np.concatenate( (Y[R].ravel(),np.zeros(m*n, dtype=Y.dtype).ravel()) )
    u = np.zeros_like(z) #np.zeros(z.shape, dtype=z.dtype)
    count = 0
    res_old = 0.
    dres = np.inf

    t = 1. #
    while count < maxiter:
        count += 1

        if np.fmod(count, restart_every) == 0: #
            t = 1.
        if nesterovs_momentum:
            told = t
            t = 0.5 * (1. + sqrt(1. + 4. * t * t))

        # update x
        dx = x.copy()
        x = pinvG.dot(z - u)
        dx = x - dx
 
        Gx = G.dot(x)
        v = Gx + u

        # update z
        dz = z.copy() #
        z[:numObsY] = prox.squ_l2(v[:numObsY], 1/rho, c=Y[R].ravel()) # (rho*v[:numObsY] + Y[R].ravel())/(rho+1.)
        # z[numObsY:] = soft_svd(v[numObsY:].reshape(m,n,order='F'), w/rho)[0].ravel()
        L, U, s, Vh = prox_rank(v[numObsY:].reshape(m,n), l/rho)
        z[numObsY:] = L.ravel()
        dz = z - dz #

        # update u
        du = u.copy() #
        u = u + Gx - z
        du = u - du #

        # Nesterov acceleration
        if nesterovs_momentum:
            z = z + ((told - 1.) / t) * dz
            u = u + ((told - 1.) / t) * du

        res = linalg.norm(x[R.ravel()] - Y[R].ravel())**2
        tr = np.sum(s)
        if verbose:
            if np.fmod(count,verbose) == 0:
                print('%2d: 0.5*||R.*(Y-Yest)||_F^2 + l * ||Yest||_* = %.2e + %.2e = %.2e' % (count, 0.5*res, l*tr, 0.5*res+l*tr))

        dres = np.abs(res - res_old)
        res_old = res

        # check convergence
        if dres < tol or (linalg.norm(dx) < rtol * linalg.norm(x) and linalg.norm(dz) < rtol * linalg.norm(z)):
            break

    return x.reshape(m,n), U, s, Vh, count



# alias
import collections
def LowRankMatrixCompletion(Y, l=1, rho=1., maxit=300, tol=1e-6, verbose=False):
    result = low_rank_matrix_completion(Y.copy(), l=l, maxiter=maxit, tol=tol, verbose=verbose)
    ret = collections.namedtuple('ret', 'Yest, U, s, V, iter')
    return ret(Yest=result[0], U=result[1], s=result[2], V=result[3], iter=result[4])



