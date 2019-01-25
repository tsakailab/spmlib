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
import spmlib.thresholding as th



#%% alias
import collections
def LowRankMatrixCompletion(Y, l=1, rho=1., maxit=300, tol=1e-6, verbose=False):
    result = low_rank_matrix_completion(Y.copy(), l=l, maxiter=maxit, tolr=tol, verbose=verbose)
    ret = collections.namedtuple('ret', 'Yest, U, s, V, iter')
    return ret(Yest=result[0], U=result[1], s=result[2], V=result[3], iter=result[4])



def low_rank_matrix_completion(Y, R=None, l=1., rtol=1e-12, tol=None, rho=1., maxiter=300, verbose=False, nesterovs_momentum=False, restart_every = np.nan, prox_rank=lambda Q,l: prox.nuclear(Q,l)):
    """
    Low-rank matrix completion by ADMM
    solves
    Yest = arg min_X f(X) + g(X)   where f(X)=0.5||(Y-X)[R]||_F^2, g(X)=l*||X||_*
    
    Parameters
    ----------
    Y : array_like, shape (`m`, `n`)
        `m` x `n` matrix to be completed, some of whose entries can be np.nan, meaning "masked (not ovserved)".
    R : array_like, optional, default None
        `m` x `nc` bool matrix of a mask of Y, whose entries True and False indicates "observed" and "masked", respectively.
    l : scalar, optional, default 1.
        Barancing parameter lambda.
    rtol : scalar, optional, default 1e-12
        Relative convergence tolerance of `u` and `z` in ADMM, i.e., the primal and dual residuals.
    tol : scalar, optional, default None
        Convergence tolerance of residual.
    rho : scalar, optional, default 1.
        Augmented Lagrangian parameter.
    maxiter : int, optional, default 300
        Maximum iterations.
    verbose: int or bool, optional, default False
        Print the costs f(X) and g(X) every this number.
    nesterovs_momentum : bool, optional, default False
        Nesterov acceleration.
    restart_every : int, optional, default `np.nan`
        Restart the Nesterov acceleration every `restart_every` iterations. If `np.nan`, this is disabled.
    prox_rank: function, optinal, default `spmlib.proxop.nuclear`
        Proximity operator as a Python function for regularizer g, or the matrix rank. By default, `prox_rank` is `lambda Q,l: prox.nuclear(Q,l)`, i.e., the soft thresholding of singular values of Q.

    Returns
    -------
    Yest : (`m`, `n`) ndarray, x.reshape(`m`,`n`) = np.dot(U, np.dot(sv, Vh))
        Low-rank matrix estimate.
    U : ndarray, shape (`M`, `r`) with ``r = min(m, n)``
        Unitary matrix having left singular vectors as columns.
    sv : ndarray, shape (`r`,)
        Singular values, sorted in non-increasing order.
    Vh : ndarray, shape (`r`, `n`)
        Unitary matrix having right singular vectors as rows.
    count : int
        Loop count at termination.
        
    Example
    -------
    >>> Yest = low_rank_matrix_completion_F(Y, R=R, l=1., tol=1e-4*linalg.norm(Y[R]), maxiter=100, nesterovs_momentum=True, verbose=10)[0]
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

    #if tol is None:
    #    tol = rtol * linalg.norm(Y[R].ravel())
    
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
    z = np.zeros(numObsY+m*n, dtype=Y.dtype)
    u = np.zeros_like(z) #np.zeros(z.shape, dtype=z.dtype)
    count = 0

    t = 1. #
    while count < maxiter:
        
        count += 1

        if np.fmod(count, restart_every) == 0: #
            t = 1.
        if nesterovs_momentum:
            told = t
            t = 0.5 * (1. + sqrt(1. + 4. * t * t))

        # update x
        #dx = x.copy()
        x = pinvG.dot(z - u)
        #dx = x - dx
 
        Gx = G.dot(x)
        q = Gx + u

        # update z
        dz = z.copy() #
        z[:numObsY] = prox.squ_l2(q[:numObsY], 1/rho, c=Y[R].ravel()) # (rho*q[:numObsY] + Y[R].ravel())/(rho+1.)
        # z[numObsY:] = soft_svd(q[numObsY:].reshape(m,n,order='F'), w/rho)[0].ravel()
        L, U, sv, Vh = prox_rank(q[numObsY:].reshape(m,n), l/rho)
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

        res = linalg.norm(x[R.ravel()] - Y[R].ravel())
        tr = np.sum(sv)
        if verbose:
            if np.fmod(count,verbose) == 0:
                print('%2d: 0.5*||R.*(Y-Yest)||_F^2 + l*||Yest||_* = %.2e(%.1f*tol) + %.2e = %.2e' % (count, 0.5*res*res, res/tol, l*tr, 0.5*res*res+l*tr))

        # check convergence of primal and dual residuals
        if linalg.norm(du) < rtol * linalg.norm(u) and linalg.norm(dz) < rtol * linalg.norm(z):
            break
        if tol is not None:
            if res < tol:
                break

    return x.reshape(m,n), U, sv, Vh, count



#%% This is experimental.
def low_rank_matrix_completion_ind(Y, R=None, rtol=1e-12, tol=None, rho=1., maxiter=300, verbose=False, nesterovs_momentum=False, restart_every = np.nan, prox_rank=lambda Q,l: prox.nuclear(Q,l)):
    """
    Low-rank matrix completion
    solves, by ADMM, 
    (vecL, [z_LR; z_L]) = arg min_(x,z)  g_LR(z_LR) + g_L(z_L)
                                s.t. [z_LR; z_L] = [R; I] * vecL.
    Here, by default,
    g_LR(z_LR) = indicator function, i.e., zero if ||Y[R] - mat(z_LR)||_F <= tol, infinity otherwise,
    g_L(z_L) = ||mat(z_L)||_*, i.e., nuclear norm of mat(z_L).
    
    Parameters
    ----------
    Y : array_like, shape (`m`, `n`)
        `m` x `n` matrix to be completed, some of whose entries can be np.nan, meaning "masked (not ovserved)".
    R : array_like, optional, default None
        `m` x `nc` bool matrix of a mask of Y, whose entries True and False indicates "observed" and "masked", respectively.
    rtol : scalar, optional, default 1e-12
        Relative convergence tolerance of `u` and `z` in ADMM, i.e., the primal and dual residuals.
    tol : scalar, optional, default None
        Tolerance for residual. If None, tol = rtol * (sum of squares of Y[R]).
    rho : scalar, optional, default 1.
        Augmented Lagrangian parameter.
    maxiter : int, optional, default 300
        Maximum iterations.
    verbose: int or bool, optional, default False
        Print the residual and nuclear norm every this number.
    nesterovs_momentum : bool, optional, default False
        Nesterov acceleration.
    restart_every : int, optional, default `np.nan`
        Restart the Nesterov acceleration every `restart_every` iterations. If `np.nan`, this is disabled.
    prox_rank: function, optinal, default `spmlib.proxop.nuclear`
        Proximity operator as a Python function for g_L(vecQ), or the rank of `Q`. By default, `prox_rank` is `lambda Q,l: prox.nuclear(Q,l)`, i.e., the soft thresholding of singular values of Q.

    Returns
    -------
    Yest : (`m`, `n`) ndarray, x.reshape(`m`,`n`) = np.dot(U, np.dot(sv, Vh))
        Low-rank matrix estimate.
    U : ndarray, shape (`M`, `r`) with ``r = min(m, n)``
        Unitary matrix having left singular vectors as columns.
    sv : ndarray, shape (`r`,)
        Singular values, sorted in non-increasing order.
    Vh : ndarray, shape (`r`, `n`)
        Unitary matrix having right singular vectors as rows.
    count : int
        Loop count at termination.
        
    Example
    -------
    >>> Yest = low_rank_matrix_completion_ind(Y, R=R, tol=1e-2*linalg.norm(Y[R]), maxiter=100, nesterovs_momentum=True, restart_every=4, verbose=10, rtol=1e-3)[0]
    See demo_spmlib_solvers_matrix.py    
    """
    Y = Y.copy()
    m, n = Y.shape
    # if a bool matrix R (observation) is given, mask Y with False of R
    if R is not None:
#        z = Y.copy().ravel()
#        if not np.isfinite(Y[~R]).all():
#            z = np.zeros(m*n, dtype=Y.dtype)
        Y[~R] = np.nan
    #   NaN in Y is masked
    Y = np.ma.masked_invalid(Y)
    numObsY = Y.count()
    # Y and R are modified to data and mask arrays, respectively.
    Y, R = Y.data, ~Y.mask

    #l=linalg.norm(Y)/sqrt(Y.size)
    #scale = linalg.norm(Y[R].ravel())
    #Y = Y / scale

    if tol is None:
        tol = rtol * linalg.norm(Y[R].ravel())
    
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
#    z = np.concatenate( (Y[R].ravel(), z) )
    u = np.zeros_like(z) #np.zeros(z.shape, dtype=z.dtype)
    count = 0
    #w = 1e-4

    t = 1. #
    while count < maxiter:
        count += 1

        if np.fmod(count, restart_every) == 0: #
            t = 1.
        if nesterovs_momentum:
            told = t
            t = 0.5 * (1. + sqrt(1. + 4. * t * t))

        # update x
        #dx = x.copy()
        x = pinvG.dot(z - u)
        #dx = x - dx
 
        Gx = G.dot(x)
        q = Gx + u

        # update z
        dz = z.copy() #
        #z[:numObsY] = prox.ave_squ_ind_l2ball(q[:numObsY], 1., tol, w, c=Y[R].ravel())
        #w = w * 1.5;
        z[:numObsY] = prox.ind_l2ball(q[:numObsY], tol, c=Y[R].ravel())
        #
        # z[numObsY:] = soft_svd(q[numObsY:].reshape(m,n,order='F'), w/rho)[0].ravel()
        L, U, sv, Vh = prox_rank(q[numObsY:].reshape(m,n), 1./rho)
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

        if verbose:
            if np.fmod(count,verbose) == 0:
                res = linalg.norm(x[R.ravel()] - Y[R].ravel())
                print('%2d: ||R.*(Y-Yest)||_F = %.2e(%.1f*tol), ||Yest||_* = %.2e' % (count, res, res/tol, np.sum(sv)))

        # check convergence of primal and dual residuals
        if linalg.norm(du) < rtol * linalg.norm(u) and linalg.norm(dz) < rtol * linalg.norm(z):
            break

    return x.reshape(m,n), U, sv, Vh, count



def stable_principal_component_pursuit(C, R=None, tol=None, ls=None, rtol=1e-12, rho=1., maxiter=300, verbose=False, nesterovs_momentum=False, restart_every = np.nan, 
                                       prox_LS=lambda q,r,c: prox.ind_l2ball(q,r,c),
                                       prox_L=lambda Q,l: prox.nuclear(Q,l), 
                                       prox_S=lambda q,l: prox.l1(q,l)):
    """
    Low-rank and sparse matrix approximation (a.k.a. Stable principal component pursuit; SPCP, three-term decomposition; TTD)
    solves the following minimization problem by ADMM to find low-rank and sparse matrices, L and S, that approximate C as L + S.
    ([vecL; vecS], [z_LS; z_L; z_S]) = arg min_(x,z) g_LS(z_LS) + g_L(z_L) + g_S(z_S)
                                s.t. [z_LS; z_L; z_S] = [M M; I O; O I] * [vecL; vecS].
    Here, by default,
    g_LS(z_LS) = indicator function, i.e., zero if ||C - mat(z_LS)||_F <= tol, infinity otherwise,
    g_L(z_L) = ||mat(z_L)||_*, i.e., nuclear norm of mat(z_L),
    g_S(z_S) = ||ls.*z_S||_1, i.e., l1 norm of mat(z_S) with the weight ls,
    M  : linear operator that extracts valid entries from vecC=C.ravel(), i.e., M(v)=v[R.ravel()].
    
    Parameters
    ----------
    C : array_like, shape (`m`, `n`)
        `m` x `n` matrix to be completed and separated into L and S, some of whose entries can be np.nan, meaning "masked (not ovserved)".
    R : array_like, optional, default None
        `m` x `nc` bool matrix of a mask of C. False indicates "masked".
    tol : scalar, optional, default None
        Tolerance for residual. If None, tol = rtol * ||C||_F
    ls : scalar, optional, default None
        Weight of sparse regularizer.  `ls` can be a matrix of weights for each entries of `Z_s.reshape(m,n)`.
        If None, ls = 1./sqrt(max(C.shape)).
    rtol : scalar, optional, default 1e-12
        Relative convergence tolerance of `u` and `z` in ADMM, i.e., the primal and dual residuals.
    rho : scalar, optional, default 1.
        Augmented Lagrangian parameter.
    maxiter : int, optional, default 300
        Maximum iterations.
    verbose: int or bool, optional, default False
        Print the costs every this number.
    nesterovs_momentum : bool, optional, default False
        Nesterov acceleration.
    restart_every : int, optional, default `np.nan`
        Restart the Nesterov acceleration every `restart_every` iterations. If `np.nan`, this is disabled.
    prox_LS : function, optional, default `spmlib.proxop.ind_l2ball`
        Proximity operator as a Python function for the regularizer g_LS of `z_LS` = `vecL`+`vecS`. By default, `prox_LS` is `lambda q,r,c:spmlib.proxop.ind_l2ball(q,r,c)`, i.e., the prox. of the indicator function of l2-ball with radius 'r' and center 'c'.
    prox_L : function, optional, default `spmlib.proxop.squ_l2_from_subspace`
        Proximity operator as a Python function for the regularizer g_L of `z_L` = `vecL`. By default, `prox_L` is `lambda Q,l:spmlib.proxop.nuclear(Q,1)`, i.e., the prox. of the nuclear norm * l*||mat z_L||_*.
    prox_S : function, optional, default `spmlib.proxop.l1`
        Proximity operator as a Python function for the regularizer g_S of `z_S` = `vecS`. By default, `prox_S` is `lambda q,l:spmlib.proxop.l1(q,l)`, i.e., the soft thresholding operator as the prox. of l1 norm ||l.*z_S||_1.

    Returns
    -------
    L : (`m`, `n`) ndarray, x[:m].reshape(`m`,`n`) = np.dot(U, np.dot(sv, Vh))
        Low-rank matrix estimate.
    S : (`m`, `n`) ndarray, x[m:].reshape(`m`,`n`)
        Sparse matrix estimate.
    U : ndarray, shape (`M`, `r`) with ``r = min(m, n)``
        Unitary matrix having left singular vectors as columns.
    sv : ndarray, shape (`r`,)
        Singular values, sorted in non-increasing order.
    Vh : ndarray, shape (`r`, `n`)
        Unitary matrix having right singular vectors as rows.
    count : int
        Loop count at termination.
        
    Example
    -------
    >>> stable_principal_component_pursuit(C, ls=1., tol=1e-4*linalg.norm(C), rtol=1e-4, maxiter=300, verbose=10)[0]
    See demo_spmlib_solvers_matrix.py
    """
    C = C.copy()
    # if a bool matrix R (observation) is given, mask C with False of R
    if R is not None:
        C[~R] = np.nan
    #   NaN in C is masked
    C = np.ma.masked_invalid(C)
    numObsC = C.count()
    # C and R are modified to data and mask arrays, respectively.
    C, R = C.data, ~C.mask
    
    m, n = C.shape
    mn = m*n
    if ls is None:
        ls = 1./sqrt(max(C.shape))
    #ls = np.array(ls).ravel()
    if tol is None:
        tol = rtol * linalg.norm(C)

    def G(x, q):
        q[:numObsC]     = (x[:mn] + x[mn:])[R.ravel()]
        q[numObsC:-mn]  = x[:mn]
        q[-mn:]         = x[mn:]

    def GT(q, p):
        p[:mn] = 0.
        p[:mn][R.ravel()] = q[:numObsC]
        p[mn:] = p[:mn] + q[-mn:]
        p[:mn] += q[numObsC:-mn]

    def p2x(p, x):
        rp = np.zeros(mn, dtype=C.dtype)
        rp[R.ravel()] = (1./3.) * (p[:mn] + p[mn:])[R.ravel()]
        x[:mn] = p[:mn] - rp
        x[mn:] = p[mn:] - rp

    # initialize
    x = np.zeros(2*mn, dtype=C.dtype)
    z = np.zeros(numObsC+2*mn, dtype=C.dtype)
    u = np.zeros_like(z)
    count = 0

    p = np.zeros_like(x)
    t = 1. #
    while count < maxiter:
        count += 1

        if np.fmod(count, restart_every) == 0: #
            t = 1.
        if nesterovs_momentum:
            told = t
            t = 0.5 * (1. + sqrt(1. + 4. * t * t))

        # update x
        #dx = x.copy()
        #p = G.T.dot(z - u)
        q = z - u
        GT(q, p)
        p2x(p, x)
        #dx = x - dx

        G(x, q)
        q += u
 
        # update z
        dz = z.copy() #
        z[:numObsC]  = prox_LS(q[:numObsC], tol, C[R].ravel())
        L, U, sv, Vh = prox_L(q[numObsC:-mn].reshape(m,n), 1./rho)
        z[numObsC:-mn] = L.ravel()
        z[-mn:]  = prox_S(q[-mn:], ls/rho)
        dz = z - dz #

        if nesterovs_momentum:
           # update x again (heuristic)
           q = z - u
           GT(q, p)
           p2x(p, x)

        # update u
        # u = u + G(x) - z
        du = u.copy()
        G(x, q)
        u = u + q - z
        du = u - du

        # Nesterov acceleration
        if nesterovs_momentum:
#            z = z + ((told - 1.) / t) * dz
            u = u + ((told - 1.) / t) * du    # update u only (heuristic)

        if verbose:
            if np.fmod(count,verbose) == 0:
                print('%2d: ||R*(C-(L+S))||_F=%.2e, ||L||_*=%.2e (rnk=%d), ||S||_1=%.2e (nnz=%2.1f%%)' 
                      % (count, linalg.norm((x[:mn]+x[mn:])[R.ravel()]-C[R].ravel()), 
                         np.sum(sv), np.count_nonzero(sv), 
                         np.sum(np.abs(x[mn:])), 100.*np.count_nonzero(z[-mn:])/mn))

        # check convergence of primal and dual residuals
        if linalg.norm(du) < rtol * linalg.norm(u) and linalg.norm(dz) < rtol * linalg.norm(z):
            break

    return x[:mn].reshape(m,n), x[mn:].reshape(m,n), U, sv, Vh, count



def _stable_principal_component_pursuit(D, tol=None, ls=None, rtol=1e-12, rho=1., maxiter=300, verbose=False, nesterovs_momentum=False, restart_every = np.nan, 
                                       prox_LS=lambda q,r,c: prox.ind_l2ball(q,r,c),
                                       prox_L=lambda Q,l: prox.nuclear(Q,l), 
                                       prox_S=lambda q,l: prox.l1(q,l)):
    """
    Low-rank and sparse matrix approximation (a.k.a. Stable principal component pursuit; SPCP, three-term decomposition; TTD)
    solves the following minimization problem by ADMM to find low-rank and sparse matrices, L and S, that approximate D as L + S.
    ([vecL; vecS], [z_LS; z_L; z_S]) = arg min_(x,z) g_LS(z_LS) + g_L(z_L) + g_S(z_S)
                                s.t. [z_LS; z_L; z_S] = [I I; I O; O I] * [vecL; vecS].
    Here, by default,
    g_LS(z_LS) = indicator function, i.e., zero if ||D - mat(z_LS)||_F <= tol, infinity otherwise,
    g_L(z_L) = ||mat(z_L)||_*, i.e., nuclear norm of mat(z_L),
    g_S(z_S) = ||ls.*z_S||_1, i.e., l1 norm of mat(z_S) with the weight ls.
    
    Parameters
    ----------
    D : array_like, shape (`m`, `n`)
        `m` x `n` matrix to be separated into L and S.
    tol : scalar, optional, default None
        Tolerance for residual. If None, tol = rtol * ||D||_F
    ls : scalar, optional, default None
        Weight of sparse regularizer.  `ls` can be a matrix of weights for each entries of `z_S.reshape(m,n)`.
        If None, ls = 1./sqrt(max(D.shape)).
    rtol : scalar, optional, default 1e-12
        Relative convergence tolerance of `u` and `z` in ADMM, i.e., the primal and dual residuals.
    rho : scalar, optional, default 1.
        Augmented Lagrangian parameter.
    maxiter : int, optional, default 300
        Maximum iterations.
    verbose: int or bool, optional, default False
        Print the costs every this number.
    nesterovs_momentum : bool, optional, default False
        Nesterov acceleration.
    restart_every : int, optional, default `np.nan`
        Restart the Nesterov acceleration every `restart_every` iterations. If `np.nan`, this is disabled.
    prox_LS : function, optional, default `spmlib.proxop.ind_l2ball`
        Proximity operator as a Python function for the regularizer g_LS of `z_LS` = `vecL`+`vecS`. By default, `prox_LS` is `lambda q,r,c:spmlib.proxop.ind_l2ball(q,r,c)`, i.e., the prox. of the indicator function of l2-ball with radius 'r' and center 'c'.
    prox_L : function, optional, default `spmlib.proxop.squ_l2_from_subspace`
        Proximity operator as a Python function for the regularizer g_L of `z_L` = `vecL`. By default, `prox_L` is `lambda Q,l:spmlib.proxop.nuclear(Q,1)`, i.e., the prox. of the nuclear norm * l*||mat z_L||_*.
    prox_S : function, optional, default `spmlib.proxop.l1`
        Proximity operator as a Python function for the regularizer g_S of `z_S` = `vecS`. By default, `prox_S` is `lambda q,l:spmlib.proxop.l1(q,l)`, i.e., the soft thresholding operator as the prox. of l1 norm ||l.*z_S||_1.

    Returns
    -------
    L : (`m`, `n`) ndarray, x[:m].reshape(`m`,`n`) = np.dot(U, np.dot(sv, Vh))
        Low-rank matrix estimate.
    S : (`m`, `n`) ndarray, x[m:].reshape(`m`,`n`)
        Sparse matrix estimate.
    U : ndarray, shape (`M`, `r`) with ``r = min(m, n)``
        Unitary matrix having left singular vectors as columns.
    sv : ndarray, shape (`r`,)
        Singular values, sorted in non-increasing order.
    Vh : ndarray, shape (`r`, `n`)
        Unitary matrix having right singular vectors as rows.
    count : int
        Loop count at termination.
        
    Example
    -------
    >>> stable_principal_component_pursuit(D, ls=1., tol=1e-4*linalg.norm(D), rtol=1e-4, maxiter=300, verbose=10)[0]
    See demo_spmlib_solvers_matrix.py
    """
    m, n = D.shape
    mn = m*n
    if ls is None:
        ls = 1./sqrt(max(D.shape))
    ls = np.array(ls).ravel()
    if tol is None:
        tol = rtol * linalg.norm(D)

    # initialize
    x = np.zeros(2*mn, dtype=D.dtype)
    z = np.zeros(3*mn, dtype=D.dtype)
    u = np.zeros_like(z)
    count = 0

    t = 1. #
    while count < maxiter:
        count += 1

        if np.fmod(count, restart_every) == 0: #
            t = 1.
        if nesterovs_momentum:
            told = t
            t = 0.5 * (1. + sqrt(1. + 4. * t * t))

        # update x
        #dx = x.copy()
        q = z - u
        x[:mn] = (1./3.) * (q[:mn] + 2.*q[mn:2*mn] - q[2*mn:])
        x[mn:] = (1./3.) * (q[:mn] - q[mn:2*mn] + 2.*q[2*mn:])
        #dx = x - dx

        # q = G(x) + u
        q[:mn]     = x[:mn] + x[mn:] + u[:mn]
        q[mn:2*mn] = x[:mn]          + u[mn:2*mn]
        q[2*mn:]   = x[mn:]          + u[2*mn:]

        # update z
        dz = z.copy() #
        z[:mn]    = prox_LS(q[:mn], tol, D.ravel())
        L, U, sv, Vh = prox_L(q[mn:2*mn].reshape(m,n), 1./rho)
        z[mn:2*mn] = L.ravel()
        z[2*mn:]  = prox_S(q[2*mn:], ls/rho)
        dz = z - dz #

        if nesterovs_momentum:
           # update x again (heuristic)
           q = z - u
           x[:mn] = (1./3.) * (q[:mn] + 2.*q[mn:2*mn] - q[2*mn:])
           x[mn:] = (1./3.) * (q[:mn] - q[mn:2*mn] + 2.*q[2*mn:])

        # update u
        # u = u + G(x) - z
        du = u.copy()
        u[:mn]     += x[:mn] + x[mn:] - z[:mn]
        u[mn:2*mn] += x[:mn]          - z[mn:2*mn]
        u[2*mn:]   += x[mn:]          - z[2*mn:]
        du = u - du

        # Nesterov acceleration
        if nesterovs_momentum:
#            z = z + ((told - 1.) / t) * dz
            u = u + ((told - 1.) / t) * du    # update u only (heuristic)

        if verbose:
            if np.fmod(count,verbose) == 0:
                print('%2d: ||D-(L+S)||_F = %.2e, ||L||_* = %.2e (rnk=%d), ||S||_1 = %.2e (nnz=%2.1f%%)' 
                      % (count, linalg.norm(x[:mn]+x[mn:]-D.ravel()), 
                         np.sum(sv), np.count_nonzero(sv), 
                         np.sum(np.abs(x[mn:])), 100.*np.count_nonzero(z[2*mn:])/mn))

        # check convergence of primal and dual residuals
        if linalg.norm(du) < rtol * linalg.norm(u) and linalg.norm(dz) < rtol * linalg.norm(z):
            break

    return x[:mn].reshape(m,n), x[mn:].reshape(m,n), U, sv, Vh, count




if __name__ == '__main__':
    from time import time
    rng = np.random.RandomState()

    print('====(Deomo: SPCP)====')

    dtype = np.float32
    m, n, rnk = 3000, 500, 10
    print('D.shape = (%d, %d)' % (m, n))
    Ut = rng.randn(m, rnk).astype(dtype)  # random design
    Vt = rng.randn(rnk, n).astype(dtype)  # random design
    L = Ut.dot(Vt) / sqrt(m)    # rank 10 matrix
    print('singular values of L =')
    print(linalg.svd(L, compute_uv=False)[:10])
    print('mean(abs(L)) = %.2e' % (np.mean(np.abs(L))))

    S = np.zeros_like(L)
    support = rng.choice(m*n, int(m*n*0.15), replace=False)
    S.ravel()[support] = 10.*rng.randn(support.size) / sqrt(m) # sparse matrix
    print('mean(abs(S)) = %.2e, %d nonzeros in S' % (np.mean(np.abs(S)),support.size))

    D = L + S
    E = rng.randn(m,n).astype(dtype) / sqrt(m*n) * linalg.norm(D) * 0.03
    D += E
    print('||D||_F = %.2e, ||D-(L+S)||_F = %.2e' % (linalg.norm(D), linalg.norm(E)))
    
    t0 = time()
    Lest, Sest, _, sest, _, it = stable_principal_component_pursuit(D, tol=linalg.norm(E), ls=None, rtol=1e-2, rho=1, maxiter=100, nesterovs_momentum=False,
                                                                       verbose=10,
                                                                       prox_L=lambda Q,l: th.singular_value_thresholding(Q,l,thresholding=th.smoothly_clipped_absolute_deviation),
                                                                       prox_S=th.smoothly_clipped_absolute_deviation)
#                                                                       prox_L=lambda Q,l: th.singular_value_thresholding(Q,l,thresholding=th.smoothly_clipped_absolute_deviation),
#                                                                       prox_L=lambda Q,l: th.soft_svds(Q, l, k=20, tol=1e-1),
#                                                                       prox_LS=lambda q,r,c: prox.ind_l2ball(q,3.*linalg.norm(D),c),

    np.set_printoptions(suppress=True)
    print('done in %.2fs with %d steps' % (time() - t0, it))
    print('rel. error in L: %.2e,  S: %.2e' % (linalg.norm(Lest-L)/linalg.norm(L), linalg.norm(Sest-S)/linalg.norm(S)))
    print('%d nonzeros in S estimated, mean(abs(S)) = %.2e' % (sum(np.abs(Sest.ravel()) > 1e-2), np.mean(np.abs(Sest))))
    #print('nonzero sv = ', sest[np.nonzero(sest)])
    print('nonzero sv = ')
    print(sest[sest > np.spacing(np.float32(1.0))])
