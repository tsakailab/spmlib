#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:15:10 2017

@author: tsakai
"""

from math import sqrt
import numpy as np
from scipy import linalg
import spmlib.proxop as prox
import spmlib.thresholding as th


#%%
# Example:
# from spmlib import solver as sps
# U, sv = np.empty([0,0]), np.empty(0)  # initialize
# L, S = np.zeros(C.shape), np.zeros(C.shape)
# for j in range(C.shape[1]):
#    Lest[:,j], Sest[:,j], U, sv = sps.OnlSPCP_SCAD(C[:,j], U, sv,
#                                                   ls=0.05, maxiter=100, switch_to_scad_after=40,
#                                                   rtol=1e-4, rdelta=1e-6, max_rank=20)
#
def OnlSPCP_SCAD(c, U, sv,
                 l=None, s=None, rtol=1e-4, maxiter=1000,
                 rdelta=1e-4, ls=1., rho=1., update_basis=True,
                 adjust_basis_every=np.nan, forget=1., max_rank=np.inf, min_sv=0., rorth_eps=1e-8, orthogonalize_basis=True,
                 nesterovs_momentum=False, restart_every = np.nan,
                 a=3.7, switch_to_scad_after=0):
    # OnlSPCP up to switch_to_scad_after times to find good initial guess
    normc = linalg.norm(c)
    if switch_to_scad_after > 0:
        l, s, UU, svsv, count = column_incremental_stable_principal_component_pursuit(c, U, sv, 
                        l=l, s=s, rtol=rtol, maxiter=switch_to_scad_after,
                        delta=normc*rdelta, ls=ls, rho=rho, update_basis=False,
                        adjust_basis_every=adjust_basis_every, forget=forget, max_rank=max_rank, min_sv=min_sv, orth_eps=normc*rorth_eps, orthogonalize_basis=False,
                        nesterovs_momentum=nesterovs_momentum, restart_every = restart_every)

    # OnlSPCP with SCAD thresholding to debias
    return column_incremental_stable_principal_component_pursuit(c, U, sv, 
                        l=l, s=s, rtol=rtol, maxiter=maxiter,
                        delta=normc*rdelta, ls=ls, rho=rho, update_basis=update_basis,
                        adjust_basis_every=adjust_basis_every, forget=forget, max_rank=max_rank, min_sv=min_sv, orth_eps=normc*rorth_eps, orthogonalize_basis=orthogonalize_basis,
                        nesterovs_momentum=nesterovs_momentum, restart_every = restart_every,
                        prox_s=lambda q,ls: th.smoothly_clipped_absolute_deviation(q,ls,a=a))



#%%
def column_incremental_SVD(C, U, sv, forget=1., max_rank=np.inf, min_sv=0., orth_eps=1e-12, orthogonalize_basis=False):
    """
    incremental SVD
    performs the incremental SVD.

    Parameters
    ----------
    C : ndarray, shape (`m`, `nc`)
        `m` x `nc` matrix of column vectors to append.
    U : ndarray, shape (`m`, `r`)
        `m` x `r` matrix of left singular vectors (overwritten with the update).
    sv : array_like, shape(`r`,)
        `r`-dimensional vector of singular values (overwritten with the update).
    forget : scalar, optional, default 1.
        Forgetting parameter (0<`forget` and `forget` <=1).
    max_rank : int, optional, default `m`
        Maximum rank.
    min_sv : scalar, optional, default 0.
        Singular values smaller than `min_sv` is neglected. `sv` >= max(`sv`)*abs(`min_sv`) if `min_sv` is negative.
    orth_eps : scalar, optional, default 1e-12
        Rank increases if the magnitude of `C` in the orthogonal subspace is larger than `orth_eps`.
    orthogonalize_basis : bool, optional, default False
        If True, perform QR decomposition to orthogonalize `U`.

    Returns
    -------
    U : ndarray
        Updated U.
    sv : ndarray
        Updated sv.

    References
    ----------
    M. Brand
    "Incremental singular value decomposition of uncertain data with missing values"
    ECCV2002, pp. 707-720, 2002.

    Example
    -------
    >>> from spmlib import solver as sps
    >>> m, n = Y.shape
    >>> U, sv = np.empty([0,0]), np.empty(0)  # initialize
    >>> count = 0
    >>> for j in range(C.shape[1]):
    >>>     U, sv = sps.column_incremental_SVD(Y[:,j:j+1], U, sv, max_rank=50, orth_eps=linalg.norm(Y[:,j:j+1])*1e-12)
    >>>     count = count + 1
    
    """
    r = sv.size
    
    if r == 0:
        U, sv, V = linalg.svd(np.atleast_2d(C.T).T, full_matrices=False)
        return U, sv

    Cin = U.conj().T.dot(C)   # Cin = U' * C;
    Cout = C - U.dot(Cin)        # Cout = C - U * Cin;
    Cin, Cout = np.atleast_2d(Cin.T).T, np.atleast_2d(Cout.T).T
    Q, R = linalg.qr(Cout, mode='economic')     # QR decomposition

    if linalg.norm(R) > orth_eps:
        # track moving subspace, rank can increase up to max_rank.
        # B = [f * diag(diagK), Cin; zeros(size(C,2), r), R];
        B = np.concatenate((
                np.concatenate((np.diag(forget * sv), Cin),1), 
                np.concatenate((np.zeros((np.atleast_2d(C.T).T.shape[1], r), dtype=Cin.dtype), R), 1)))

        # [UB, sv, ~] = svd(full(B), 'econ'); sv = diag(sv);
        UB, sv, VB = linalg.svd(B, full_matrices=False)

        # rank must not be greater than max_rank, and the singlular values must be greater than min_sv
        r = min(max_rank, sv.size)       # r = min(max_rank, numel(sv));
        sv = sv[:r]                       # sv = sv(1:r);
        sv = sv[sv>=min_sv] if min_sv >= 0. else sv[sv>=-min_sv*np.max(sv)]  # sv = sv(sv >= min_sv);
        r = sv.size                      # ｒ = numel(sv);
        U = np.concatenate((U, Q), 1).dot(UB)    # U = [U, Q] * UB;
        U = U[:,:r]                     # U = U(:,1:r)
        if orthogonalize_basis:
            U = linalg.qr(U, mode='economic')[0]    # [U,~] = qr(U(:,1:r),0);

    else:
        # non-moving (rotating) subspace, rank non-increasing
        B = np.concatenate((np.diag(forget * sv), Cin),1)

        # [UB, sv, ~] = svd(full(B), 'econ'); sv = diag(sv);
        UB, sv, VB = linalg.svd(B, full_matrices=False)

        U = U.dot(UB)    # U = U * UB;
        if orthogonalize_basis:
            U = linalg.qr(U, mode='economic')[0]    # [U,~] = qr(U,0);
        

    #v = sp.diags(1/sv, format='csr').dot(U.conj().T.dot(Xc))

    return U, sv #, v



#%%
def column_incremental_stable_principal_component_pursuit(c, U, sv, 
                        l=None, s=None, rtol=1e-12, maxiter=1000,
                        delta=1e-12, ls=1., rho=1., update_basis=False,
                        adjust_basis_every=np.nan, forget=1., max_rank=np.inf, min_sv=0., orth_eps=1e-12, orthogonalize_basis=False,
                        nesterovs_momentum=False, restart_every = np.nan,
                        prox_ls=lambda q,r,c: prox.ind_l2ball(q,r,c),
                        prox_l=lambda q,th,U: prox.squ_l2_from_subspace(q,U,th), 
                        prox_s=lambda q,ls: prox.l1(q,ls) ):
    """
    Column incremental online stable principal component pursuit (OnlSPCP)
    performs the low-rank and sparse matrix approximation by solving
    ([l;s], [zls; zl; zs]) = arg min_(x,z)  g_ls(z_ls) + g_l(z_l) + g_s(z_s)
                                s.t. [zls; zl; zs] = [I I; I O; O I] * [l; s].
    Here, by default,
    g_ls(z_ls) = indicator function, i.e., zero if ||c - z_ls||_2 <= delta, infinity otherwise,
    g_l(z_l) = 0.5 * ||(I-U*U')*z_l||_2^2,
    g_s(z_s) = ||ls.*z_s||_1

    Parameters
    ----------
    c : ndarray, shape (`m`,)
        `m`-dimensional vector to be decomposed into `l` and `s` such that ||d-(l+s)||_2<=delta.
    U : ndarray, shape (`m`,`r`)
        `m` x `r` matrix of left singular vectors approximately spanning the subspace of low-rank components
        (overwritten with the update if update_basis is True).
    sv : array_like, shape ('r',)
        `r`-dimensional vector of singular values
        (overwritten with the update if update_basis is True).
    l : array_like, shape (`m`,), optional, default None
        Initial guess of `l`. If None, `c`-`s` is used.
    s : array_like, shape (`m`,), optional, default None
        Initial guess of `s`. If None, `s` is numpy.zeros_like(c) is used.
    rtol : scalar, optional, default 1e-12
        Relative convergence tolerance of `x` and `z` in ADMM.
    maxiter : int, optional, default 1000
        Maximum iterations.
    delta : scalar, optional, default 1e-12
        l2-ball radius used in the indicator function for the approximation error.
    ls : scalar or 1d array, optional, default 1.
        Weight of sparse regularizer.  `ls` can be a 1d array of weights for the entries of `s`.
    rho : scalar, optional, default 1.
        Augmented Lagrangian parameter.
    update_basis : bool, optional, default False
        Update `U` and `sv` with `l` after convergence.
    adjust_basis_every : int, optional, default `np.nan`
        Temporalily update `U` with `l` every `adjust_basis_every` iterations in the ADMM loop. If `np.nan`, this is disabled.
    forget : scalar, optional, default 1.
        Forgetting parameter in updating `U`.
    max_rank : int, optional, default np.inf
        Maximum rank. `U.shape[1]` and `sv.shape[0]` won't be greater than `max_rank`.
    min_sv : scalar, optional, default 0.
        Singular values smaller than `min_sv` is neglected. `sv` >= max(`sv`)*abs(`min_sv`) if `min_sv` is negative.
    orth_eps : scalar, optional, default 1e-12
        Rank increases if the magnitude of `c` in the orthogonal subspace is larger than `orth_eps`.
    orthogonalize_basis : bool, optional, default False
        If True, perform QR decomposition to orthogonalize `U`.
    nesterovs_momentum : bool, optional, default False
        Nesterov acceleration.
    restart_every : int, optional, default `np.nan`
        Restart the Nesterov acceleration every `restart_every` iterations. If `np.nan`, this is disabled.
    prox_ls : function, optional, default `spmlib.proxop.ind_l2ball`
        Proximity operator as a Python function for the regularizer g_ls of `z_ls` = `l`+`s`. By default, `prox_ls` is `lambda q,r,c:spmlib.proxop.ind_l2ball(q,r,c)`, i.e., the prox. of the indicator function of l2-ball with radius 'r' and center 'c'.
    prox_l : function, optional, default `spmlib.proxop.squ_l2_from_subspace`
        Proximity operator as a Python function for the regularizer g_l of `z_l` = `l`. By default, `prox_l` is `lambda q,U:spmlib.proxop.squ_l2_from_subspace(q,U,th)`, i.e., the prox. of the distance function defined as 0.5*(squared l2 distance between `l` and span`U`).
    prox_s : function, optional, default `spmlib.proxop.l1`
        Proximity operator as a Python function for the regularizer g_s of `z_s` = `s`. By default, `prox_s` is `lambda q,ls:spmlib.proxop.l1(q,ls)`, i.e., the soft thresholding operator as the prox. of l1 norm ||ls.*z_s||_1.

    Returns
    -------
    l : ndarray, shape (`m`,)
        Low-rank component.
    s : ndarray, shape (`m`,)
        Sparse component
    U : ndarray
        Matrix of left singular vectors.
    sv : ndarray
        Vector of singular values.
    count : int
        Iteration count.

    References
    ----------
    Tomoya Sakai, Shun Ogawa, and Hiroki Kuhara
    "Sequential decomposition of 3D apparent motion fields basedon low-rank and sparse approximation"
    APSIPA2017 (to appear).

    Example
    -------
    >>> from spmlib import solver as sps
    >>> U, sv = np.empty([0,0]), np.empty(0)  # initialize
    >>> L, S = np.zeros(C.shape), np.zeros(C.shape)
    >>> for j in range(n):
    >>>     L[:,j], S[:,j], U, sv = sps.column_incremental_stable_principal_component_pursuit(C[:,j], U, sv, ls=0.5, update_basis=True, max_rank=50, orth_eps=linalg.norm(Y[:,j])*1e-12)[:4]

    """
    m = c.shape[0]

    # initialize l and s
    if s is None:
        s = np.zeros_like(c) # np.zeros(m, dtype=c.dtype)
    if l is None:
        l = c.ravel() - s
    
    if sv.size == 0:
        U, sv, V = linalg.svd(np.atleast_2d(c.T).T, full_matrices=False)
        return l, s, U, sv, 0

    # G = lambda x: np.concatenate((x[:m]+x[m:], x[:m], x[m:]))
    # x = np.concatenate((l,s))
    x = np.zeros(2*m, dtype=c.dtype)
    x[:m] = l
    x[m:] = s

    # z = G(x)
    z = np.zeros(3*m, dtype=c.dtype)
    z[:m]    = x[:m] + x[m:]
    z[m:2*m] = x[:m]
    z[2*m:]  = x[m:]

    y = np.zeros_like(z) # np.zeros(3*m, dtype=c.dtype)
    
    t = 1.
    count = 0
    Ut = U
    while count < maxiter:
        count += 1

        if np.fmod(count, restart_every) == 0:
            t = 1.
        if nesterovs_momentum:
            told = t
            t = 0.5 * (1. + sqrt(1. + 4. * t * t))

        # update x
        dx = x.copy()
        q = z - y
        x[:m] = (1./3.) * (q[:m] + 2.*q[m:2*m] - q[2*m:])
        x[m:] = (1./3.) * (q[:m] - q[m:2*m] + 2.*q[2*m:])
        dx = x - dx
        
        # q = G(x) + y
        q[:m]    = x[:m] + x[m:] + y[:m]
        q[m:2*m] = x[:m]         + y[m:2*m]
        q[2*m:]  = x[m:]         + y[2*m:]
        
        # update z
        if np.fmod(count, adjust_basis_every) == 0:
            Ut = column_incremental_SVD(x[:m], U, sv, 
                                           forget=forget, max_rank=max_rank, min_sv=min_sv,
                                           orth_eps=orth_eps, orthogonalize_basis=False)[0]
        dz = z.copy()
        z[:m]    = prox_ls(q[:m], delta, c.ravel())
        z[m:2*m] = prox_l(q[m:2*m], 1./rho, Ut)
        z[2*m:]  = prox_s(q[2*m:], ls/rho)
        dz = z - dz

        # update y
        #y = y + G(x) - z
        dy = y.copy()
        y[:m]    += x[:m] + x[m:] - z[:m]
        y[m:2*m] += x[:m]         - z[m:2*m]
        y[2*m:]  += x[m:]         - z[2*m:]
        dy = y - dy

        # Nesterov acceleration
        if nesterovs_momentum:
            z = z + ((told - 1.) / t) * dz
            y = y + ((told - 1.) / t) * dy
        
        # check convergence
        if linalg.norm(dx) < rtol * linalg.norm(x) and linalg.norm(dz) < rtol * linalg.norm(z):
            break
        
    l = x[:m]
    s = x[m:]
    if update_basis:
        U, sv = column_incremental_SVD(l, U, sv, 
                                       forget=forget, max_rank=max_rank, min_sv=min_sv,
                                       orth_eps=orth_eps, orthogonalize_basis=orthogonalize_basis)

    return l, s, U, sv, count


