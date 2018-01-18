# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 08:42:11 2017

@author: tsakai
"""

import numpy as np
from scipy import linalg
import scipy.sparse.linalg as splinalg

from spmlib.linalg import lstsq, spvec


# todo: return nonzero only, then OMP makes it spvec



#%% Matching pursuit (MP)
#
# S .G. Mallat and Z. Zhang. "Matching pursuits with time-frequency dictionaries."
# IEEE TSP 41(12), pp. 3397-3415, 1993.
def matching_pursuit(A, b, tol=1e-5, maxiter=None, toarray=True):
    m, n = A.shape
    if maxiter is None:
        maxiter = m
    xnz = []
    support = []
    r = b.copy()

    # main loop
    count = 0
    while linalg.norm(r) > tol and count < maxiter:
        count += 1
        c = splinalg.aslinearoperator(A).rmatvec(r)        # c = A.conj().T.dot(r)
        s = np.argmax(np.abs(c))
        support.append(s)
        xs = c[s] / linalg.norm(A[:,s])
        xnz.append(xs)
        r -= A[:,s] * xs
    return spvec(n, (xnz, support), duplicate=True, toarray=toarray), r, count



#%% Orthogonal matching pursuit (OMP)
#
# Y. C. Pati, R. Rezaiifar, and P. S. Krishnaprasad, 
# "Orthogonal matching pursuit: Recursive function approximation with applications to wavelet decomposition."
# The Twenty-Seventh Asilomar Conference on Signals, Systems and Computers, pp. 40-44, 1993.
#
# s0 is an initial guess of the support (a set of the indices of nonzeros)
def orthogonal_matching_pursuit(A, b, s0=None, tol=1e-5, maxnnz=None, toarray=True,
                                iter_lim=None, solver='eig', atol=1e-3, btol=1e-3, conlim=1e+4):
    m, n = A.shape
    if maxnnz is None:
        maxnnz = m // 2
    if s0 is None or len(s0) == 0:
        support = np.array([], dtype=int)
        xnz = np.array([])
        r = b.copy()
    else:
        support = np.array(s0, dtype=int)
        xnz, r = lstsq(A, b, support=support,
                       iter_lim=iter_lim, solver=solver, atol=atol, btol=btol, conlim=conlim)

    # main loop
    count = 0
    while len(support) < maxnnz and linalg.norm(r) > tol:
        count += 1
        s = np.argmax(np.abs( splinalg.aslinearoperator(A).rmatvec(r) ))
        support = np.union1d(support, [s])
        xnz, r = lstsq(A, b, support=support,
                       iter_lim=iter_lim, solver=solver, atol=atol, btol=btol, conlim=conlim)
    return spvec(n, (xnz, support), toarray=toarray), r, count



#%% Generalized orthogonal matching pursuit (gOMP)
# 
# J. Wang, S.Kwon, and B. Shim, "Generalized orthogonal matching pursuit",
# IEEE TSP, 60(12), pp. 6202-6216, 2012.
def generalized_orthogonal_matching_pursuit(A, b, N=3, s0=None, tol=1e-5, maxnnz=None, toarray=True,
                                            iter_lim=None, solver='eig', atol=1e-3, btol=1e-3, conlim=1e+4):
    m, n = A.shape
    if maxnnz is None:
        maxnnz = m // 2
    if s0 is None or len(s0) == 0:
        support = np.array([], dtype=int)
        xnz = np.array([])
        r = b.copy()
    else:
        support = np.array(s0, dtype=int)
        xnz, r = lstsq(A, b, support=support,
                       iter_lim=iter_lim, solver=solver, atol=atol, btol=btol, conlim=conlim)

    # main loop
    count = 0
    while len(support) < min(maxnnz, m/N) and linalg.norm(r) > tol:
        count += 1
        c = splinalg.aslinearoperator(A).rmatvec(r)        # c = A.conj().T.dot(r)
        s = np.argsort(-np.abs(c))
        support = np.union1d(support, s[:N])
        xnz, r = lstsq(A, b, support=support,
                       iter_lim=iter_lim, solver=solver, atol=atol, btol=btol, conlim=conlim)
        # xnz = linalg.lstsq(A[:,T],b)[0]
        #r = b - A.dot(x)
    return spvec(n, (xnz, support), toarray=toarray), r, count



#%% Subspace pursuit
# 
# W. Dai and M. O. Milenkovic, "Subspace pursuit for compressive sensing signal reconstruction",
# IEEE TIT, 55(5), pp.2230-2249, 2009.
def subspace_pursuit(A, b, K=None, s0=None, maxiter=None, toarray=True,
                     iter_lim=None, solver='eig', atol=1e-3, btol=1e-3, conlim=1e+4):
    m, n = A.shape
    if K is None:
        #K = m // 4
        K = int(0.5*m / np.log2(n/m))
        K = min(K, m//4)
    if maxiter is None:
        maxiter = K
    if s0 is None or len(s0) == 0:
        c = splinalg.aslinearoperator(A).rmatvec(b)
        supp = np.array(np.argsort(-np.abs(c))[:K], dtype = int)
    else:
        supp = np.array(s0, dtype=int)
    xnzp, r = lstsq(A, b, support=supp,
                    iter_lim=iter_lim, solver=solver, atol=atol, btol=btol, conlim=conlim)

    #Told = np.array([], dtype = int)
    normrold = np.inf
    normr = linalg.norm(r)
    
    # main loop
    count = 0
    while normr < normrold and count < maxiter:
        count += 1
        normrold = normr
        support = supp
        xnz = xnzp
        # T = union of T and {K indices corresponding to the largest magnitude entries in the vector A'*r}
        c = splinalg.aslinearoperator(A).rmatvec(r)
        supp = np.union1d(supp, np.argsort(-np.abs(c))[:K])
        xnzp, r = lstsq(A, b, support=supp,
                        iter_lim=iter_lim, solver=solver, atol=atol, btol=btol, conlim=conlim)
        # T = {K indices corresponding to the largest elements of xp}
        supp = supp[np.argsort(-np.abs(xnzp))[:K]]
        xnzp, r = lstsq(A, b, support=supp,
                        iter_lim=iter_lim, solver=solver, atol=atol, btol=btol, conlim=conlim)
        normr = linalg.norm(r)
    return spvec(n, (xnz, support), toarray=toarray), r, count
