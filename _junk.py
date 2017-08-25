#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 23:30:00 2017

@author: tsakai
"""
from scipy.linalg.blas import dgemm

# in lstsq.py
#%%
# Compute least squares b = lstsq(A[:,ls], b)[0] and its residues
# A is a matrix or a linear operator
def lstsq(A_ls, b, tol=1e-5, x0=None, max_nnz=None, maxiter=None, atol=1e-3, btol=1e-3, conlim=1e+4):

    # A and a list ls are given as a tuple
    A, ls = A_ls
    m, n = A.shape
    s = len(ls)
    #if not (b.shape == (m,1) or b.shape == (m,)):
    #    raise ValueError('A and b have incompatible dimensions')
    #xnz = np.zeros((s,))

    # if A is a numpy array
    if type(A) is np.ndarray:
        #As = A[:,ls]
        #xnz = linalg.lstsq(As, b)[0]
        # Using dgemm is much faster ..
        As = np.asarray(A[:,ls], order='C')
        xnz = np.linalg.solve(dgemm(alpha=1.0, a=As.conj().T, b=As.conj().T, trans_b=True),
                              dgemm(alpha=1.0, a=As.conj().T, b=b)).reshape(s,)
    else:
        # Assuming A to be a linear operator, create a linear operator of A[:,ls]
        #A = splinalg.aslinearoperator(A)
        As = splinalg.LinearOperator((m, s), matvec=lambda xnz: A.matvec(spvec(n,xnz,ls)), rmatvec=lambda y: A.rmatvec(y)[ls])
        if maxiter is None:
            maxiter = s
        xnz = splinalg.lsqr(As, b, damp=0.0, atol=atol, btol=btol, conlim=conlim, iter_lim=maxiter, show=False, calc_var=False)[0]
        #xnz = lstsq_cg(As, b, x0=x0, maxiter=maxiter)[0]
    return xnz, b - As.dot(xnz)



#%%
# Compute least squares B = lstsq(A[:,ls], B)[0] and its residues
# A is a matrix or a linear operator
def lstsq(A_ls, B, tol=1e-5, x0=None, max_nnz=None, maxiter=None, atol=1e-3, btol=1e-3, conlim=1e+4):

    # A and a list ls are given as a tuple
    A, ls = A_ls
    m, n = A.shape
    s = len(ls)
    B = np.atleast_2d(B).T
    c = B.shape[1]

    # if A is a numpy array
    if type(A) is np.ndarray:
        #As = A[:,ls]
        #Xnz = linalg.lstsq(As, B)[0]
        # Using dgemm is faster ..
        As = np.asarray(A[:,ls], order='C')
        Xnz = np.linalg.solve(dgemm(alpha=1.0, a=As.conj().T, b=As.conj().T, trans_b=True),
                              dgemm(alpha=1.0, a=As.conj().T, b=B)).reshape((s,c))
        #Xnz = np.linalg.solve(As.conj().T.dot(As), As.conj().T.dot(B)).reshape((s,c))

    else:
        # Assuming A to be a linear operator, create a linear operator of A[:,ls]
        #A = splinalg.aslinearoperator(A)
        As = splinalg.LinearOperator((A.shape[0],len(ls)), matvec=lambda xnz: A.matvec(spvec(A.shape[1],xnz,ls)), rmatvec=lambda y: A.rmatvec(y)[ls])

        if maxiter is None:
            maxiter = s
        Xnz = np.zeros((s,c))
        for j in range(c):
            Xnz[:,j] = splinalg.lsqr(As, B[:,j], damp=0.0, atol=atol, btol=btol, conlim=conlim, iter_lim=maxiter, show=False, calc_var=False)[0]
            #Xnz[:,j] = lstsq_cg(As, B[:,j], x0=x0, maxiter=maxiter)[0]

    if c == 1:
        R = (B - As.dot(Xnz)).ravel()
        Xnz = Xnz.ravel()
    return Xnz, R



#%% Orthogonal matching pursuit (OMP)
# s0 is an initial guess of the support (a set of the indices of nonzeros)
def npls_orthogonal_matching_pursuit(A, b, s0=None, tol=1e-5, max_nnz=None):
    m, n = A.shape
    if max_nnz is None:
        max_nnz = m // 2
    if s0 is None or len(s0) == 0:
        ls = np.array([], dtype = int)        #support = set([])
        xnz = np.array([])
        r = b.copy()
    else:
        ls = np.array(s0)
        #support = set(s0)
        #ls = list(support)
        xnz, r = lstsq((A,ls),b)

    while len(ls) < max_nnz and linalg.norm(r) > tol:
        ls = np.union1d(ls, [np.argmax(np.abs( splinalg.aslinearoperator(A).rmatvec(r) ))])
        #support.add(np.argmax(np.abs( splinalg.aslinearoperator(A).rmatvec(r) )))
        #ls = list(support)
        #xnz, r = lstsq((A,ls),b,x0=np.concatenate((xnz.ravel(),np.array([0.]))),maxiter=10)
        xnz, r = lstsq(A, b, support=ls)
    return spvec(n, xnz, ls), r




#%% Generalized orthogonal matching pursuit (gOMP)
# 
# J. Wang, S.Kwon, and B. Shim, "Generalized orthogonal matching pursuit",
# IEEE TSP, 60(12), pp. 6202-6216, 2012.
def npls_generalized_orthogonal_matching_pursuit(A, b, N=3, tol=1e-5, max_nnz=None):
    m, n = A.shape
    #b = bb.reshape(m,1)
    if max_nnz is None:
        max_nnz = m // 2
    x = np.zeros(n)
    T = np.array([], dtype = int)
    r = b.copy()
    count = 0
    while T.size < min(max_nnz, m/N) and linalg.norm(r) > tol:
        count += 1
        c = splinalg.aslinearoperator(A).rmatvec(r)        # c = A.conj().T.dot(r)
        s = np.argsort(-np.abs(c))
        T = np.union1d(T, s[:N])
        x[T] = linalg.lstsq(A[:,T],b)[0]
        r = b - A.dot(x)
    return x, r, count



#%% Generalized orthogonal matching pursuit (gOMP)
# 
# W. Dai and M. O. Milenkovic, "Subspace pursuit for compressive sensing signal reconstruction",
# IEEE TIT, 55(5), pp.2230-2249, 2009.
def subspace_pursuit(A, b, K=None, tol=1e-5, maxiter=None):
    m, n = A.shape
    if K is None:
        K = m // 4
    if maxiter is None:
        maxiter = K
    #Told = np.array([], dtype = int)
    T = np.array(np.argsort(-np.abs(A.conj().T.dot(b)))[:K], dtype = int)
    r = b - A[:,T].dot(linalg.lstsq(A[:,T],b)[0])
    normrold = np.inf
    normr = linalg.norm(r)
    count = 0
    while normr < normrold and count < maxiter:
        count += 1
        normrold = normr
        Told = T
        # T = union of T and {K indices corresponding to the largest magnitude entries in the vector A'*r}
        T = np.union1d(T, np.argsort(-np.abs(A.conj().T.dot(r)))[:K])
        xp = linalg.lstsq(A[:,T],b)[0]
        # T = {K indices corresponding to the largest elements of xp}
        T = T[np.argsort(-np.abs(xp))[:K]]
        r = b - A[:,T].dot(linalg.lstsq(A[:,T],b)[0])
        normr = linalg.norm(r)
    x = np.zeros(n)
    x[Told] = linalg.lstsq(A[:,Told],b)[0]
    return x, b-A.dot(x), count
