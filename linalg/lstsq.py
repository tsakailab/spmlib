# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 08:42:11 2017

@author: tsakai
"""

import numpy as np
from scipy import linalg
import scipy.sparse.linalg as splinalg


#%% Create full vector from its nonzero components and support (can be duplicate)
def spvec(n, nonzeros, support, duplicate=False):
    spv = np.zeros(n)
    if not duplicate:
        spv = np.zeros(n)
        spv[support] = nonzeros
    else:
        # because spv[support] += nonzeros doesn't work as expected ..
        j = 0
        for s in support:
            spv[s] += nonzeros[j]
            j += 1
    return spv



#%%
# Compute least squares B = lstsq(A[:,ls], B)[0] and its residues
# A is a matrix or a linear operator
def lstsq(A, B, tol=1e-5, support=None, x0=None, max_nnz=None, maxiter=None, solver='eig', atol=1e-3, btol=1e-3, conlim=1e+4):

    m, n = A.shape
    if support is not None:
        ls = list(support)
    else:
        ls = range(n)
    s = len(ls)
    B = np.atleast_2d(B).T
    c = B.shape[1]

    # if A is a numpy array
    if type(A) is np.ndarray:
        As = A[:,ls]
        if solver == 'eig':
            # Using solve (eig-based) is faster ..
            Xnz = linalg.solve(As.conj().T.dot(As), As.conj().T.dot(B)).reshape((s,c))
        else:
            Xnz = linalg.lstsq(As, B, overwrite_a=True, overwrite_b=True, lapack_driver=solver)[0]

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

    R = B - As.dot(Xnz)
    if c == 1:
        R = R.ravel()
        Xnz = Xnz.ravel()
    return Xnz, R



#%%
# Solve (As'*As)*x=As*b for x as lstsq(A,b)[0]
def lstsq_cg(As, b, x0=None, tol=1e-05, maxiter=None, xtype=None, M=None, callback=None):
    As = splinalg.aslinearoperator(As)
    AA = splinalg.LinearOperator((As.shape[1],As.shape[1]), matvec=lambda x: As.rmatvec(As.matvec(x)), rmatvec=lambda x: As.rmatvec(As.matvec(x)))
    return splinalg.cg(AA, As.rmatvec(b), x0=x0, tol=tol, maxiter=maxiter, xtype=xtype, M=M, callback=callback)

