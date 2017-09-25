# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 08:42:11 2017

@author: tsakai
"""

import numpy as np
from scipy import linalg
import scipy.sparse.linalg as splinalg



#%%
# Compute least squares B = lstsq(A[:,support], B)[0] and its residues
# A is a matrix or a linear operator
def lstsq(A, B, support=None, x0=None, iter_lim=None, solver='eig', atol=1e-3, btol=1e-3, conlim=1e+4):

    m, n = A.shape
    if support is not None:
        ls = list(support)
    else:
        ls = range(n)
    s = len(ls)
    B = np.atleast_2d(B.T).T  # (m,)->(m,1), (m,n)
    c = B.shape[1]

    # if A is a numpy array
    if type(A) is np.ndarray:
        As = A[:,ls]
        if solver == 'eig':
            Xnz = linalg.solve(As.conj().T.dot(As), As.conj().T.dot(B)).reshape((s,c))    # Using solve (eig-based) is faster ..
        else:
            Xnz = linalg.lstsq(As, B, overwrite_a=True, overwrite_b=True, lapack_driver=solver)[0]  # I prefer this for the better condition number ..

    else:
        # Assuming A to be a linear operator, create a linear operator of A[:,ls]
        #A = splinalg.aslinearoperator(A)
        As = splinalg.LinearOperator((A.shape[0],len(ls)), matvec=lambda xnz: A.matvec(spvec(A.shape[1],xnz,ls)), rmatvec=lambda y: A.rmatvec(y)[ls])

        if iter_lim is None:
            iter_lim = s
        Xnz = np.zeros((s,c))
        for j in range(c):
            Xnz[:,j] = splinalg.lsqr(As, B[:,j], damp=0.0, atol=atol, btol=btol, conlim=conlim, iter_lim=iter_lim, show=False, calc_var=False)[0]
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

