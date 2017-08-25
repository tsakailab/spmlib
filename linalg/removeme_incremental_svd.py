# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:19:19 2017

@author: tsakai
"""
#from __future__ import print_function

import numpy as np
from scipy import linalg


#%%
# Incremental SVD
#
# U, s = column_incremental_SVD(U, s, C, forget=1., max_rank=np.inf, min_sv=0, orth_eps=1e-12, OrthogonalizeU=False)
#
# performs the incremental SVD in
# M. Brand, "Incremental singular value decomposition of uncertain data with missing values",
# ECCV2002, p. 707-720, 2002.
#
# input:
# U        : m x r matrix of left singular vectors
# s        : r-dimensional vector of singular values
# C        : m x nc matrix of column vectors to append
# forget   : forgetting parameter (0<forget<=1, 1 by default)
# max_rank : maximum rank (m by default)
# min_sv   : smaller singular values than min_sv is neglected (0 by default)
# orth_eps : rank increases if the magnitude of C in the orthogonal subspace is larger than orth_eps (1e-12 by default)
# OrthogonalizeU : if True, perform QR decomposition to orthogonalize U (True by default)
#
# output:
# U        : updated U
# s        : updated s
#
# Example:
# m, n = Y.shape
# U, s = np.empty([0,0]), np.empty(0)
# count = 0
# for j in range(n):
#    U, s = column_incremental_SVD(U, s, Y[:,j:j+1], max_rank=50, orth_eps=linalg.norm(Y[:,j:j+1])*1e-12)
#    count = count + 1
#
def column_incremental_SVD(U, s, C, forget=1., max_rank=np.inf, min_sv=0, orth_eps=1e-12, OrthogonalizeU=False):

    r = s.size
    
    if r == 0:
        U, s, V = linalg.svd(C, full_matrices=False)
        return U, s

    Cin = U.conj().T.dot(C)   # Cin = U' * C;
    Cout = C - U.dot(Cin)        # Cout = C - U * Cin;
    Q, R = linalg.qr(Cout, mode='economic')     # QR decomposition

    if linalg.norm(R) > orth_eps:
        # track moving subspace, rank can increase up to max_rank.
        # B = [f * diag(diagK), Cin; zeros(size(C,2), r), R];
        B = np.concatenate((np.concatenate((np.diag(forget * s), Cin),1), np.concatenate((np.zeros((C.shape[1], r)), R), 1)))

        # [UB, s, ~] = svd(full(B), 'econ'); s = diag(s);
        UB, s, VB = linalg.svd(B, full_matrices=False)

        # rank must not be greater than max_rank, and the singlular values must be greater than min_sv
        r = min(max_rank, s.size)       # r = min(max_rank, numel(s));
        s = s[:r]                       # s = s(1:r);
        s = s[s>=min_sv]                # s = s(s >= min_sv);
        r = s.size                      # ｒ = numel(s);
        U = np.concatenate((U, Q), 1).dot(UB)    # U = [U, Q] * UB;
        U = U[:,:r]                     # U = U(:,1:r)
        if OrthogonalizeU:
            U = linalg.qr(U, mode='economic')[0]    # [U,~] = qr(U(:,1:r),0);

    else:
        # non-moving (rotating) subspace, rank non-increasing
        B = np.concatenate((np.diag(forget * s), Cin),1)

        # [UB, s, ~] = svd(full(B), 'econ'); s = diag(s);
        UB, s, VB = linalg.svd(B, full_matrices=False)

        U = U.dot(UB)    # U = U * UB;
        if OrthogonalizeU:
            U = linalg.qr(U, mode='economic')[0]    # [U,~] = qr(U,0);
        

    #v = sp.diags(1/s, format='csr').dot(U.conj().T.dot(Xc))

    return U, s #, v



if __name__ == "__main__":    
    from time import time
    from math import sqrt
    print('====(Deomo: incremental SVD)====')
    rng = np.random.RandomState(int(time()))
    m, n, rnk = 3000, 2000, 10
    Ut = rng.randn(m, rnk)  # random design
    Vt = rng.randn(rnk, n)  # random design
    Y = Ut.dot(Vt) / sqrt(m)
    
    t0 = time()
    U, s, V = linalg.svd(Y, full_matrices=False)
    print('done in %.2fs by batch SVD.' % (time() - t0))
    print('rank = %d, nonzero sv = ' % (s.size), s[s > np.spacing(np.float32(1.0))])
    
    t0 = time()
    U_inc, s_inc = np.empty([0,0]),np.empty(0)
    count = 0
    for j in range(n):
        U_inc, s_inc = column_incremental_SVD(U_inc, s_inc, Y[:,j:j+1], max_rank=20,
                                              orth_eps=linalg.norm(Y[:,j:j+1])*1e-12, OrthogonalizeU=True)
        count = count + 1
    
    print('done in %.2fs (%d iter.) by incremental SVD.' % (time() - t0, count))
    print('rank = %d, nonzero sv = ' % (s_inc.size), s_inc[s_inc > np.spacing(np.float32(1.0))])


