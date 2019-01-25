    # -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 19:36:00 2017

@author: tsakai
"""

from math import sqrt
import numpy as np
from scipy import linalg
from time import time


from spmlib import solver as sps
from spmlib import thresholding as th

np.set_printoptions(suppress=True)



print('====(Deomo 1: low-rank matrix completion)====')
#Y = np.array([[     5,      0,      5,      0],
#              [     0, np.nan, np.nan,      4],
#              [np.nan,      0,      4, np.nan],
#              [     5,      1,      4,      0],
#              [     0,      5,      0, np.nan]])
#Y = np.array([[     5,      5,      0,      0],
#              [     5, np.nan, np.nan,      0],
#              [np.nan,      4,      0, np.nan],
#              [     0,      0,      5,      4],
#              [     0,      0,      5, np.nan]])
#Y = np.array([[     5,      0,      5,      1],
#              [np.nan,      0,      4, np.nan],
#              [     0,      5,      0,      4],
#              [     4, np.nan, np.nan,      0],
#              [     0,      5,      0, np.nan]])
#Y = np.array([[     5,      0,      5,      1],
#              [     4,      0,      4,      1],
#              [     0,      5,      0,      4],
#              [     4,      0,      5,      0],
#              [     0,      5,      0,      4]])
Y = np.array([[     4,      0,      5,      1],
              [np.nan,      0,      4, np.nan],
              [     0,      5,      0,      4],
              [     4, np.nan, np.nan,      0],
              [     0,      4,      0, np.nan]])
#Y = np.array([[     4,      0,      5,      1],
#              [     3,      0,      4,      1],
#              [     0,      5,      0,      4],
#              [     4,      0,      5,      0],
#              [     0,      4,      0,      3]])
print('Y = ')
print(Y)
#result = LowRankMatrixCompletion(Y, l=1,tol=linalg.norm(Y[~np.isnan(Y)])*1e-6, maxit=100)
#print('Yest = ', result.Yest)
#print('sv = ', result.s)
a = 3.7
Yest, _, s, _, it = sps.low_rank_matrix_completion(Y, l=1., rtol=1e-4, tol=linalg.norm(Y[~np.isnan(Y)])*1e-4,
                                                   maxiter=100, verbose=10,
                                                   prox_rank=lambda Z,thresh: th.singular_value_thresholding(Z,thresh,thresholding=lambda s,thresh: th.smoothly_clipped_absolute_deviation(s,thresh,a)))
#Yest, x, s, x, it = sps.low_rank_matrix_completion_ind(Y, tol=linalg.norm(Y[~np.isnan(Y)])*1e-3, maxiter=100, rtol=1e-4, verbose=10,
#                                        prox_rank=lambda Z,thresh: th.singular_value_thresholding(Z,thresh,thresholding=lambda s,thresh: th.smoothly_clipped_absolute_deviation(s,thresh,a)))
print('%d steps, rel. error = %.2e' % (it, linalg.norm((Yest-Y)[~np.isnan(Y)])/linalg.norm(Y[~np.isnan(Y)])))
print('Yest = ')
print(Yest)
print('sv = ')
print(s)


print('====(Deomo 2: low-rank matrix completion)====')
dtype = np.float32

rng = np.random.RandomState(int(time()))
m, n, rnk = 3000, 500, 10
Ut = rng.randn(m, rnk).astype(dtype)  # random design
Vt = rng.randn(rnk, n).astype(dtype)  # random design
Y = Ut.dot(Vt) / sqrt(m)
# Y = np.where(Y < 0.2, np.nan, Y)  # Set all data smaller than 0.2 to NaN

R = rng.rand(m,n) < 0.15
#Y = Y / np.linalg.norm(Y)
print('%.2f %% entries are given ..' % (100.0 * sum(R.ravel())/R.size))

t0 = time()
Yest, Uest, sest, Vest, it = sps.low_rank_matrix_completion(Y, R=R, l=1., rtol=1e-4, tol=1e-3*linalg.norm(Y[R]), 
                                                              maxiter=10, verbose=10, nesterovs_momentum=True, 
                                                              prox_rank=lambda Z,thresh: th.singular_value_thresholding(Z,thresh,thresholding=lambda s,thresh: th.smoothly_clipped_absolute_deviation(s,thresh,a)))
#Yest, Uest, sest, Vest, it = sps.low_rank_matrix_completion_ind(Y, R=R, tol=1e-3*linalg.norm(Y[R]), maxiter=100, nesterovs_momentum=False, restart_every=4, verbose=10, rtol=1e-2,
#                                               prox_rank=lambda Z,thresh: th.singular_value_thresholding(Z,thresh,thresholding=lambda s,thresh: th.smoothly_clipped_absolute_deviation(s,thresh,a)))
print('done in %.2fs with %d steps' % (time() - t0, it))
print('rel. error = %.2e' % (linalg.norm(Yest-Y)/linalg.norm(Y)))
#print(Yest)
print('nonzero sv = ')
print(sest[np.nonzero(sest)])




print('====(Deomo3: SPCP)====')

ng, dim = 3, 1000
dtype = np.float32
m, n, rnk = ng*dim, 300, 10
print('D.shape = (%d, %d)' % (m, n))
Ut = rng.randn(m, rnk).astype(dtype)  # random design
Vt = rng.randn(rnk, n).astype(dtype)  # random design
L = Ut.dot(Vt) / sqrt(m)    # rank 10 matrix
print('singular values of L =')
print(linalg.svd(L, compute_uv=False)[:10])
print('mean(abs(L)) = %.2e' % (np.mean(np.abs(L))))

S = np.zeros((dim, n),dtype=dtype)
support = rng.choice(dim*n, int(dim*n*0.15), replace=False)
S.ravel()[support] = 10.*rng.randn(support.size) / sqrt(dim)
S = np.tile(S,(ng,1)) # group sparse matrix
print('mean(abs(S)) = %.2e, %d nonzeros in S (%2.1f%%)' % (np.mean(np.abs(S)),ng*support.size,100.*ng*support.size/(m*n)))

D = L + S
noisep = 0.03
E = rng.randn(m,n).astype(dtype) / sqrt(m*n) * linalg.norm(D) * noisep
D += E
print('||D||_F = %.2e, ||D-(L+S)||_F = %.2e' % (linalg.norm(D), linalg.norm(E)))



#from numba import jit, prange
#@jit(nopython=True,nogil=True,parallel=True)
def sf_soft(q,l,dim,n):
    numelQ = dim*n
    #sf = np.zeros(3,dtype=q.dtype)
    p = np.zeros_like(q)
#    for i in prange(numelQ):
    for i in range(numelQ):
        a = sqrt(q[i]*q[i]+q[i+numelQ]*q[i+numelQ]+q[i+2*numelQ]*q[i+2*numelQ]) + 2e-16
        #a = sqrt(sum(sf*sf))
        #if a != 0.:
        #    a = max(a-l, 0.) / a
        #a = sqrt(sum(sf*sf)) + 2e-16
        a = max(a-l, 0.) / a
        p[i] = a * q[i]
        p[i+numelQ] = a * q[i+numelQ]
        p[i+2*numelQ] = a * q[i+2*numelQ]

    return p


#import spmlib.proxop as prox
import spmlib.thresholding._jit as th_jit
#import spmlib.thresholding as th_jit
t0 = time()
Lest, Sest, _, sest, _, it = sps.stable_principal_component_pursuit(D, tol=linalg.norm(E), ls=None, rtol=1e-2, rho=1., maxiter=100, nesterovs_momentum=True,
                                                                verbose=10)
#                                                                prox_L=lambda Q,l: th.singular_value_thresholding(Q,2*l,thresholding=th_jit.smoothly_clipped_absolute_deviation),
#                                                                prox_L=lambda Q,l: th.svt_svds(Q, l, k=13, tol=1e-1, thresholding=th_jit.smoothly_clipped_absolute_deviation),
#                                                                prox_LS=lambda q,r,c: prox.ind_l2ball(q,3.*linalg.norm(D),c),
#                                                                prox_S=lambda q,l: sf_soft(q,l,dim,n))
#                                                                prox_S=lambda q,l: th_jit.group_scad(q,2*l, np.tile(np.reshape(np.arange(dim*n), (dim,n)), (ng,1)), normalize=False))
#                                                                prox_S=lambda q,l: th_jit.group_scad(q,l*np.sqrt(dim*m,dtype=np.float32), np.tile(np.reshape(np.arange(dim*n), (dim,n)), (ng,1)), normalize=True))
#                                                                prox_S=lambda q,l: th_jit.smoothly_clipped_absolute_deviation(q,l))

np.set_printoptions(suppress=True)
print('done in %.2fs with %d steps' % (time() - t0, it))
print('rel. error in L: %.2e,  S: %.2e' % (linalg.norm(Lest-L)/linalg.norm(L), linalg.norm(Sest-S)/linalg.norm(S)))
print('%d nonzeros in S estimated, mean(abs(S)) = %.2e' % (sum(np.abs(Sest.ravel()) > 1e-2), np.mean(np.abs(Sest))))
#print('nonzero sv = ', sest[np.nonzero(sest)])
print('nonzero sv = ')
print(sest[sest > np.spacing(np.float32(1.0))])

from sklearn import metrics
print(metrics.classification_report(np.abs(S.ravel()) > 1e-2, np.abs(Sest.ravel()) > 1e-2))
print(metrics.confusion_matrix(np.abs(S.ravel()) > 1e-2, np.abs(Sest.ravel()) > 1e-2))




print('====(Deomo4: SPCP for incomplete data)====')

R = rng.rand(m,n) < 0.90
print('%.2f %% entries are given ..' % (100.0 * sum(R.ravel())/R.size))
Lest, Sest, _, sest, _, it = sps.stable_principal_component_pursuit(D, R=R, tol=linalg.norm(E), ls=None, rtol=1e-2, rho=1., maxiter=100, nesterovs_momentum=True,
                                                                verbose=10)
#                                                                prox_L=lambda Q,l: th.singular_value_thresholding(Q,2*l,thresholding=th_jit.smoothly_clipped_absolute_deviation),
#                                                                prox_L=lambda Q,l: th.svt_svds(Q, l, k=13, tol=1e-1, thresholding=th_jit.smoothly_clipped_absolute_deviation),
#                                                                prox_LS=lambda q,r,c: prox.ind_l2ball(q,3.*linalg.norm(D),c),
#                                                                prox_S=lambda q,l: sf_soft(q,l,dim,n))
#                                                                prox_S=lambda q,l: th_jit.group_scad(q,2*l, np.tile(np.reshape(np.arange(dim*n), (dim,n)), (ng,1)), normalize=False))
#                                                                prox_S=lambda q,l: th_jit.group_scad(q,l*np.sqrt(dim*m,dtype=np.float32), np.tile(np.reshape(np.arange(dim*n), (dim,n)), (ng,1)), normalize=True))
#                                                                prox_S=lambda q,l: th_jit.smoothly_clipped_absolute_deviation(q,l))

np.set_printoptions(suppress=True)
print('done in %.2fs with %d steps' % (time() - t0, it))
print('rel. error in L: %.2e,  S: %.2e' % (linalg.norm(Lest-L)/linalg.norm(L), linalg.norm(Sest-S)/linalg.norm(S)))
print('%d nonzeros in S estimated, mean(abs(S)) = %.2e' % (sum(np.abs(Sest.ravel()) > 1e-2), np.mean(np.abs(Sest))))
#print('nonzero sv = ', sest[np.nonzero(sest)])
print('nonzero sv = ')
print(sest[sest > np.spacing(np.float32(1.0))])

from sklearn import metrics
print(metrics.classification_report(np.abs(S[R].ravel()) > 1e-2, np.abs(Sest[R].ravel()) > 1e-2))
print(metrics.confusion_matrix(np.abs(S[R].ravel()) > 1e-2, np.abs(Sest[R].ravel()) > 1e-2))



