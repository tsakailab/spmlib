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
Y = np.array([[     5,      0,      5,      1],
              [np.nan,      0,      4, np.nan],
              [     0,      5,      0,      4],
              [     4, np.nan, np.nan,      0],
              [     0,      5,      0, np.nan]])
#Y = np.array([[     5,      0,      5,      1],
#              [     4,      0,      4,      1],
#              [     0,      5,      0,      4],
#              [     4,      0,      5,      0],
#              [     0,      5,      0,      4]])
print('Y    = ', Y)
#result = LowRankMatrixCompletion(Y, l=1,tol=linalg.norm(Y[~np.isnan(Y)])*1e-6, maxit=100)
#print('Yest = ', result.Yest)
#print('sv = ', result.s)
a = 3.7
Yest, U, s = sps.low_rank_matrix_completion(Y, l=1, tol=linalg.norm(Y[~np.isnan(Y)])*1e-6, maxiter=100,
                                        prox_rank=lambda Z,thresh: th.soft_svd(Z,thresh,thresholding=lambda s,thresh: th.smoothly_clipped_absolute_deviation(s,thresh,a)))[0:3]
print('rel. error = %.2e' % (linalg.norm((Yest-Y)[~np.isnan(Y)])/linalg.norm(Y[~np.isnan(Y)])))
print('Yest = ', Yest)
print('sv = ', s)



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
Yest, Uest, sest, Vest, it = sps.low_rank_matrix_completion(Y, R=R, l=1, tol=1e-4*linalg.norm(Y[R]), maxiter=100,nesterovs_momentum=True, verbose=10,
                                               prox_rank=lambda Z,thresh: th.soft_svd(Z,thresh,thresholding=lambda s,thresh: th.smoothly_clipped_absolute_deviation(s,thresh,a)))[:5]
print('done in %.2fs with %d steps' % (time() - t0, it))
print('rel. error = %.2e' % (linalg.norm(Yest-Y)/linalg.norm(Y)))
#print(Yest)
print('nonzero sv = ', sest[np.nonzero(sest)])
