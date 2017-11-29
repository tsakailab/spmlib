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

dtype = np.float32

rng = np.random.RandomState(int(time()))
m, n, rnk = 3000, 300, 10
Ut = rng.randn(m, rnk).astype(dtype)  # random design
Vt = rng.randn(rnk, n).astype(dtype)  # random design
Y = Ut.dot(Vt) / sqrt(m)


print('====(Deomo: incremental SVD)====')

t0 = time()
U, s, V = linalg.svd(Y, full_matrices=False)
print('done in %.2fs by batch SVD.' % (time() - t0))
print('rank = %d, nonzero sv = ' % (s.size), s[s > 16.*np.spacing(np.float32(1.0))])
    
t0 = time()
U_inc, sv_inc = np.empty([0,0]), np.empty(0)
count = 0
for j in range(n):
    U_inc, sv_inc = sps.column_incremental_SVD(Y[:,j:j+1], U_inc, sv_inc, max_rank=20,
                                          orth_eps=linalg.norm(Y[:,j:j+1])*1e-12, orthogonalize_basis=True)
    count = count + 1

print('done in %.2fs (%d iter.) by incremental SVD.' % (time() - t0, count))
print('rank = %d, nonzero sv = ' % (sv_inc.size), sv_inc[sv_inc > np.spacing(np.float32(1.0))])


print('====(Deomo: IncSPCP)====')
t0 = time()
U_spcp, sv_spcp = np.empty([0,0]), np.empty(0)
R = rng.rand(m,n-rnk) < 0.1
print('%d (%.2f %%) entries are corrupted.' % (sum(R.ravel()), 100.0 * sum(R.ravel())/Y.size))
S = np.zeros_like(Y)
S[:,rnk:][R] = rng.randn(sum(R.ravel())).astype(dtype)
Yc = Y + S
count = 0
Lest = np.zeros_like(Y)
Sest = np.zeros_like(Y)
for j in range(n):
    Lest[:,j], Sest[:,j], U_spcp, sv_spcp, c = sps.OnlSPCP_SCAD(Yc[:,j], U_spcp, sv_spcp,
                                                   ls=0.05, maxiter=100, switch_to_scad_after=40,
                                                   rtol=1e-3, rdelta=1e-6, max_rank=20, min_sv=-0.1, update_basis=True)
#    l, Sest[:,j], U_spcp, sv_spcp, c = sps.column_incremental_stable_principal_component_pursuit(
#                                Yc[:,j], U_spcp, sv_spcp, ls=0.1, update_basis=True, maxiter=100,
#                                max_rank=20, orth_eps=linalg.norm(Yc[:,j])*1e-12, orthogonalize_basis=True)
#    print c
    count = count + 1

print('done in %.2fs (%d iter.) by OnlSPCP.' % (time() - t0, count))
print('rank = %d, nonzero sv = ' % (sv_spcp.size), sv_spcp[sv_spcp > np.spacing(np.float32(1.0))])

print('rel. error = %.2e' % (linalg.norm(Lest-Y)/linalg.norm(Y)))
#print('rel. error = %.2e' % (linalg.norm(Lest[:,4*rnk:]-Y[:,4*rnk:])/linalg.norm(Y[:,rnk:])))
print('%d nonzeros' % (sum(np.abs(Sest.ravel()) > 1e-2)))
