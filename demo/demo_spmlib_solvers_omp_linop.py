# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:16:41 2017

@author: tsakai
"""

from math import sqrt
import numpy as np
from scipy import linalg
import scipy.sparse.linalg as splinalg

from time import time
import matplotlib.pyplot as plt

from spmlib import solver as sps
from spmlib.linalg import RandomProjection


#x0 = rng.randn(3) + rng.randn(3)*1j
##print np.real(x0.conj().T.dot(x0))
#x = sgn(x0)
#print linalg.norm(x,1)


#%% Demo: Sparse solvers
rng = np.random.RandomState(int(time()))
#m, n = 512, 2048
#m, n = 1024, 8192
#m, n = 2048, 8192*32
m, n = 2048, 100000
dtype = np.float32


# use a random matrix as a basis (design matrix)
A = RandomProjection(n,m).aslinearoperator()
#A = rng.randn(m, n).astype(dtype) / sqrt(m)  # random design


# generate a k-sparse Gaussian signal vector
k = 200
stdx = 1.
snr = 20.

x_true = np.zeros(n, dtype=dtype)
T = rng.choice(n,k,replace=False)
x_true[T] = rng.randn(k).astype(dtype) * stdx
#x_true = rng.randn(n)
#x_true[abs(x_true) < 2.5] = 0  ## sparsify

# make the query vector
b = splinalg.aslinearoperator(A).matvec(x_true).astype(dtype)

# add noise
normb = linalg.norm(b)
noise = rng.randn(m).astype(dtype)
noise = noise / linalg.norm(noise) * normb / snr
tol = linalg.norm(noise)
b = b + noise



plt.close('all')




import numpy as np
from scipy import linalg


# todo: return nonzero only, then OMP makes it spvec


# OMP
print("Running OMP..")
t0 = time()
#result_OMP = sps.orthogonal_matching_pursuit(A, b, tol=tol)
result_OMP = sps.orthogonal_matching_pursuit_using_linearoperator(splinalg.aslinearoperator(A), b, tol=tol*0.8)
x_est = result_OMP[0]
print('done in %.2fs.' % (time() - t0))
print('%d iterations, supprt of %d nonzeros = ' % (result_OMP[2], np.count_nonzero(x_est)))
print(np.nonzero(x_est)[0])
print('rel. error = %.2e' % (linalg.norm(x_est-x_true)/linalg.norm(x_true)))

plt.figure()
#plt.stem(x_true, markerfmt='g.', label='True')
plt.plot(np.arange(n), x_true, 'g.', markersize=8, mec='green', label='True')
plt.plot(np.arange(n), x_est, 'ro', mfc = 'None', markersize=8, mec='red', label='Estimated')
plt.legend(loc='upper right', shadow=False)
plt.show()




