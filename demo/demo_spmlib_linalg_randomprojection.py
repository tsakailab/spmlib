..# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:16:41 2017

@author: tsakai
"""

import numpy as np

from time import time

from spmlib.linalg import RandomProjection


import timeit
rng = np.random.RandomState()
d, k = 2**15, 512
rp = RandomProjection(d, k)
x = rng.randn(d).astype(np.float32)
y = rp.erp(x)
normx = np.linalg.norm(x)
normy = np.linalg.norm(y)

print("%1.4e = norm(x)" % normx)
print("%1.4e = norm(erp(x, %d))" % (normy, y.shape[0]))
print("Relative error = %1.1f [%%]" % ((normy-normx)/normx*100))

print("FFT length = %d" % rp.key[0].shape[0])
print("%f [s] per a random projection from %d dim. to %d dim." % (timeit.timeit("y = rp.erp(x)", 
      setup="from __main__ import rp, x, y", 
      number=50)/50., x.shape[0], y.shape[0]))

RR = rng.randn(k,d).astype(np.float32) / np.sqrt(k,dtype=np.float32)
print("%f [s] per a random projection from %d dim. to %d dim. by matrix" % (timeit.timeit("y = RR.dot(x)", 
      setup="from __main__ import RR, x, y", 
      number=50)/50., x.shape[0], y.shape[0]))

    
print("Making histogram of norm errors .. ")
n = 500
relerr = np.zeros(n)
t0 = time()
for i in range(n):
    #x = rng.randn(dx)# +0.j#rng.randn(dx)*1.j
    x = rng.randn(d).astype(np.float32)
    p = rp.erp(x, k)
    #p = RR.dot(x)
    normx, normp = np.linalg.norm(x), np.linalg.norm(p)
    relerr[i] = (normp - normx) / normx

print('done in %.2fs.' % (time() - t0))


import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hist(relerr, bins=30, color='blue', alpha=0.5)
ax.set_xlim(-0.3, 0.3)
ax.set_title('Relative error distribution')
ax.set_xlabel('Relative error')
ax.set_ylabel('Frequency')
fig.show()

