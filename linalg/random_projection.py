# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 08:42:11 2017

@author: tsakai
"""

import numpy as np
from scipy import fftpack
from time import time


def _generate_cipher_key(d, seed=None, dtype=np.float64):
    if seed is None:
        seed = int(time())
    rng = np.random.RandomState(seed)
    key0 = fftpack.fft(rng.randn(fftpack.next_fast_len(d)).astype(dtype))   # w
    key1 = rng.randint(2, size=key0.shape[0]).astype(np.int8)               # s
    key1[key1 == 0] = -1
    return (key0, key1)


def _efficient_random_projection(x, k, key):
    d = x.shape[0]
    maxdim = key[0].shape[0]
    if d > maxdim or d < k:
        raise ValueError("Wrong dimensionality")
    sx = np.zeros_like(key[0])
    sx[:d] = key[1][:d] * x
    sx = fftpack.fft(sx)
    if np.isrealobj(x):
        return np.real(sx[:k] * key[0][:k]) * np.sqrt(2.0/(maxdim*k))
    else:
        return fftpack.ifft(sx * key[0])[:k] * (1.0 / np.sqrt(k))
        


class RandomProjection:
    def __init__(self, d, target_dim=1000, seed=None, dtype=np.float64):
        self.key = _generate_cipher_key(d, seed, dtype)
        self.target_dim = target_dim
        
    def erp(self, x, k=None):
        if k is None:
            k = self.target_dim
        return _efficient_random_projection(x, k, self.key)
#    def erpT():
#        return 1



if __name__ == '__main__':
    import timeit
    rng = np.random.RandomState()
    d, k = 123456, 256
    x = rng.randn(d) #+ rng.randn(d)*1.j
    rp = RandomProjection(d, k)

    y = rp.erp(x)
    normx = np.linalg.norm(x)
    normy = np.linalg.norm(y)

    print("%1.4e = norm(x)" % normx)
    print("%1.4e = norm(erp(x, %d))" % (normy, y.shape[0]))
    print("Relative error = %1.1f [%%]" % ((normy-normx)/normx*100))

    print("FFT length = %d" % rp.key[0].shape[0])
    print("%f [s] per a random projection from %d dim. to %d dim." % (timeit.timeit("y = rp.erp(x, y.shape[0])", 
                          setup="from __main__ import rp, x, y", 
                          number=1000)/1000., d, y.shape[0]))
    