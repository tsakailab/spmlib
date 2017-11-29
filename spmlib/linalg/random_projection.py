# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 23:08:00 2017

@author: tsakai
"""

import numpy as np
from scipy import fftpack
from scipy.sparse.linalg import LinearOperator
from time import time


def _generate_cipher_key(d, seed=None, dtype=np.float32):
    if seed is None:
        seed = int(time())
    rng = np.random.RandomState(seed)
    key0 = rng.randn(fftpack.next_fast_len(d)).astype(dtype)   # w
    if np.issubdtype(dtype, np.complexfloating):
        key0 = (key0 + rng.randn(key0.shape[0]).astype(dtype) * 1j) * (1./np.sqrt(2.))
    key0 = fftpack.fft(key0)   # w
    key1 = rng.randint(2, size=key0.shape[0]).astype(np.int8)               # s
    key1[key1 == 0] = -1
    return (key0, key1)


def _efficient_random_projection(x, k, key, dtype):
    dx = x.shape[0]
    maxdim = key[0].shape[0]
    if dx > maxdim or k > maxdim:
        raise ValueError("Wrong dimensionality")
    sx = np.zeros_like(key[0])
    sx[:dx] = key[1][:dx] * x  # cast x to key[0].dtype
    sx = fftpack.fft(sx)
    sx = fftpack.ifft(key[0] * sx.conjugate())[:k] * (np.sqrt(1.0/k, dtype=dtype))
    if np.isrealobj(x) and not np.issubdtype(dtype, np.complexfloating):
        return sx.real
    else:
        return sx


def _efficient_random_projection_adjoint(y, d, key, dtype):
    dy = y.shape[0]
    maxdim = key[0].shape[0]
    if dy > maxdim or d > maxdim:
        raise ValueError("Wrong dimensionality")
    sy = np.zeros_like(key[0])
    sy[:dy] = y * np.sqrt(1.0/dy)
    sy = fftpack.fft(sy)
    # sy = fftpack.ifft(key[0] * sy.conjugate())[:d] * key[1][:d] # transposition
    sy = fftpack.ifft(key[0] * sy.conjugate())[:d].conjugate() * key[1][:d]
    if np.isrealobj(y) and not np.issubdtype(dtype, np.complexfloating):
        return sy.real
    else:
        return sy


class RandomProjection:
    """
    Random projection from ambient_dim-D space onto target_dim-D space.
        
    Parameters
    ----------
    ambient_dim : int
        Ambient dimensionality.
    target_dim : int, optional, default None
        Dimensionality of target space. Must be no greater than `ambient_dim`.
        If given, random projection returns (`target_dim`,) array by default.
    seed : int, optional, default None
        seed for RandomState
    dtype : numpy.dtype object, optional, default numpy.float32
        Data type object of random numbers
        
    References
    ----------
    Tomoya Sakai and Atsushi Imiya
        "Practical algorithms of spectral clustering: Toward large-scale vision-based motion analysis"
        Machine Learning for Vision-Based Motion Analysis
        Springer London, 2011. 3-26.
    Tomoya Sakai and Daisuke Miyata
        "Learning high-dimensional nonlinear mapping via compressed sensing"
        Acoustics, Speech and Signal Processing (ICASSP),
        2014 IEEE International Conference on. IEEE, 2014.
    """
    def __init__(self, ambient_dim, target_dim=None, seed=None, dtype=np.float32):
        self.dtype = dtype
        self.ambient_dim = ambient_dim
        if target_dim is None:
            self.target_dim = ambient_dim
        else:
            self.target_dim = target_dim
        self.key = _generate_cipher_key(ambient_dim, seed, dtype)


    def erp(self, u, target_dim=None, adjoint=False):
        """
        Efficient random projection
        
        Parameters
        ----------
        u : array_like
            Random projection of u is performed.
            u.shape[0] must be no greater than `ambient_dim`.
        target_dim : int, optional, default None
            Dimensionality of target space. Must be no greater than `ambient_dim`.
            If given, random projection returns (`target_dim`,) array.
        adjoint : bool, optional, default False
            If set to true, performs random projection by
            conjugate transpose of random matrix,
            and returns (`ambient_dim`,) array.
        Returns
        -------
        v : array_like, shape (`target_dim`,) or (`ambient_dim`,)
            Random projection of `u`.
            v.shape is (`ambient_dim`,) if `adjoint` is true.
        """
        if adjoint:
            return _efficient_random_projection_adjoint(u, self.ambient_dim, self.key, self.dtype)
        if target_dim is None:
            target_dim = self.target_dim
        return _efficient_random_projection(u, target_dim, self.key, self.dtype)


    def aslinearoperator(self, target_dim=None):
        """
        Return random matrix as scipy linear operator
        
        Parameters
        ----------
        target_dim : int, optional, default None
            Dimensionality of target space. Must be no greater than `ambient_dim`.
            If given, random projection returns (`target_dim`,) array.
        Returns
        -------
        scipy linear operator with shape, matvec and rmatvec
        """
        if target_dim is None:
            target_dim = self.target_dim
        return LinearOperator(tuple([target_dim, self.ambient_dim]),
                              matvec=lambda u: _efficient_random_projection(u, target_dim, self.key, self.dtype),
                              rmatvec=lambda v: _efficient_random_projection_adjoint(v, self.ambient_dim, self.key, self.dtype),
                              dtype=self.dtype)


if __name__ == '__main__':
    import timeit
    rng = np.random.RandomState()
    d, k = 123456, 256
    dx = d
    #x = rng.randn(dx) #+ 0.j #+ rng.randn(dx)*1.j
    x = rng.randn(dx)
    rp = RandomProjection(d)

    y = rp.erp(x,k)
    normx = np.linalg.norm(x)
    normy = np.linalg.norm(y)

    print("%1.4e = norm(x)" % normx)
    print("%1.4e = norm(erp(x, %d))" % (normy, y.shape[0]))
    print("Relative error = %1.1f [%%]" % ((normy-normx)/normx*100))

    print("FFT length = %d" % rp.key[0].shape[0])
    print("%f [s] per a random projection from %d dim. to %d dim." % (timeit.timeit("y = rp.erp(x, y.shape[0])", 
                          setup="from __main__ import rp, x, y", 
                          number=100)/100., x.shape[0], y.shape[0]))

    
    print("Making histogram of norm errors .. ")
    n = 500
    relerr = np.zeros(n)
    t0 = time()
    for i in range(n):
        #x = rng.randn(dx)# +0.j#rng.randn(dx)*1.j
        x = rng.randn(dx).astype(np.float32)
        p = rp.erp(x, k)
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


    dd, kk = 8, 3
    rpkd = RandomProjection(dd, kk, seed=80,dtype=np.float64)
    Rkd = np.zeros((kk,dd),dtype=rpkd.dtype)

    print("Show random matrix: ")
    for jj in range(dd):
        xd = np.zeros(dd)
        xd[jj] = 1.0
        Rkd[:,jj] = rpkd.erp(xd)
    np.set_printoptions(precision=2)
    print(Rkd)

    xdd = rng.randn(dd) # + 0.j
    print(xdd)
    ykk = Rkd.dot(xdd)
    print(ykk)
    print(rpkd.erp(xdd))
    print(Rkd.T.dot(ykk))
    print(rpkd.erp(ykk,adjoint=True))

    
    print("Show random matrix via linear operator: ")
    erp = rpkd.aslinearoperator()
    for jj in range(dd):
        xd = np.zeros(dd)
        xd[jj] = 1.0
        Rkd[:,jj] = erp.matvec(xd)

    np.set_printoptions(precision=2)
    print(Rkd)
    print(xdd)
    ykk = Rkd.dot(xdd)
    print(ykk)
    print(erp.matvec(xdd))
    print(Rkd.conj().T.dot(ykk))
    print(erp.rmatvec(ykk))
    