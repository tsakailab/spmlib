# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 23:00:05 2018

@author: tsakai
"""

from math import copysign
import numpy as np


#%% Numba, nopython with float32, nogil and parallel
from numba import jit, prange


#%% soft thresholding function compatible with complex values
#@jit(nopython=True,nogil=True,cache=True)
@jit('f4[:](f4[:],f4[:])',nopython=True,nogil=True,cache=True)
def _soft(z, th):
    return np.sign(z) * np.maximum(np.abs(z) - th, 0)


@jit('f4[:](f4[:],f4)',nopython=True,nogil=True,parallel=True)
def soft(z, th):
    s = z.copy()
    vs = s.ravel()
    vz = z.ravel()
    nz = vz.size
    for i in prange(nz):
        vs[i] = copysign(1,vz[i])*max(abs(vz[i])-th, 0)
    return vs


#%% smoothly clipped absolute deviation (SCAD) [Fan&Li, 01]
# vectoral version
@jit('f4[:](f4[:],f4[:],f4)',nopython=True,nogil=True,cache=True)
def _smoothly_clipped_absolute_deviation(vz, vth, a):
    scad = vz.copy()
    absz = np.abs(vz)
    idx = absz <= 2*vth
    scad[idx] = _soft(vz[idx], vth[idx])
    idx = np.logical_and(absz > 2*vth, absz <= a*vth)
    scad[idx] = ((a - 1) * vz[idx] - np.sign(vz[idx]) * a * vth[idx]) / (a - 2)
    return scad

# single threshold version
#@jit('f4[:](f4[:],f4,optional(f4))',cache=True)
def smoothly_clipped_absolute_deviation(z, th, a=3.7):
    return __smoothly_clipped_absolute_deviation(z.ravel(),th,a)

#%% SCAD, jit faster version
@jit('f4[:](f4[:],f4,f4)',nopython=True,nogil=True,parallel=True)
def __smoothly_clipped_absolute_deviation(vz, th, a):
    sc = vz.copy()
    nz = vz.size
    for i in prange(nz):
        absz = abs(vz[i])
        if absz <= 2*th:
            sc[i] = copysign(1,vz[i])*max(absz-th, 0)
        elif absz <= a*th:
            sc[i] = ((a - 1) * vz[i] - copysign(1,vz[i]) * a * th) / (a - 2)
    return sc



@jit(parallel=True)
def group_soft(z, th, garray, normalize=True):
    """
    Group-wise soft threshlding
        soft(||z(g)||_2, `th`) / ||z(g)||_2 * z(g) for g in `groups`, 
        i.e., `l2_soft(z[g], th)`.
    Here, `groups` is a list of index lists.
    The l2 norm of each subvector z(g) shrinks by `th` without changing its direction.
    The groups (entries of z) must not overlap.
    If you have the list `groups`, make `garray` = index_groups_to_array(`groups`).

    Note: this function is the proximity operator.
        arg min_x sum_{g in `groups`}( th*||x(g)||_2 + 0.5*||x(g)-z(g)||_2^2 )

    Parameters
    ----------
    z : array_like, shape (`m`,)
        `m`-dimensional vector to be thresholded group-wise.
    th : scalar,
        Threshold.
    garray : array_like, shape(`m`,)
        `m`-dimensional vector of group IDs. There are max(garray)+1 groups.
    normaize : bool, optional, default True
        If True, the threshold th is scaled by sqrt(number of group members)/sqrt(m) for each subvector z(g).

    Returns
    -------
    v : ndarray
        The thresholded vector.
    """

    m = z.size
    ng = np.max(garray)+1   # number of groups(subvectors)
    normsz = np.zeros(ng, dtype=np.float32)   # squared norms of subvectors
    ms = np.zeros(ng, dtype=np.float32)       # numbers of group members (dims. of subvectors)
    vecz = z.ravel()
    for i in prange(m):
        g = garray.ravel()[i]
        normsz[g] += vecz[i]*vecz[i]
        ms[g] += 1

    # compute scales
    normzsn = normsz.copy()
    normzsn[normsz == 0.0] = 1.0    # to avoid zero-div
    normzsn = np.sqrt(normzsn)
    # thresholding
    if normalize:
        #az = np.max(normzsn - np.sqrt(ms) * th / np.sqrt(m), 0.) / normzsn
        az = _soft(normzsn, np.sqrt(ms) * (th/np.sqrt(m,dtype=np.float32)) ) / normzsn
    else:
        #az = np.max(normzsn - th, 0.) / normzsn
        az = soft(normzsn, th) / normzsn

    # scaling
    v = np.zeros_like(z)
    vecv = v.ravel()
    vecg = garray.ravel()
    for i in prange(m):
        vecv[i] = vecz[i] * az[vecg[i]]

    return v



@jit(parallel=True)
def group_scad(z, th, garray, normalize=True, a=3.7):
    """
    Group-wise soft threshlding
        soft(||z(g)||_2, `th`) / ||z(g)||_2 * z(g) for g in `groups`, 
        i.e., `l2_soft(z[g], th)` with SCAD thresholding.
    Here, `groups` is a list of index lists.
    The l2 norm of each subvector z(g) shrinks by `th` without changing its direction.
    The groups (entries of z) must not overlap.
    If you have the list `groups`, make `garray` = index_groups_to_array(`groups`).

    Note: this function is the proximity operator.
        arg min_x sum_{g in `groups`}( th*||x(g)||_2 + 0.5*||x(g)-z(g)||_2^2 )

    Parameters
    ----------
    z : array_like, shape (`m`,)
        `m`-dimensional vector to be thresholded group-wise.
    th : scalar,
        Threshold.
    garray : array_like, shape(`m`,)
        `m`-dimensional vector of group IDs. There are max(garray)+1 groups.
    normaize : bool, optional, default True
        If True, the threshold th is scaled by sqrt(number of group members)/sqrt(m) for each subvector z(g).

    Returns
    -------
    v : ndarray
        The thresholded vector.
    """

    m = z.size
    ng = np.max(garray)+1   # number of groups(subvectors)
    normsz = np.zeros(ng, dtype=np.float32)   # squared norms of subvectors
    ms = np.zeros(ng, dtype=np.float32)       # numbers of group members (dims. of subvectors)
    vecz = z.ravel()
    for i in prange(m):
        g = garray.ravel()[i]
        normsz[g] += vecz[i]*vecz[i]
        ms[g] += 1

    # compute scales
    normzsn = normsz.copy()
    normzsn[normsz == 0.0] = 1.0    # to avoid zero-div
    normzsn = np.sqrt(normzsn)
    # thresholding
    if normalize:
        #az = np.max(normzsn - np.sqrt(ms) * th / np.sqrt(m), 0.) / normzsn
        az = _smoothly_clipped_absolute_deviation(normzsn, np.sqrt(ms) * (th / np.sqrt(m,dtype=np.float32)), a) / normzsn
    else:
        #az = np.max(normzsn - th, 0.) / normzsn
        az = smoothly_clipped_absolute_deviation(normzsn, th, a) / normzsn

    # scaling
    v = np.zeros_like(z)
    vecv = v.ravel()
    vecg = garray.ravel()
    for i in prange(m):
        vecv[i] = vecz[i] * az[vecg[i]]

    return v





if __name__ == '__main__':
    from time import time
    rng = np.random.RandomState()
    x = rng.randn(10000000).astype(np.float32)

    t0 = time()
#    y = soft(x,0.5)
    y = smoothly_clipped_absolute_deviation(x,0.5)
    print('done in %.6fs.' % (time() - t0))

