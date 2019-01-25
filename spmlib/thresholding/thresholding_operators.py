# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 08:42:11 2017

@author: tsakai
"""

import numpy as np
from scipy import linalg
from scipy import sparse as sp

    
#%% sign function compatible with complex values
def sgn(z):
    if np.all(np.isreal(z)):
        return np.sign(z)
    return np.divide(z, np.abs(z))


#%% soft thresholding function compatible with complex values
def soft(z, th):
    return sgn(z) * np.maximum(np.abs(z) - th, 0)


#%% smoothly clipped absolute deviation (SCAD) [Fan&Li, 01]
def smoothly_clipped_absolute_deviation(z, th, a=3.7):
    scad = z.copy()
    absz = np.abs(z)
    idx = absz <= 2*th
    scad[idx] = soft(z[idx], th)
    idx = np.logical_and(absz > 2*th, absz <= a*th)
    scad[idx] = ((a - 1) * z[idx] - sgn(z[idx]) * a * th) / (a - 2)
    return scad

def __smoothly_clipped_absolute_deviation(vz, vth, a=3.7):
    scad = vz.copy()
    absz = np.abs(vz)
    idx = absz <= 2*vth
    scad[idx] = soft(vz[idx], vth[idx])
    idx = np.logical_and(absz > 2*vth, absz <= a*vth)
    scad[idx] = ((a - 1) * vz[idx] - sgn(vz[idx]) * a * vth[idx]) / (a - 2)
    return scad


#%% hard thresholding
def hard(z, th):
    return z * (np.abs(z)>th)


#%% singular value thresholding
def singular_value_thresholding(Z, th, thresholding=lambda z,th: soft(z, th)):
    U, sv, Vh = linalg.svd(Z,full_matrices=False)
    sv = thresholding(sv, th)
    r = np.count_nonzero(sv)
    U = U[:,:r]
    sv = sv[:r]
    Vh = Vh[:r,:]
    return U.dot(sp.diags(sv).dot(Vh)), U, sv, Vh


#%% singular value thresholding (svds version)
import scipy.sparse.linalg as splinalg
def svt_svds(Z, th, thresholding=lambda z,th: soft(z, th), k=None, tol=0):
    if k is None:
        k = min(min(Z.shape), 30)
    U, sv, Vh = splinalg.svds(splinalg.aslinearoperator(Z), k=k, tol=tol)
    sv = thresholding(sv, th)
    return U.dot(sp.diags(sv).dot(Vh)), U, sv, Vh

#svt = singular_value_thresholding
#scad = smoothly_clipped_absolute_deviation

    
def l2_soft(z, th, c=None, thresholding=soft):
    """
    c + soft(||z-c||_2,th)/||z-c||_2 * (z-c)
    """
    if c is None:
        normz = linalg.norm(z)
        if normz == 0.0:
            return np.zeros_like(z)
        return (thresholding(normz, th) / normz) * z
    else:
        zc = z - c
        normzc = linalg.norm(zc)
        if normzc == 0.0:
            return np.zeros_like(z)
        return c + (thresholding(normzc, th) / normzc) * zc


def group_soft(z, th, garray, normalize=True, thresholding=soft):
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
    normsz = np.zeros(ng, dtype=z.dtype)   # squared norms of subvectors
    ms = np.zeros(ng, dtype=z.dtype)       # numbers of group members (dims. of subvectors)
    vecz = z.ravel()
    for i in range(m):
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
        az = thresholding(normzsn, np.sqrt(ms) * (th / np.sqrt(m,dtype=ms.dtype))) / normzsn
    else:
        #az = np.max(normzsn - th, 0.) / normzsn
        az = thresholding(normzsn, th) / normzsn

    # scaling
    v = np.zeros_like(z)
    vecv = v.ravel()
    vecg = garray.ravel()
    for i in range(m):
        vecv[i] = vecz[i] * az[vecg[i]]

    return v


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
    normsz = np.zeros(ng, dtype=z.dtype)   # squared norms of subvectors
    ms = np.zeros(ng, dtype=z.dtype)       # numbers of group members (dims. of subvectors)
    vecz = z.ravel()
    for i in range(m):
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
        az = __smoothly_clipped_absolute_deviation(normzsn, np.sqrt(ms) * (th / np.sqrt(m,dtype=ms.dtype)),a) / normzsn
    else:
        #az = np.max(normzsn - th, 0.) / normzsn
        az = smoothly_clipped_absolute_deviation(normzsn, th, a) / normzsn

    # scaling
    v = np.zeros_like(z)
    vecv = v.ravel()
    vecg = garray.ravel()
    for i in range(m):
        vecv[i] = vecz[i] * az[vecg[i]]

    return v


#def group_scad(z, th, garray, normalize=True, a=3.7):
#    return group_soft(z, th, garray, normalize=normalize, thresholding=lambda x,t: __smoothly_clipped_absolute_deviation(x,t,a=a))
