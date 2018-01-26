# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 21:05:11 2018

@author: tsakai
"""

import numpy as np
from spmlib.thresholding import soft
from spmlib.thresholding import smoothly_clipped_absolute_deviation as scad

from numba import jit
@jit
def group_soft(z, th, garray, normalize=True):
    """
    Group-wise soft threshlding
        soft(||z(g)||_2, `th`) / ||z(g)||_2 * z(g) for g in `groups`, 
        i.e., `l2_soft_thresholding(z[g], th)`.
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
    normsz = np.zeros(ng)   # squared norms of subvectors
    ms = np.zeros(ng)       # numbers of group members (dims. of subvectors)
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
        az = soft(normzsn, np.sqrt(ms) * th / np.sqrt(m)) / normzsn
    else:
        #az = np.max(normzsn - th, 0.) / normzsn
        az = soft(normzsn, th) / normzsn

    # scaling
    v = np.zeros_like(z)
    vecv = v.ravel()
    vecg = garray.ravel()
    for i in range(m):
        vecv[i] = vecz[i] * az[vecg[i]]

    return v



from numba import jit
@jit
def group_scad(z, th, garray, normalize=True):
    """
    Group-wise soft threshlding
        soft(||z(g)||_2, `th`) / ||z(g)||_2 * z(g) for g in `groups`, 
        i.e., `l2_soft_thresholding(z[g], th)`.
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
    normsz = np.zeros(ng)   # squared norms of subvectors
    ms = np.zeros(ng)       # numbers of group members (dims. of subvectors)
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
        az = scad(normzsn, np.sqrt(ms) * th / np.sqrt(m)) / normzsn
    else:
        #az = np.max(normzsn - th, 0.) / normzsn
        az = scad(normzsn, th) / normzsn

    # scaling
    v = np.zeros_like(z)
    vecv = v.ravel()
    vecg = garray.ravel()
    for i in range(m):
        vecv[i] = vecz[i] * az[vecg[i]]

    return v
