#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 23:47:00 2017

@author: tsakai
"""

import numpy as np


#%% Create n-d full vector from nz =(nonzeros, support) (indices can be duplicate)
def _toarray1d(n, nz, duplicate=False):
    nonzeros = np.array(nz[0])
    v = np.zeros(n, dtype=nonzeros.dtype)
    if not duplicate:
#        v = np.zeros(n, dtype=nonzeros.dtype)
        v[nz[1]] = nonzeros.ravel()
    else:
        # because spv[support] += nonzeros doesn't work as expected ..
        j = 0
        for s in nz[1]:
            v[s] += nonzeros[j]
            j += 1
    return v


def spvec(n, nz, duplicate=False, toarray=True):
    if toarray:
        return _toarray1d(n, nz, duplicate=duplicate)
    else:
        return nz


#%% numbers of data and dimensionality
def num_and_dim(X, axis=0):
    Xshape = X.shape
    if len(Xshape) == 1:
        return (1, Xshape[0])
    if Xshape[0] == 1 or Xshape[1] == 1:
        return (1, max(Xshape))
    return (Xshape[axis], Xshape[axis-1])


#%% Vectorize M
def vec(M):
    return M.ravel()


#%% unroll v to the shape
def unvec(v, shape):
    return np.reshape(v, shape)


#%% unroll v to the shape
def mat(v, shape):
    return np.reshape(v, shape)


def _dtype_uint_upto(N):
  for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
    if N <= dtype(-1):
      return dtype
#  raise StandardError('{} is too large integer.'.format(N))



# Convert a list of index lists into a 1d array of group IDs, .e.g.,
# [[0,2,3],[4,5],[1]] -> array([0,2,0,0,1,1], dtype=uint8)
# groups[g] is a list of indices in range(n) of the g-th group.
def index_groups_to_array(groups):
    # count the number of all members
    m = len(sum(groups,[]))   # any other efficient way?
    garray = np.zeros(m, dtype=_dtype_uint_upto(m))
    gid = 0
    for g in groups:
        for i in g:
            garray[i] = gid
        gid += 1
    return garray


# order preserving uniquification of list
def unique(seq):
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]


