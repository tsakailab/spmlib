#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 23:47:00 2017

@author: tsakai
"""

import numpy as np


#%% Create full vector from its nonzero components and support (indices can be duplicate)
def spvec(n, nonzeros, support, duplicate=False):
    nonzeros = np.array(nonzeros)
    spv = np.zeros(n, dtype=nonzeros.dtype)
    if not duplicate:
        spv = np.zeros(n, dtype=nonzeros.dtype)
        spv[support] = nonzeros
    else:
        # because spv[support] += nonzeros doesn't work as expected ..
        j = 0
        for s in support:
            spv[s] += nonzeros[j]
            j += 1
    return spv


#%% Vectorize M
def vec(M):
    return M.ravel()


#%% unroll v to the shape
def unvec(v, shape):
    return np.reshape(v, shape)


#%% unroll v to the shape
def mat(v, shape):
    return np.reshape(v, shape)

