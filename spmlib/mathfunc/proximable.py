#!/usr/bnin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 19:41:50 2018

@author: tsakai
"""

import numpy as np
from scipy import linalg
import scipy.sparse.linalg as splinalg


def lp(x, p=2):
    return linalg.norm(x, p)

def linf(x):
    return linalg.norm(x,np.inf)

def squ_l2_from_subspace(x, U, w=1.):
    """
    0.5*(squared l2 distance between `x` and the subspace span`U`)
    0.5*||w.*(I - U*U')*x||_2^2
    """
    if type(U) is tuple:
        fU = U[0]
        fUT = U[1]
    else:
        Us = splinalg.aslinearoperator(U)
        fU = Us.matvec
        fUT = Us.rmatvec
    return 0.5*linalg.norm(w*(x+fU(fUT(x))))**2

def squ_l2(x):
    return 0.5*linalg.norm(x)**2

def l2(x):
    return linalg.norm(x)

def l1(x):
    return np.sum(np.abs(x))

def l0(x):
    return np.count_nonzero(x)


def _indicator(f, r):
    if f > r:
        return np.inf
    else:
        return 0.

def ind_lpball(x, r, p=2, c=None):
    pi = (0, 1, 2, np.inf)
    lpnorm = (l0, l1, l2, linf)
    if p in pi:
        if c is None:
            return _indicator(lpnorm[p](x), r)
        else:
            return _indicator(lpnorm[p](x-c), r)
    else:
        if c is None:
            return _indicator(lp(x,p), r)
        else:
            return _indicator(lp(x-c,p), r)

def ind_linfball(x, r, c=None):
    if c is None:
        return _indicator(linf(x), r)
    else:
        return _indicator(linf(x-c), r)

def ind_l2ball(x, r, c=None):
    if c is None:
        return _indicator(l2(x), r)
    else:
        return _indicator(l2(x-c), r)

def ind_l1ball(x, r, c=None):
    if c is None:
        return _indicator(l1(x), r)
    else:
        return _indicator(l1(x-c), r)

def ind_l0ball(x, r, c=None):
    if c is None:
        return _indicator(l0(x), r)
    else:
        return _indicator(l0(x-c), r)


def nuclear(Q):
    return np.sum(linalg.svd(Q, compute_uv=False))
