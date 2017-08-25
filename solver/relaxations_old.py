# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 08:42:11 2017

@author: tsakai
"""

from math import sqrt
import numpy as np
from scipy import linalg
import scipy.sparse.linalg as splinalg

import spmlib.thresholding as th
import spmlib.proxop as prox


#%%
# Iterative soft thresholding algorithm
# Use FISTA in practice
def iterative_soft_thresholding(A, b, x=None, tol=1e-5, maxiter=1000, l=1., L=None):
#    pobj = []
    if x is None:
        x = np.zeros(A.shape[1])
    if L is None:
        L = linalg.norm(A) ** 2  # Lipschitz constant
#    time0 = time.time()
    count = 0
    r = b - A.dot(x)
    while count < maxiter and linalg.norm(r) > tol:
#    for _ in xrange(maxiter):
        count = count + 1
        x = th.soft(x + A.conj().T.dot(r) / L, l / L)
        r = b - A.dot(x)
        if np.max(np.abs(x)) > 1e+20:          # check overflow
            x = np.zeros(A.shape[1])
            L *= 10
            count = 0
            print('IST(): Overflow. Restarted with a larger Lipschitz constant.')
#        this_pobj = 0.5 * linalg.norm(A.dot(x) - b) ** 2 + l * linalg.norm(x, 1)
#        pobj.append((time.time() - time0, this_pobj))

#    times, pobj = map(np.array, zip(*pobj))
#    return x, pobj, times
    return x, r, count


#%%
# Fast IST algorithm
#
# x, r, count = FISTA(A, b, x=None, tol=1e-5, maxiter=1000, xtol=1e-12, l=1., L=None, eta=2., 
#                     Nesterov=True, restart_every=np.nan, prox=lambda z,th: soft_thresh(z, th))
# solves
# x = arg min_x f(x) + g(x)   where f(x)=0.5||b-Ax||^2, g(x)=l*||x||_1
#   = arg min_x 0.5*|| b - A x ||_2^2 + l' * abs(x)
#
# input:
# A      : m x n matrix, LinearOperator, or tuple (fA, fAT) of lambda functions fA(z)=A.dot(z) and fAT(r)=A.conj().T.dot(r)
# b      : m-dimensional vector
# x      : initial guess, (A.conj().T.dot(b) by default), will be mdified in this function
# tol    : tolerance for residual (1e-5 by default)
# maxiter: max. iterations (1000 by default)
# xtol   : tolerance for x displacement (1e-12 by default)
# l      : barancing parameter lambda (1. by default)
# L      : Lipschitz constant (automatically computed by default)
# eta    : magnification L*=eta in the linear search of L
# Nesterov: Nesterov acceleration (True by default)
# restart_every: restart the Nesterov acceleration every this number of iterations (disabled by default)
# prox   : proximity operator of g(x), the soft thresholding soft_thresh(z, th) by default for g(x)=th*||x||_1.
#
# output:
# x     : sparse solution
# r     : residual (b - Ax)
# count : loop count at termination
#
# Example:
# x = FISTA(A, b, l=1.5, maxiter=30000, tol=linalg.norm(b)*1e-12, restart_every=500)[0]
#
def fista(A, b, x=None, tol=1e-5, maxiter=1000, xtol=1e-12, l=1., L=None, eta=2., Nesterov=True, restart_every=np.nan, prox=lambda z,l: prox.l1(z,l)):

    # define the functions that compute projections by A and its adjoint
    if type(A) is tuple:
        fA = A[0]
        fAT = A[1]
    else:
        A = splinalg.aslinearoperator(A)
        fA = A.matvec
        fAT = A.rmatvec

    # initialize x
    if x is None:
        x = fAT(b)

    # initialize variables
    t = 1
    w = x.copy()

    # roughly estimate the Lipschitz constant
    if L is None:
        L = 2*linalg.norm(fA(fAT(b))) / linalg.norm(b)

    count = 0
    r = b - fA(w)  # residual
    while count < maxiter and linalg.norm(r) > tol:
        count += 1
        xold = x.copy()
        # x = prox(w + A.conj().T.dot(r) / L, l / L)
        x = prox(w + fAT(r) / L, l / L)
        dx = x - xold

        if np.mod(count, restart_every) == 0:
            t = 1.
            
        if Nesterov:
            told = t
            t = 0.5 * (1. + sqrt(1. + 4. * t * t))
            w = x + ((told - 1.) / t) * dx
        else:
            w = x

        r = b - fA(w)
        
        if linalg.norm(dx) < xtol:
            break

        if np.max(np.abs(x)) > 1e+20:          # check overflow
            x = fAT(b)
            w = x.copy()
            r = b.copy()
            L *= eta
            count = 0
            print('FISTA(): Overflow. Restarted with a larger Lipschitz constant L = %.2e' % (L))

    return x, r, count

