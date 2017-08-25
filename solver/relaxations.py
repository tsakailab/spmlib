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
from spmlib.solver import orthogonal_matching_pursuit as omp




#%%
# FISTA in practical use
def fista(A, b, x=None, 
          tol=1e-5, maxiter=1000, tolx=1e-12, 
          l=1., L=None, eta=2., 
          nesterovs_momentum=True, restart_every=np.nan, 
          prox=lambda z,l: prox.l1(z,l),
          debias=False):
    # FISTA to find support
    x_result, r_result, count = iterative_soft_thresholding(A, b, x=x, tol=tol, maxiter=maxiter, tolx=tolx, l=l, L=L, eta=eta, nesterovs_momentum=nesterovs_momentum, restart_every=restart_every, prox=prox)

    # debias by least squares (equivalent to a single step of OMP)
    if debias:
        x_result, r_result = omp(A, b, maxnnz=np.count_nonzero(x_result), s0=np.nonzero(x_result)[0])[0:2]
    return x_result, r_result, count



def fista_scad(A, b, x=None, tol=1e-5, maxiter=1000, tolx=1e-12, 
               l=1., L=None, eta=2., nesterovs_momentum=True, restart_every=np.nan, 
               a=3.7, switch_to_scad_after = 0):
    # FISTA up to switch_to_scad_after times to find good initial guess with bias
    if switch_to_scad_after > 0:
        x_result = iterative_soft_thresholding(A, b, x=x, tol=tol, maxiter=switch_to_scad_after, tolx=tolx, l=l, L=L, eta=eta, nesterovs_momentum=nesterovs_momentum, restart_every=restart_every)[0]
    else:
        x_result = None

    # FISTA with SCAD thresholding to debias
    return iterative_soft_thresholding(A, b, x=x_result, tol=tol, maxiter=maxiter, tolx=tolx, l=l, L=L, eta=eta, nesterovs_momentum=nesterovs_momentum, #restart_every=restart_every,
                                       prox=lambda z,thresh: th.smoothly_clipped_absolute_deviation(z,thresh,a=a))



#%%
# Iterative soft thresholding algorithm
#
# x, r, count = iterative_soft_thresholding(A, b, x=None, 
#                   tol=1e-5, maxiter=1000, tolx=1e-12, l=1., L=None, eta=2., 
#                   Nesterov=True, restart_every=np.nan, prox=lambda z,th: soft_thresh(z, th))
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
# tolx   : tolerance for x displacement (1e-12 by default)
# l      : barancing parameter lambda (1. by default)
# L      : Lipschitz constant (automatically computed by default)
# eta    : magnification L*=eta in the linear search of L
# nesterovs_momentum: Nesterov acceleration (False by default)
# restart_every: restart the Nesterov acceleration every this number of iterations (disabled by default)
# prox   : proximity operator of g(x), the soft thresholding soft_thresh(z,l) (=prox.l1(z,l)) by default for g(x)=l*||x||_1.
#
# output:
# x     : sparse solution
# r     : residual (b - Ax)
# count : loop count at termination
#
# Example:
# x = iterative_soft_thresholding(A, b, l=1.5, maxiter=1000, tol=linalg.norm(b)*1e-12, nesterovs_momentum=True)
#
def iterative_soft_thresholding(A, b, x=None, tol=1e-5, maxiter=1000, tolx=1e-12, l=1., L=None, eta=2., nesterovs_momentum=False, restart_every=np.nan, prox=lambda z,l: prox.l1(z,l)):

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

    # A = splinalg.LinearOperator((b.shape[0],x.shape[0]), matvec=fA, rmatvec=fAT)

    # initialize variables
    t = 1
    w = x.copy()

    # roughly estimate the Lipschitz constant
    if L is None:
        L = 2*linalg.norm(fA(fAT(b))) / linalg.norm(b)
        #L = linalg.norm(A) ** 2  # Lipschitz constant

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
            
        if nesterovs_momentum:
            told = t
            t = 0.5 * (1. + sqrt(1. + 4. * t * t))
            w = x + ((told - 1.) / t) * dx
        else:
            w = x

        r = b - fA(w)
        
        if linalg.norm(dx) < tolx:
            break

        if np.max(np.abs(x)) > 1e+20:          # check overflow
            x = fAT(b)
            w = x.copy()
            r = b.copy()
            L *= eta
            count = 0
            print('FISTA(): Overflow. Restarted with a larger Lipschitz constant L = %.2e' % (L))

    return x, r, count

