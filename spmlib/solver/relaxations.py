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
        x = iterative_soft_thresholding(A, b, x=x, tol=tol, maxiter=switch_to_scad_after, tolx=tolx, l=l, L=L, eta=eta, nesterovs_momentum=nesterovs_momentum, restart_every=restart_every)[0]

    # FISTA with SCAD thresholding to debias
    return iterative_soft_thresholding(A, b, x=x, tol=tol, maxiter=maxiter, tolx=tolx, l=l, L=L, eta=eta, nesterovs_momentum=nesterovs_momentum, #restart_every=restart_every,
                                       prox=lambda z,thresh: th.smoothly_clipped_absolute_deviation(z,thresh,a=a))



def BPdelta_scad(A, b, x=None, delta=None, maxiter=1000, rtol=1e-4, alpha=None, rho=1., eta=2., nesterovs_momentum=True, restart_every=np.nan, a=3.7, switch_to_scad_after = 0,
                                                 prox=lambda z,l: prox.l1(z,l), prox_loss=lambda z,delta: prox.ind_l2ball(z,delta)):
    # BPdelta up to switch_to_scad_after times to find good initial guess with bias
    if switch_to_scad_after > 0:
        x = basis_pursuit_delta(A, b, x=x, delta=delta, maxiter=switch_to_scad_after, rtol=rtol, alpha=alpha, rho=rho, eta=eta,
                                nesterovs_momentum=nesterovs_momentum, restart_every=restart_every, prox_loss=prox_loss)[0]

    # BPdelta with SCAD thresholding to debias
    return basis_pursuit_delta(A, b, x=x, delta=delta, maxiter=maxiter, rtol=rtol, alpha=alpha, rho=rho, eta=eta, nesterovs_momentum=nesterovs_momentum, #restart_every=restart_every,
                                       prox=lambda z,thresh: th.smoothly_clipped_absolute_deviation(z,thresh,a=a))



def iterative_soft_thresholding(A, b, x=None, 
                                tol=1e-5, maxiter=1000, tolx=1e-12, 
                                l=1., L=None, eta=2., 
                                nesterovs_momentum=False, restart_every=np.nan, 
                                prox=lambda z,l: prox.l1(z,l)):
    """
    Iterative soft thresholding algorithm    
    solves
    x = arg min_x f(x) + g(x)   where f(x)=0.5||b-Ax||^2, g(x)=l*||x||_1
      = arg min_x 0.5*|| b - A x ||_2^2 + l' * abs(x)

    Parameters
    ----------
    A : m x n matrix, LinearOperator, or tuple (fA, fAT) of functions fA(z)=A.dot(z) and fAT(r)=A.conj().T.dot(r).
    b : m-dimensional vector.
    x : initial guess, (A.conj().T.dot(b) by default), will be mdified in this function.
    tol : tolerance for residual (1e-5 by default) as a stopping criterion.
    maxiter: max. iterations (1000 by default) as a stopping criterion.
    tolx : tolerance for x displacement (1e-12 by default) as a stopping criterion.
    l : barancing parameter lambda (1. by default).
    L : Lipschitz constant (automatically computed by default).
    eta : magnification L*=eta in the linear search of L.
    nesterovs_momentum : Nesterov acceleration (False by default).
    restart_every: restart the Nesterov acceleration every this number of iterations (disabled by default).
    prox : proximity operator of g(x), the soft thresholding soft_thresh(z,l) (=prox.l1(z,l)) by default for g(x)=l*||x||_1.

    Returns
    -------
    x : sparse solution.
    r : residual (b - Ax).
    count : loop count at termination.

    Example
    -------
    >>> x = iterative_soft_thresholding(A, b, l=1.5, maxiter=1000, tol=linalg.norm(b)*1e-12, nesterovs_momentum=True)[0]
    """

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
        dx = x.copy()  # old x
        # x = prox(w + A.conj().T.dot(r) / L, l / L)
        x = prox(w + fAT(r) / L, l / L)
        dx = x - dx

        if np.fmod(count, restart_every) == 0:
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



def greedy_coordinate_descent(A, b, x=None, 
                                tol=1e-5, maxiter=1000, tolx=1e-12, 
                                l=1.,
                                nesterovs_momentum=False, restart_every=np.nan, 
                                prox=lambda z,l: prox.l1(z,l), N=1):
    """
    Coordinate descent algorithm
    solves
    x = arg min_x f(x) + g(x)   where f(x)=0.5||b-Ax||^2, g(x)=l*||x||_1
      = arg min_x 0.5*|| b - A x ||_2^2 + l' * abs(x)

    Parameters
    ----------
    A : m x n matrix, LinearOperator, or tuple (fA, fAT) of functions fA(z)=A.dot(z) and fAT(r)=A.conj().T.dot(r).
    b : m-dimensional vector.
    x : initial guess, (A.conj().T.dot(b) by default), will be mdified in this function.
    tol : tolerance for residual (1e-5 by default) as a stopping criterion.
    maxiter: max. iterations (1000 by default) as a stopping criterion.
    tolx : tolerance for x displacement (1e-12 by default) as a stopping criterion.
    l : barancing parameter lambda (1. by default).
    nesterovs_momentum : Nesterov acceleration (False by default).
    restart_every: restart the Nesterov acceleration every this number of iterations (disabled by default).
    prox : proximity operator of g(x), the soft thresholding soft_thresh(z,l) (=prox.l1(z,l)) by default for g(x)=l*||x||_1.
    N : number of coordinates to pick at each step.

    Returns
    -------
    x : sparse solution.
    r : residual (b - Ax).
    count : loop count at termination.

    Example
    -------
    >>> x = greedy_coordinate_descent(A, b, l=1.5, maxiter=1000, tol=linalg.norm(b)*1e-12, N=3)[0]
    """

    # define the functions that compute projections by A and its adjoint
    if type(A) is tuple:
        fA = A[0]
        fAT = A[1]
    else:
        linopA = splinalg.aslinearoperator(A)
        fA = linopA.matvec
        fAT = linopA.rmatvec

    # initialize x
    if x is None:
        x = np.zeros_like(fAT(b))
        #x = np.zeros(A.shape[1], dtype=b.dtype)

    # A = splinalg.LinearOperator((b.shape[0],x.shape[0]), matvec=fA, rmatvec=fAT)

    # initialize variables
    t = 1

    count = 0
    r = b - fA(x)  # residual
    c = fAT(b)
    while count < maxiter and linalg.norm(r) > tol:
        count += 1

        z = prox(c, l)
        dx = z - x
        s = np.argsort(-np.abs(dx))[:N]  # multiple coordinate choice (tsakai heuristic)
        #s = np.argmax(np.abs(dx))

        dxs = np.zeros_like(dx)
        dxs[s] = dx[s]
        #cs = c[s]
        c = c + dxs - fAT(fA(dxs))  # Gregor&LeCun version
        #c[s] = cs  # Li&Osher original version

        dx = x.copy()
        x[s] = z[s]
        dx = x - dx

        if np.fmod(count, restart_every) == 0:
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
            c = fAT(b)
            count = 0
            print('CoD(): Overflow.')

    return x, r, count



def basis_pursuit_delta(A, b, x=None, 
                                delta=None, maxiter=1000, rtol=1e-4, 
                                alpha=None, rho=1., eta=2.,
                                nesterovs_momentum=False, restart_every=np.nan, 
                                prox=lambda q,l: prox.l1(q,l), prox_loss=lambda q,delta: prox.ind_l2ball(q,delta)):
    """
    ADMM algorithm with subproblem approximation for constrained basis pursuit denoising (a.k.a. BP delta) in Eq. (2.16) [Yang&Zhang11] 
    solves
    x = arg min_x g(x) s.t. f(b-Ax)=||b-Ax||_2 <= delta   where g(x)=||x||_1 (by default)

    Parameters
    ----------
    A : m x n matrix, LinearOperator, or tuple (fA, fAT) of functions fA(z)=A.dot(z) and fAT(r)=A.conj().T.dot(r).
    b : m-dimensional vector.
    x : initial guess, (A.conj().T.dot(b) by default), will be mdified in this function.
    delta : tolerance for residual (1e-3*||b||_2 by default).
    maxiter: max. iterations (1000 by default) as a stopping criterion.
    rtol : scalar, optional, default 1e-4
        Relative convergence tolerance of `y` and `z` in ADMM, i.e., the primal and dual residuals as a stopping criterion.
    alpha : scaling factor of the gradient of the quadratic term in the ADMM subproblem for g(x) to approximate the proximity operator (automatically computed by default).
    rho : scalar, optional, default 1.
        Augmented Lagrangian parameter.
    nesterovs_momentum : Nesterov acceleration (False by default).
    restart_every: restart the Nesterov acceleration every this number of iterations (disabled by default).
    prox : proximity operator of sparsity inducing function g(x), the soft thresholding soft_thresh(q,l) (=prox.l1(q,l)) by default.
    prox_loss : proximity operator of loss function f(b-Ax), the orthogonal projection onto l2 ball with radius delta (=prox.ind_l2ball(q,delta)) by default.

    Returns
    -------
    x : sparse solution.
    r : residual (b - Ax).
    count : loop count at termination.

    References
    ----------
    J. Yang and Y. Zhang
    "Alternating Direction Algorithms for $\ell_1$-Problems in Compressive Sensing"
    SIAM J. Sci. Comput., 33(1), pp. 250-278, 2011.
    https://epubs.siam.org/doi/10.1137/090777761

    Example
    -------
    >>> x = basis_pursuit_delta(A, b, delta=0.05*linalg.norm(b), maxiter=1000, nesterovs_momentum=True)[0]
    """

    # define the functions that compute projections by A and its adjoint
    if type(A) is tuple:
        fA = A[0]
        fAT = A[1]
    else:
        linopA = splinalg.aslinearoperator(A)
        fA = linopA.matvec
        fAT = linopA.rmatvec

    # initialize x
    if x is None:
        x = fAT(b)

    # initialize delta
    if delta is None:
        delta = linalg.norm(b) * 1e-3

    # initialize alpha using rough estimate of the Lipschitz constant
    # alpha L + 1/rho < 2
    if alpha is None:
        L = 2*linalg.norm(fA(fAT(b))) / linalg.norm(b)
        #L = linalg.norm(A) ** 2  # Lipschitz constant
        alpha = (2.-1./rho)/L
        #print('basis_pursuit_delta(): alpha = %.2e' % (alpha))

    y = np.zeros_like(b) # np.zeros(3*m, dtype=c.dtype)

    # initialize variables
    t = 1.

    count = 0
    while count < maxiter:
        count += 1

        if np.fmod(count, restart_every) == 0:
            t = 1.
        if nesterovs_momentum:
            told = t
            t = 0.5 * (1. + sqrt(1. + 4. * t * t))

        # update r
        #dr = r.copy()  # old r
        r = prox_loss(b - fA(x) - y, delta)
        #dr = r - dr

        # update x
        g = fAT( fA(x) + r - b + y )
        dx = x.copy()  # old x
        x = prox(x - alpha * g, alpha/rho)
        dx = x - dx
        
        # update y
        dy = y.copy()
        y = y + r + fA(x) - b
        dy = y - dy

        # Nesterov acceleration
        if nesterovs_momentum:
            #r = r + ((told - 1.) / t) * dr
            #x = x + ((told - 1.) / t) * dx
            y = y + ((told - 1.) / t) * dy
        
        # check convergence of primal and dual residuals
        if linalg.norm(dx) < rtol * linalg.norm(x) and linalg.norm(dy) < rtol * linalg.norm(y):
            break

        if np.max(np.abs(x)) > 1e+20:          # check overflow
            x = fAT(b)
            y = np.zeros_like(b)
            r = b.copy()
            alpha /= eta
            count = 0
            print('basis_pursuit_delta(): Overflow. Restarted with a smaller constant alpha = %.2e' % (alpha))

    r = b - fA(x)  # residual
    return x, r, count



def basis_pursuit_delta_DADM(A, b, x=None, 
                                delta=None, maxiter=1000, rtol=1e-4, rho=1.,
                                nesterovs_momentum=False, restart_every=np.nan, 
                                prox=lambda q,r: prox.ind_linfball(q,r), prox_loss=lambda q,r: prox.ind_l2ball(q,r)):
    """
    DADM: Dual-based alternating directino method, appeared in [Yang&Zhang11] as Eq. (2.28),
    solves the dual of constrained basis pursuit denoising:
    y = arg max_y  b.dot(y) - delta*f(y) + g(fAT(y)),
    where f(y)=||y||_2 and g(fAT(y))=indicator function for fA(y), i.e., zero if ||fAT(y)||_inf <= 1, infinity otherwise (by default).

    Parameters
    ----------
    A : m x n matrix, LinearOperator, or tuple (fA, fAT) of functions fA(z)=A.dot(z) and fAT(r)=A.conj().T.dot(r).
    b : m-dimensional vector.
    x : initial guess, (A.conj().T.dot(b) by default), will be mdified in this function.
    delta : tolerance for residual (1e-3*||b||_2 by default).
    maxiter: max. iterations (1000 by default) as a stopping criterion.
    rtol : scalar, optional, default 1e-4
        Relative convergence tolerance of `y` and `z` in ADMM, i.e., the primal and dual residuals as a stopping criterion.
    rho : scalar, optional, default 1.
        Augmented Lagrangian parameter.
    nesterovs_momentum : Nesterov acceleration (False by default).
    restart_every: restart the Nesterov acceleration every this number of iterations (disabled by default).
    prox : proximity operator of sparsity inducing function g(x), the orthogonal projection onto linf ball with radius 1 (=prox.ind_linfball(q,r) by default.
    prox_loss : orthogonal projection onto l2 ball with radius delta (=prox.ind_l2ball(q,delta)) by default.

    Returns
    -------
    x : sparse solution.
    r : residual (b - Ax).
    count : loop count at termination.

    References
    ----------
    J. Yang and Y. Zhang
    "Alternating Direction Algorithms for $\ell_1$-Problems in Compressive Sensing"
    SIAM J. Sci. Comput., 33(1), pp. 250-278, 2011.
    https://epubs.siam.org/doi/10.1137/090777761

    Example
    -------
    >>> x = basis_pursuit_delta_DADM(A, b, delta=0.05*linalg.norm(b), maxiter=1000, nesterovs_momentum=True)[0]
    """

    # define the functions that compute projections by A and its adjoint
    if type(A) is tuple:
        fA = A[0]
        fAT = A[1]
    else:
        linopA = splinalg.aslinearoperator(A)
        fA = linopA.matvec
        fAT = linopA.rmatvec

    # initialize x
    if x is None:
        x = fAT(b)

    # initialize delta
    if delta is None:
        delta = linalg.norm(b) * 1e-3

    y = np.zeros_like(b) # np.zeros(3*m, dtype=c.dtype)

    # initialize variables
    t = 1.

    count = 0
    z = x
    while count < maxiter:
        count += 1

        if np.fmod(count, restart_every) == 0:
            t = 1.
        if nesterovs_momentum:
            told = t
            t = 0.5 * (1. + sqrt(1. + 4. * t * t))

        # update z
        #dz = z.copy()  # old z
        #z = prox(x + fAT(y), 1)
        z = prox(x/rho + fAT(y), 1)
        #dz = z - dz

        # update y
        dy = y.copy()  # old y
        #q = fA(z - x) + b/rho
        q = fA(z) - (fA(x) - b)/rho
        #print linalg.norm(q)
        y = q - prox_loss(q, delta/rho)
        dy = y - dy
        
        # update x
        dx = x.copy()
        #x = x - z + fAT(y)
        x = x - rho*(z - fAT(y))
        dx = x - dx

        # Nesterov acceleration
        if nesterovs_momentum:
            #z = z + ((told - 1.) / t) * dz
            #x = x + ((told - 1.) / t) * dx
            y = y + ((told - 1.) / t) * dy
        
        # check convergence of primal and dual residuals
        if linalg.norm(dx) < rtol * linalg.norm(x) and linalg.norm(dy) < rtol * linalg.norm(y):
            break

    r = b - fA(x)  # residual
    return x, r, count

