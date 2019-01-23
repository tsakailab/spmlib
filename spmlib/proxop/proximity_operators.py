# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 08:42:11 2017

@author: tsakai
"""

import numpy as np
from scipy import linalg
import scipy.sparse.linalg as splinalg
import spmlib.thresholding as th

# Proximity operators:
#     prox_(l*f())(q) = arg min_x  l * f(x) + (1./2.) || x - q ||_2^2
# Example:
# >>> from spmlib import proxop as prox



def ind_linfball(q, r, c=None):
    """
    prox of linf-ball with center c and radius r
    
    arg min_x ind_linfball(x,r,c) + 0.5*||x-q||_2^2 = q if q in the linf-ball, 
    otherwise q + soft(c-q, r)
    q can be an ndarray.
    """
    if c is None:
        return q + th.soft(-q, r)
    else:
        return q + th.soft(c-q, r)



def ind_l2ball(q, r, c=None):
    """
    prox of l2-ball with center c and radius r
    
    arg min_x ind_l2ball(x,r,c) + 0.5*||x-q||_2^2 = q if q in the l2-ball, 
    otherwise c + r * (q-c)/||q-c||_2
    q and c can be ndarrays.
    """
    if c is None:
        normq = linalg.norm(q)
        if normq > r:
            return r * q / normq
        else:
            return q
    else:
        qc = q - c
        normqc = linalg.norm(qc)
        if normqc > r:
            return c + r * qc / normqc
        else:
            return q



def squ_l2_from_subspace(q, U, l=1):
    """
    prox of q w.r.t a distance from a linear subspace whose orthonormal basis is U:

        arg min_x 0.5*||l.*(I - U*U')*x||_2^2 + 0.5*||x-q||_2^2 = (q + l.*U*(U'*q)) ./ (l+1)

    U : m x r matrix, LinearOperator, or tuple(fU, fUT) of lambda functions fU(x)=U.dot(x) and fUT(y)=U.conj().T.dot(y),
         as an orthonormal basis of an r-dimensional linear subspace of an m-dimensional space.

    For a linear affine subspace with an offset c, use squ_l2_from_subspace(q-c, l, U).
    Note: the l2 norm squared in this definition has the coefficient 0.5.
    """

    # define the functions that compute projections by U and its adjoint
    if type(U) is tuple:
        fU = U[0]
        fUT = U[1]
    else:
        Us = splinalg.aslinearoperator(U)
        fU = Us.matvec
        fUT = Us.rmatvec

    return (q + l * fU(fUT(q))) / (l+1.)



def squ_l2(q, l=1, c=None, U=None):
    """
    arg min_x 0.5*||sqrt(l).*(x-c)||_2^2 + 0.5*||x-q||_2^2 = (q+l.*c) ./ (l+1)
    q, c and l can be ndarrays if U is None.
    Note: the l2 norm squared in this definition has the coefficient 0.5.
    """
    if U is None:
        if c is None:
            return q / (l+1.)
        else:
            return (q + l * c) / (l+1.)
    else:
        return squ_l2_from_subspace(q, l, U)



def l2(q, l=1, c=None):
    """
    arg min_x l*||x-c||_2 + 0.5*||x-q||_2^2 = c + soft(||q-c||_2,l)/||q-c||_2 * (q-c)
    """
    th.l2_soft_thresholding(q, l, c=c)



def l1(q, l=1):
    """
    arg min_x ||l.*x||_1 + 0.5*||x-q||_2^2
    q and l can be ndarrays.
    """
    return th.soft(q, l)



def l0(q, l=1):
    """
    arg min_x |||l.*x||_0 + 0.5*||x-q||_2^2 = q .* (q>sqrt(2*l))
    q and l can be ndarrays.
    """
    return th.hard(q, np.sqrt(2.*l))



def scad(z, th, a=3.7):
    """
    prox of the moothly clipped absolute deviation regularizer
    """
    return th.smoothly_clipped_absolute_deviation(z, th, a)



def nuclear(Q, l=1):
    """
    arg min_X l*||X||_* + 0.5*||X-Q||_F^2
    """
    return th.singular_value_thresholding(Q, l)



def l21(Q, l=1):
    """
    Soft threshlding of l2 norms of row vectors without changing the directions,
    as the proximity operator for the matrix l2,1 norm.
        P = arg min_X ||diag(l)*X||_2,1 + 0.5*||X-Q||_F^2
    Here ||X||_2,1 = sum(sqrt(sum(abs(X).^2,2))), i.e., sum of l2 norm of row vectors of X.

    Parameters
    ----------
    Q : ndarray, shape (`m`, `n`)
        `m` x `n` matrix.
        `m`-dimensional vector to be thresholded group-wise.
    l : scalar or 1d array, default 1
        Weight for each row vector.

    Returns
    -------
    P : ndarray, shape (`m`, `n`)
        The proximal matrix.
    """
    P = np.zeros_like(Q)
    if np.array(l).size == 1:
        for i in range(Q.shape[0]):
            P[i,:] = th.l2_soft_thresholding(Q[i,:], l)
    else:
        for i in range(Q.shape[0]):
            P[i,:] = th.l2_soft_thresholding(Q[i,:], l[i])                
    return P







#%% EXPERIMENTAL
# 1/(1+w)*(prox.squ_l2(q,l,c) + w*prox.ind_l2ball(q,r,c))
def ave_squ_ind_l2ball(q, l, r, w, c=None):
    return 1./(1.+w) * (squ_l2(q, l, c) + w * ind_l2ball(q, r, c))

