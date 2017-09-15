#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:15:10 2017

@author: tsakai
"""

from math import sqrt
import numpy as np
from scipy import linalg
import spmlib.proxop as prox
from spmlib.linalg import as2darray


#%%
# Incremental SVD
#
# U, sv = column_incremental_SVD(C, U, sv, forget=1., max_rank=np.inf, min_sv=0, orth_eps=1e-12, OrthogonalizeU=False)
#
# performs the incremental SVD in
# M. Brand, "Incremental singular value decomposition of uncertain data with missing values",
# ECCV2002, p. 707-720, 2002.
#
# input:
# C        : m x nc matrix of column vectors to append
# U        : m x r matrix of left singular vectors (overwritten with the update)
# sv       : r-dimensional vector of singular values (overwritten with the update)
# forget   : forgetting parameter (0<forget<=1, 1 by default)
# max_rank : maximum rank (m by default)
# min_sv   : smaller singular values than min_sv is neglected (0 by default)
# orth_eps : rank increases if the magnitude of C in the orthogonal subspace is larger than orth_eps (1e-12 by default)
# OrthogonalizeU : if True, perform QR decomposition to orthogonalize U (True by default)
#
# output:
# U        : updated U
# sv       : updated sv
#
# Example:
# from spmlib import solver as sps
# m, n = Y.shape
# U, sv = np.empty([0,0]), np.empty(0)  # initialize
# count = 0
# for j in range(n):
#    U, sv = sps.column_incremental_SVD(Y[:,j:j+1], U, sv, max_rank=50, orth_eps=linalg.norm(Y[:,j:j+1])*1e-12)
#    count = count + 1
#
def column_incremental_SVD(C, U, sv, forget=1., max_rank=np.inf, min_sv=0., orth_eps=1e-12, OrthogonalizeU=False):

    r = sv.size
    
    if r == 0:
        U, sv, V = linalg.svd(as2darray(C), full_matrices=False)
        return U, sv

    Cin = U.conj().T.dot(C)   # Cin = U' * C;
    Cout = C - U.dot(Cin)        # Cout = C - U * Cin;
    Cin, Cout = as2darray(Cin), as2darray(Cout)
    Q, R = linalg.qr(Cout, mode='economic')     # QR decomposition

    if linalg.norm(R) > orth_eps:
        # track moving subspace, rank can increase up to max_rank.
        # B = [f * diag(diagK), Cin; zeros(size(C,2), r), R];
        B = np.concatenate((
                np.concatenate((np.diag(forget * sv), Cin),1), 
                np.concatenate((np.zeros((as2darray(C).shape[1], r)), R), 1)))

        # [UB, sv, ~] = svd(full(B), 'econ'); sv = diag(sv);
        UB, sv, VB = linalg.svd(B, full_matrices=False)

        # rank must not be greater than max_rank, and the singlular values must be greater than min_sv
        r = min(max_rank, sv.size)       # r = min(max_rank, numel(sv));
        sv = sv[:r]                       # sv = sv(1:r);
        sv = sv[sv>=min_sv]                # sv = sv(sv >= min_sv);
        r = sv.size                      # ｒ = numel(sv);
        U = np.concatenate((U, Q), 1).dot(UB)    # U = [U, Q] * UB;
        U = U[:,:r]                     # U = U(:,1:r)
        if OrthogonalizeU:
            U = linalg.qr(U, mode='economic')[0]    # [U,~] = qr(U(:,1:r),0);

    else:
        # non-moving (rotating) subspace, rank non-increasing
        B = np.concatenate((np.diag(forget * sv), Cin),1)

        # [UB, sv, ~] = svd(full(B), 'econ'); sv = diag(sv);
        UB, sv, VB = linalg.svd(B, full_matrices=False)

        U = U.dot(UB)    # U = U * UB;
        if OrthogonalizeU:
            U = linalg.qr(U, mode='economic')[0]    # [U,~] = qr(U,0);
        

    #v = sp.diags(1/sv, format='csr').dot(U.conj().T.dot(Xc))

    return U, sv #, v



#%%
# Column incremental stable principal component pursuit (ColSPCP)
#
# l, s, U, sv = column_incremental_stable_principal_component_pursuit(c, U, sv, 
#                        l=None, s=None, rtol=1e-12, maxiter=1000,
#                        delta=1e-12, ls=1., rho=1., updateBasis=False,
#                        refineU_every=np.nan, forget=1., max_rank=np.inf, min_sv=0., orth_eps=1e-12, OrthogonalizeU=False,
#                        nesterovs_momentum=False, restart_every = np.nan,
#                        prox_ls=lambda q,r,c: prox.ind_l2ball(q,r,c),
#                        prox_l=lambda q,U: prox.squ_l2_from_subspace(q,1.,U), 
#                        prox_s=lambda q,ls: prox.l1(q,ls) ):
#
# performs the incremental stable principal component pursuit in
# S. Ogawa, H. Kuhara and T. Sakai, 
#    "Sequential decomposition of 3D apparent motion fields basedon low-rank and sparse approximation",
# APSIPA2017 (to appear).
#
#       ([l;s], [zls; zl; zs]) = arg min_(x,z)  g_ls(z_ls) + g_l(z_l) + g_s(z_s)
#                                   s.t. [zls; zl; zs] = [I I; I O; O I] * [l; s].
#
#       Here, by default,
#           g_ls(z_ls) = indicator function, i.e., zero if ||c - z_ls||_2 <= delta, infinity otherwise,
#           g_l(z_l) = 0.5 * ||(I-U*U')*z_l||_2^2,
#           g_s(z_s) = ||ls.*z_s||_1
#
#
# input:
# c        : m-dimensional vector to be decomposed into l and s such that ||d-(l+s)||_2<=delta
# U        : m x r matrix of left singular vectors approximately spanning the subspace of low-rank components
#            (overwritten with the update if updateBasis is True)
# sv       : r-dimensional vector of singular values
#            (overwritten with the update if updateBasis is True)
#
# l        : initial guess of l (None by default) # l = c - s
# s        : initial guess of s (None by default) # s = zeros
# rtol     : relative convergence torelance (1e-12 by default) of x and z in ADMM
# maxiter  : max. iterations (1000 by default)
# delta    : l2-ball radius used in the indicator function (1e-12 by default) for the approximation error
# ls       : weight of sparse regularizer (1. by default), can be a vector of weights for each pixel
# rho      : augmented Lagrangian parameter (1. by default)
#
# updateBasis : update U and sv with l after convergence (False by default)
# refineU_every  : update the basis U in the ADMM loop (disabled by default)
# forget   : forgetting parameter in updating U (1. by default)
# max_rank : maximum rank (np.inf by default)
# min_sv   : lower bound of singular values (0. by default)
# orth_eps : rank increases if the magnitude of c in the orthogonal subspace is larger than orth_eps (1e-12 by default)
# OrthogonalizeU : if True, perform QR decomposition to orthogonalize U (False by default)
#
# nesterovs_momentum: Nesterov acceleration (False by default)
# restart_every     : restart the Nesterov acceleration every this number of iterations (disabled by default)
#
# prox_ls  : prox. of regularizer g_ls for z_ls = l+s, g_ls = indicator function of l2ball by default,
#            i.e., lambda z_ls,r,c: prox.ind_l2ball(z_ls,r,c)
# prox_l   : prox. of regularizer g_l for z_l = l,
#            g_l = 0.5 * square l2 distance between l and span U by default,
#            i.e., lambda z_l,U: prox.squ_l2_from_subspace(z_l,1.,U)
# prox_s   : prox. of regularizer g_s for z_s = s, g_s = ||ls.*z_s||_1 by default,
#            i.e., lambda z_s,ls: prox.l1(z_s,ls)
#
#
# output:
# l        : low-rank component
# s        : sparse component
# U        : matrix of left singular vectors updated with l
# sv       : vector of singular values updated with l
#
#
# Example:
# from spmlib import solver as sps
# m, n = Y.shape
# U, sv = np.empty([0,0]), np.empty(0)  # initialize
# count = 0
# for j in range(n):
#    l, s, U, sv = sps.column_incremental_stable_principal_component_pursuit(Y[:,j:j+1], U, sv, ls=0.5, updateBasis=True,
#                                               max_rank=50, orth_eps=linalg.norm(Y[:,j:j+1])*1e-12)
#    count = count + 1
#
def column_incremental_stable_principal_component_pursuit(c, U, sv, 
                        l=None, s=None, rtol=1e-12, maxiter=1000,
                        delta=1e-12, ls=1., rho=1., updateBasis=False,
                        refineU_every=np.nan, forget=1., max_rank=np.inf, min_sv=0., orth_eps=1e-12, OrthogonalizeU=False,
                        nesterovs_momentum=False, restart_every = np.nan,
                        prox_ls=lambda q,r,c: prox.ind_l2ball(q,r,c),
                        prox_l=lambda q,U: prox.squ_l2_from_subspace(q,1.,U), 
                        prox_s=lambda q,ls: prox.l1(q,ls) ):
    
    m = c.shape[0]

    # initialize l and s
    if s is None:
        s = np.zeros(m)
    if l is None:
        l = c.ravel() - s
    
    if sv.size == 0:
        U, sv, V = linalg.svd(np.asmatrix(c.T).T, full_matrices=False)
        return l, s, U, sv

    # G = lambda x: np.concatenate((x[:m]+x[m:], x[:m], x[m:]))
    # x = np.concatenate((l,s))
    x = np.zeros(2*m)
    x[:m] = l
    x[m:] = s

    # z = G(x)
    z = np.zeros(3*m)
    z[:m]    = x[:m] + x[m:]
    z[m:2*m] = x[:m]
    z[2*m:]  = x[m:]

    y = np.zeros(3*m)
    
    t = 1.
    count = 0
    Ut = U
    while count < maxiter:
        count += 1

        if np.mod(count, restart_every) == 0:
            t = 1.
        if nesterovs_momentum:
            told = t
            t = 0.5 * (1. + sqrt(1. + 4. * t * t))

        # update x
        dx = x.copy()
        q = z - y
        x[:m] = (1./3.) * (q[:m] + 2.*q[m:2*m] - q[2*m:])
        x[m:] = (1./3.) * (q[:m] - q[m:2*m] + 2.*q[2*m:])
        dx = x - dx
        
        # q = G(x) + y
        q[:m]    = x[:m] + x[m:] + y[:m]
        q[m:2*m] = x[:m]         + y[m:2*m]
        q[2*m:]  = x[m:]         + y[2*m:]
        
        # update z
        if np.mod(count, refineU_every) == 0:
            Ut = column_incremental_SVD(x[:m], U, sv, 
                                           forget=forget, max_rank=max_rank, min_sv=min_sv,
                                           orth_eps=orth_eps, OrthogonalizeU=OrthogonalizeU)[0]
        dz = z.copy()
        z[:m]    = prox_ls(q[:m], delta, c.ravel())
        z[m:2*m] = prox_l(q[m:2*m], Ut)
        z[2*m:]  = prox_s(q[2*m:], ls)
        dz = z - dz

        # update y
        #y = y + G(x) - z
        dy = y.copy()
        y[:m]    += x[:m] + x[m:] - z[:m]
        y[m:2*m] += x[:m]         - z[m:2*m]
        y[2*m:]  += x[m:]         - z[2*m:]
        dy = y - dy

        # Nesterov acceleration
        if nesterovs_momentum:
            told = t
            t = 0.5 * (1. + sqrt(1. + 4. * t * t))
            z = z + ((told - 1.) / t) * dz
            y = y + ((told - 1.) / t) * dy
        
        # check convergence
        if linalg.norm(dx) < rtol * linalg.norm(x) and linalg.norm(dz) < rtol * linalg.norm(z):
            break
        
    l = x[:m]
    s = x[m:]
    if updateBasis:
        U, sv = column_incremental_SVD(l, U, sv, 
                                       forget=forget, max_rank=max_rank, min_sv=min_sv,
                                       orth_eps=orth_eps, OrthogonalizeU=OrthogonalizeU)

    return l, s, U, sv





'''
function [l, s, svdt, opt ] = ColwiseSPCP_ADMM(d, svdt, varargin)

% function [ l, s, svdt, opt ] = ColwiseSPCP_ADMM(d, svdt, varargin)
%
%     One step of online algorithm of low-rank and sparse matrix separation
%
%     x_out = arg min_x h(zOF) + 0.5 * ||(I-U*U')*zl||_2^2 + gs(zs)
%           s.t. [zOF; zl; zs] = [I I; I O; O I] * [l; s].
%     Here, h(zOF) = indicator function, i.e., zero if ||d - zOF|| <= epsilon, infinity otherwise.
%     and   gs(zs) = (1-alpha)*|| lambdas.*zs ||_1 + alpha*|| lambdas.*zs ||_2^2.
%
% input:
% d       : m-dimensional vector to be decomposed into l and s such that ||d-(l+s)||_2<=epsilon
% svdt.U  : m x r matrix (m >= r) of the orthogonal basis vectors for the low-rank component
% svdt.diagK: r x 1 vector of singular values
% varargin: structure with the following member
%        epsilon: ball radius used in the indicator function (1e-12 by default)
%        lambdas: weight of sparse regularizer (1 by default), can be a vector of weights for each pixel
%        alpha  : elasticity parameter (0<= alpha <= 1, 0 by default)
%        softs  : proximity operator of sparse regularizer for s (soft thresholding by default)
%        tol    : relative torelance (1e-12 by default)
%        iter   : max. iteration (1000 by default)
%        s0     : initial guess of s (zero vector by default)
%        rho    : augmented Lagrangian parameter (1 by default)
%        Nesterov: acceleration (false by default)
%        isUpdateU: update the basis U (true by default)
%        forget : forgetting parameter in updating U (1 by default)
%        maxRank: maximum rank (m by default)
%        meps   : epsilon not to increase ranks (1e-12 by default)
% %        minSV  : lower bound of singular values (0 by default)
%        refineUevery: update the basis U in the ADMM loop (0 by default)
%        debias : debias by s = d-l at the support of s (false by default)
%        %debias : debias by least squares solition with support of s (false by default)
%
% output:
% l       : low-rank component
% s       : sparse component
% svdt    : SVD components updated with l
% opt     : parameter set used in this function
%
% Example:
% for j=2:n, [L(:,j), S(:,j), svdt] = OnlineSRPCP_ADMM(D(:,j), svdt, 's0', S(:,j-1), 'alpha', 0.5, 'epsilon', 1e-2*norm(D(:,j)), 'maxRank', 6); end

m = size(d, 1);

%% given parameters
param = propertylist2struct(varargin{:});

%% set the parameters as default values if not given.
opt = set_defaults(param, struct('epsilon', 1e-12, 'lambdas', 1, 'alpha', 0, 'tol', 1e-12, 'iter', 1000, 's0', zeros(m,1), 'rho', 1, 'Nesterov', false, 'verbose', false));
opt = set_defaults(opt, struct('softs', @(x,tau) sign(x).*max(abs(x)-tau,0)));
opt = set_defaults(opt, struct('isUpdateU', true, 'forget', 1, 'maxRank', m, 'minSV', 0, 'meps', 1e-12,'refineUevery', 0, 'debias', false));

softs12e = @(x,w1,w2) opt.softs(x, w1) ./ (1+w2);
softs12 = @(x,w1,w2) opt.softs(x, w1) / (1+w2);
%soft21 = @(x,w1,w2) soft(x/(1+w2), w1/(1+w2));
%soft12 = @(x,w1,w2) prox_scad(x, w1, 3.7) / (1+w2);
%soft21 = @(x,w1,w2) prox_scad(x/(1+w2), w1/(1+w2), 3.7);
%soft12 = @(x,w1,w2) prox_OSCAR_APO(x, w1, w2);




while iter < opt.iter,

    %% update x
    v = z - u;
    x(1:m)     = (1/3) * (v(1:m) + 2*v(m+1:2*m) - v(2*m+1:3*m));
    x(m+1:end) = (1/3) * (v(1:m) - v(m+1:2*m) + 2*v(2*m+1:3*m));

    xpold = xp;
    xp = x;
    dx = xp - xpold;
    if opt.Nesterov,
        t_old = t; t = 0.5 * (1 + sqrt(1 + 4 * t_old * t_old));
        x = xp + (t_old - 1) / t * dx;
    end

    if mod(iter, opt.refineUevery) == 0,
%        opt.isOrthogonalize = false;
%        [U, ~] = updateU(svdt.U, svdt.diagK, x(1:m), opt);

%       Update with increasing the rank once, then throw away the most minor component.
        temp = incremental_singular_value_decomposition(isvdt, x(1:m), opt.maxRank);
        U = temp.U;
    end

    
    Gx = G(x);
    v = Gx + u;

    %% update z
    % zOF
    z(1:m) = proxIndL2(v(1:m), d, opt.epsilon);

    % zl
    z(m+1:2*m) = (opt.rho * v(m+1:2*m) + U*(U'*v(m+1:2*m))) / (1 + opt.rho);
    
    % zs
    if numel(opt.lambdas) ~= m,
        z(2*m+1:3*m) = softs12(v(2*m+1:3*m), opt.lambdas * (1-opt.alpha) / opt.rho, opt.lambdas * opt.alpha / opt.rho);
    else
        z(2*m+1:3*m) = softs12e(v(2*m+1:3*m), opt.lambdas * (1-opt.alpha) / opt.rho, opt.lambdas * opt.alpha / opt.rho);
    end

    %if opt.Nesterov,
        zpold = zp;
        zp = z;
        dz = zp - zpold;
    %    z = zp + (t_old - 1) / t * dz;
    %end

    
    %% update u
    u = u + Gx - z;

    %if opt.Nesterov,
    %    upold = up;
    %    up = u;
    %    du = up - upold;
    %    u = up + (t_old - 1) / t * du;
    %end

    
    if opt.verbose,
        xmax = max(max(abs(x(m+1:end))), 1);
        figure(98), plot(real(x(m+1:end)), '.'); ylim([-xmax, xmax]); title(num2str(iter));
    end
    
    if (norm(dx) / norm(x) < opt.tol) && (norm(dz) / norm(z) < opt.tol),
        break;
    end

    iter = iter + 1;
end

l = x(1:m);
s = x(m+1:end);
%e = d - l - s;
%idxz = find(z(2*m+1:3*m)==0); s(idxz) = 0;
%idxnz = find(z(2*m+1:3*m)~=0);
%s(idxnz) = s(idxnz) + e(idxnz);% e(idxnz) = 0;
if opt.isUpdateU,
%    opt.isOrthogonalize = true;
%    [svdt.U, svdt.diagK] = updateU(svdt.U, svdt.diagK, l, opt);

        temp = incremental_singular_value_decomposition(isvdt, l, opt.maxRank);
        svdt.U = temp.U; svdt.diagK = diag(temp.K);
end

if opt.debias,
    idxnz = find(s~=0); e = d - l - s; s(idxnz) = s(idxnz) + e(idxnz); e(idxnz) = 0;    
    %s = debias(s, d, svdt.U, struct('tol_lscg', opt.tol, 'iter_lscg', opt.iter));
end

end

'''

