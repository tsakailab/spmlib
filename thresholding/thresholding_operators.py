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
    return sgn(z) * np.maximum(np.abs(z) - th, 0.)


#%% smoothly clipped absolute deviation (SCAD) [Fan&Li, 01]
def smoothly_clipped_absolute_deviation(z, th, a=3.7):
    scad = z.copy()
    absz = np.abs(z)
    idx = absz <= 2.*th
    scad[idx] = soft(z[idx], th)
    idx = np.logical_and(absz > 2.*th, absz <= a*th)
    scad[idx] = ((a - 1) * z[idx] - sgn(z[idx]) * a * th) / (a - 2)
    return scad


#%% hard thresholding
def hard(z, th):
    return z * (np.abs(z)>th)


#%% soft thresholding function compatible with complex values
def soft_svd(Z, th, thresholding=lambda z,th: soft(z, th)):
    U, s, V = linalg.svd(Z,full_matrices=False)
#    return U.dot(np.diag(np.maximum(np.abs(s) - th, 0.)).dot(V)), U, s, V
    s = thresholding(s, th)
#    s = soft_thresh(s, th)
#    s = smoothly_clipped_absolute_deviation(s, th)
    return U.dot(sp.diags(s).dot(V)), U, s, V
