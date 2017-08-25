# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 08:42:11 2017

@author: tsakai
"""

import numpy as np
import spmlib.thresholding as th

# Proximity operators:
#     prox_(phi,w)(v) = w*arg min_z phi(z) + 1/2 || z - v ||_2^2
#

#%%
# arg min ||l.*z||_2^2 + 0.5*||z-q||_2^2 = q ./ (2*l+1)
def l2(q, l):
    return q / (2*l+1)



#%%
# arg min ||l.*z||_1 + 0.5*||z-q||_2^2
def l1(q, l):
    return th.soft(q, l)



#%%
# arg min |||l.*z||_0 + 0.5*||z-q||_2^2 = q .* (q>sqrt(2*l))
def l0(q, l):
    return th.hard(q, np.sqrt(2*l))


#%%
# arg min l*||Z||_* + 0.5*||Z-Q||_F^2
def nuclear(Q, l):
    return th.soft_svd(Q, l)


