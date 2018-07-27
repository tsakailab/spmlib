#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 15:37:33 2017

@author: tmiura
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as swf
from scipy.fftpack import dct, idct, next_fast_len
from pywt import wavedec, waverec, coeffs_to_array, array_to_coeffs
import spmlib.solver.relaxations as relax
import scipy.linalg as linalg


def dwt(data, wavelet, mode='per', level=None):
    # returns (coeff_arr, coeff_slices)
    return coeffs_to_array(wavedec(data, wavelet, mode, level))

def idwt(coeff_arr_slices, wavelet, mode='per'):
    # returns data
    return waverec(array_to_coeffs(coeff_arr_slices[0], coeff_arr_slices[1], output_format='wavedec'), wavelet, mode)

def signal_reconst_dctwt(coeffs, coeff_slices, wavelet='db10', wl_weight=0.5):
    n = coeffs.shape[0] // 2
    return idct(coeffs[:n], norm='ortho') +  wl_weight * idwt([coeffs[n:], coeff_slices], wavelet)

def signal_decomp_dctwt(signal, wavelet='db10', level=None, wl_weight=0.5):
    # returns (coeffs, coeff_slices)
    coeff_arr, coeff_slices = dwt(signal, wavelet, level=level)
    return np.concatenate((dct(signal, norm='ortho'), wl_weight*coeff_arr), axis=0), coeff_slices


    
def read_file(filename):
    #read
    if os.path.exists(filename + '.wav') == False:
        Fs, data  = swf.read(filename + '.WAV')
    else:
        Fs, data = swf.read(filename + '.wav')
    
    #normalize between -1 and 1, change dtype double
    data = data.astype(np.float64)
    data = data / 32768.0
    print(data.shape)
    return Fs,data
    
def run_LSdecompFW(filename, width = 16384, max_nnz_rate = 8000 / 262144, level=3,
                 sparsify = 0.01, wavelet='db10', wl_weight = 0.25, verbose = False, fc = 120, maxiter=60):        

    Fs,signal = read_file(filename)
    
    length = signal.shape[0]    
    signal = np.concatenate((np.zeros(width//2), signal[0:length], np.zeros(width)),axis=0)
    n_wav = signal.shape[0]

    
    signal_dct = np.zeros(n_wav,dtype=np.float64)
    signal_wl = np.zeros(n_wav,dtype=np.float64)

    pwindow = np.zeros(width, dtype=np.float64)
    for i in range(0,width // 2):
        pwindow[i] = i
    for i in range(0, width // 2 + 1):
        pwindow[-i] = i-1
    pwindow = pwindow / width * 2
    
    #if fc > 0:
        #d = fdesign.hipass()
    
    w_s = 0
    w_e = width

    START = np.empty(n_wav)
    END = np.empty(n_wav)
    i = 0
    
    while w_e < n_wav:
        
        print('\n%1.3f - %1.3f [s]\n' % (w_s/Fs, w_e/Fs))
        START[i] = w_s / Fs
        END[i] = w_e / Fs
        sig_dct, sig_wl,c ,c_list= LSDecompFW(wav=signal[w_s:w_e], width = width, level=level,
                                         max_nnz_rate = max_nnz_rate, sparsify = sparsify, wavelet = wavelet, 
                                         wl_weight = wl_weight, verbose = verbose, fc = fc, maxiter=maxiter )
        signal_dct[w_s:w_e] = signal_dct[w_s:w_e] + pwindow * sig_dct
        signal_wl[w_s:w_e] = signal_wl[w_s:w_e] + pwindow * sig_wl
        
        if w_e/Fs > length/Fs:
            break
        w_s = w_s + width // 2
        w_e = w_e + width // 2
        i+=1
        
    #dct_length = np.shape(signal_dct)[0]
    #wl_length = np.shape(signal_wl)[0]
    #raw_length = np.shape(signal)
   
    signal_dct = signal_dct[width//2:length+width//2]
    signal_wl = signal_wl[width//2:length+width//2]    
    #signal_raw = signal[width/2+1:raw_length]
    
    #signal_dct =np.round((signal_dct+1)*65535/2) - 32768.0
    signal_dct = signal_dct*32768.0
    signal_dct = signal_dct.astype(np.int16)
    
    #signal_wl =np.round((signal_wl+1)*65535/2) - 32768.0    
    signal_wl = signal_wl*32768.0
    signal_wl = signal_wl.astype(np.int16)
    
    swf.write(filename +'_F.wav',Fs,signal_dct)
    swf.write(filename +'_W.wav',Fs,signal_wl)
    
    print('end of run_LSdecomp_FW!!')
    return signal_dct,signal_wl,c,c_list
#max_nnz_rate=8000.0/262144.0

def LSDecompFW(wav, width= 16384, max_nnz_rate=0.03, sparsify = 0.01, eta = 2, wavelet='db10', 
               level = 3, wl_weight = 0.5, verbose = False, maxiter=60, fc=120):
    
   
    length = len(wav)
    
    #print('wl_weight: '+str(wl_weight)+"\n")
    n = next_fast_len(length)

    signal = np.zeros((n))
    signal[0:length] = wav[0:length]

    cnnz = float("Inf")
    c, slices = signal_decomp_dctwt(signal, wavelet, level, wl_weight)    
    c_list = []

    fA = lambda x: signal_reconst_dctwt(x, slices, wavelet, wl_weight)   
    fAT = lambda y: signal_decomp_dctwt(y, wavelet, level, wl_weight)[0]
    fh = (fA, fAT)

    maxabsThetaTy = max(abs(c))
    L = 2*linalg.norm(fh[0](fh[1](signal))) / linalg.norm(signal)
      
    while cnnz > max_nnz_rate * n:
            #FISTA
            #maxabsThetaTy = max(abs(c))
            #print('maxabs: '+str(maxabsThetaTy))
            tau = sparsify * maxabsThetaTy
            tolA = 1.0e-7
            
            c, r = relax.fista(A=fh, b=signal, tol=tolA, l=tau, maxiter=maxiter, L=L)[:2]

            c_list.append(c)
            cnnz = np.count_nonzero(c)
            cnnz_dct = np.count_nonzero(c[:n])
            cnnz_wl = np.count_nonzero(c[n:])
            #maxabsThetaTy = max(abs(c))
            print('nnz = ('+ str(cnnz_dct) + '+' + str(cnnz_wl) + ') / ' + str(n) +' at tau = '+str(tau))
            sparsify = sparsify * eta
            L *= eta
            
    print('%1.2f' % (linalg.norm(r)/linalg.norm(signal)))


    signal_dct = idct(c[:n], norm='ortho')
    signal_wl = wl_weight * idwt((c[n:], slices), wavelet)

    return  signal_dct,signal_wl,c,c_list
    ###############################
if __name__ == '__main__':
#    filepath = './080180500_5k'
    filepath = './TRACK63_11k'
    signal_dct,signal_dwt,c,c_list=run_LSdecompFW(filename = filepath, wl_weight=0.5, level=3, maxiter=60, max_nnz_rate=0.5)
    
