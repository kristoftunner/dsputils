# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 13:28:15 2020

@author: usr_tunnerk
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.pyplot import figure
from scipy import signal

#constants
Ts = 10*1e-9 #100M sampling frequency


#samples, smpl_orig are with the original impedance matching network
smpl_subdir = '7dbm'
smpl_orig = {'10M':'data/'+smpl_subdir+'/xofs_10Msig_o.csv',
           '14M':'data/'+smpl_subdir+'/xofs_14Msig_o.csv',
           '18M':'data/'+smpl_subdir+'/xofs_18Msig_o.csv',
           '22M':'data/'+smpl_subdir+'/xofs_22Msig_o.csv',
           '26M':'data/'+smpl_subdir+'/xofs_26Msig_o.csv',
           '30M':'data/'+smpl_subdir+'/xofs_30Msig_o.csv',
           '34M':'data/'+smpl_subdir+'/xofs_34Msig_o.csv',
           '38M':'data/'+smpl_subdir+'/xofs_38Msig_o.csv',
           '42M':'data/'+smpl_subdir+'/xofs_42Msig_o.csv',
           '46M':'data/'+smpl_subdir+'/xofs_46Msig_o.csv',
           '50M':'data/'+smpl_subdir+'/xofs_50Msig_o.csv',
           '52M':'data/'+smpl_subdir+'/xofs_52Msig_o.csv'}


def dB(x):
    return 20*np.log10(np.abs(x))

def read_data(smpl_key):
    shape_i = (len(smpl_key),65536,6)
    shape_o = (len(smpl_key),65536,2)
    data_i = np.zeros(shape_i)
    data_o = np.zeros(shape_o)
    for key,j in zip(smpl_orig, range(0,len(smpl_orig)+1,1)):
        data_i[j] = np.genfromtxt(smpl_orig[key], delimiter=',',dtype=None, skip_header=2)
        for i in range(0,np.size(data_i,0)):
            data_o[i] = np.delete(data_i[i],np.s_[1:5],axis=1)
    return data_o
        

#FFT plotting
def fft_plot(data,smpl_key):
    for i, item in enumerate(data):
        fft = dB(np.fft.fftshift(np.fft.fft(item[:,1])))
        fft = fft-max(fft)
        fig = figure()
        keys = list(smpl_key.keys())
        plt.plot(np.fft.fftshift(np.fft.fftfreq(n=len(fft), d=1/(100))),fft)
        plt.xticks(np.arange(-56,56,step=2))
        plt.title(keys[i]+'Hz signal, Fs=100M')

#interpolation
def sinc_interp(data, smpl_key):
    t1 = np.arange(0, 500*Ts, Ts, dtype=float)
    t2 = np.arange(0, 500*Ts, Ts/20, dtype=float)
    s = np.arange(0, len(t1), dtype=int)
    shape = (len(smpl_key),len(t2))
    y = np.zeros(shape)
    keys = list(smpl_key.keys())
    for j, item in enumerate(data):
        u = item[0:501,1]
        for i in range(len(t2)):
            sincM = (t2[i] - s*Ts) / Ts
            y[j,i] = np.dot(u, np.sinc(sincM))
        fig = figure()
        plt.title(keys[j]+'Hz signal, Fs=100M')
        plt.plot(t1, u, 'b-', t2, y[j,:], 'r-', label='interpolated')
        
data = read_data(smpl_orig)
sinc_interp(data,smpl_orig)
#fft_plot(data, smpl_orig)


