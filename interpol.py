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

def sinc_interp(x, t1, t2):
    s = np.arange(0,np.size(t1,0),1, dtype=int)
    y = np.arange(0, len(t2), dtype=float)
    for i in range(len(t2)):
        sincM = (t2[i] - s*Ts) / Ts
        y[i] = np.dot(x, np.sinc(sincM))
    return y

def dB(x):
    return 20*np.log10(np.abs(x))

#samples, smpl_orig are with the original impedance matching network
#
smpl_orig = {'10M':'data/xofs_10Msig_o.csv',
           '14M':'data/xofs_14Msig_o.csv',
           '18M':'data/xofs_18Msig_o.csv',
           '22M':'data/xofs_22Msig_o.csv',
           '26M':'data/xofs_26Msig_o.csv',
           '30M':'data/xofs_30Msig_o.csv',
           '34M':'data/xofs_34Msig_o.csv',
           '38M':'data/xofs_38Msig_o.csv',
           '42M':'data/xofs_42Msig_o.csv',
           '46M':'data/xofs_46Msig_o.csv',
           '50M':'data/xofs_50Msig_o.csv'}
           #1'54M':'data/xofs_54Msig_o.csv'}


#interpolating
shape_i = (len(smpl_orig),65536,6)
shape_o = (len(smpl_orig),65536,2)
data_i = np.zeros(shape_i)
data_o = np.zeros(shape_o)
t1 = np.arange(0, 500*Ts, Ts, dtype=float)
t2 = np.arange(0, 500*Ts, Ts/20, dtype=float)
shape_y = (len(smpl_orig), len(t2))
y = np.zeros(shape_y)
j = 0

for key in smpl_orig:
    data_i[j] = np.genfromtxt(smpl_orig[key], delimiter=',',dtype=None, skip_header=2)
    for i in range(0,np.size(data_i,0)-1):
        data_o[i] = np.delete(data_i[i],np.s_[1:5],axis=1)
    u = data_o[j,500:1000,1]
    # y[j] = sinc_interp(u, t1, t2)
    # fig = figure()
    # plt.title(key+'Hz signal, Fs=100M')
    # plt.plot(t1, u, 'b-', t2, y[j], 'r-', label='interpolated')
    j += 1

#FFT
for i, item in enumerate(data_o):
    fft = dB(np.fft.fftshift(np.fft.fft(item[:,1])))
    fft = fft-max(fft)
    fig = figure()
    keys = list(smpl_orig.keys())
    plt.plot(np.fft.fftshift(np.fft.fftfreq(n=len(fft), d=1/(100))),fft)
    plt.xticks(np.arange(-50,50,step=2))
    plt.title(keys[i]+'Hz signal, Fs=100M')


