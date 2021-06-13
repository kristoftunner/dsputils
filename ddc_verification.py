#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 20:31:01 2020

@author: kristof
"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

#%%
def decimate(data,decNum):
    dataDec = np.zeros(len(data)//7)
    for i in range(len(dataDec)):
        dataDec[i] = data[i*7]
    return dataDec

def avg(data,window):
    dataAvg = np.zeros(len(data)-window)
    for i in range(window//2, len(x)-window//2):
        accu = 0
        for j in range(-window//2,window//2):
            accu += x[i+j]
        dataAvg[i-window//2-1] = accu/window
    return dataAvg

def plot_fft(data,color,label,fs,figure):
    f,Pper_spec = signal.periodogram(data,fs/7,'flattop',scaling='spectrum',return_onesided=False)
    fig = plt.figure(figure)
    ax = fig.add_subplot(1,1,1)
    ax.semilogy(f,Pper_spec,color,label=label)
    ax.grid()
    fig.show()
#%%
fs = 64e6
bins = 10
fsignal = np.zeros(bins,dtype=int)
for i in range(0,bins-1):
    fsignal[i] = 44e6+i*8e3

phi = 0
t = np.arange(0,1/fs*100000, 1/fs)
x = np.zeros(len(t))
for i in range(0,bins-1):
    x = x + np.exp(1j*2*np.pi*fsignal[i]*t).real

floc = 16e6
local = np.exp(1j*2*np.pi*floc*t).real + (np.exp(1j*2*np.pi*floc*t)).imag * -1

ddc = local*x
plot_fft(ddc,'r','ddc signal',fs,0)
avg = avg(ddc,7)
plot_fft(avg,'b','average signal',fs,0)
dec = decimate(avg,7)
plot_fft(dec,'y','decimated',fs,0)
#%%

#%%
f,Pper_spec = signal.periodogram(dec,fs/7,'flattop',scaling='spectrum',return_onesided=False)
fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
ax.semilogy(f,Pper_spec,label='signal in')
ax.grid()
fig.show()
