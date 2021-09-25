"""
    Dsputils test functions
"""

import matplotlib.pyplot as plt
import freq_domain
import signal_gen

if __name__ == '__main__':
    fs = 10
    fin = 0.3
    dt,x = signal_gen.gen_sine_real(fs,fin,10,10000)
    df,fft = freq_domain.rfft(x,fs)
    