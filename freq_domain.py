
import numpy as np

def fft(x,fs):
    """
    return the complex normalized FFT of the signal

    Parameters
    ----------
    x : input signal
    fs : sampling frequency
    
    Returns
    -------
    df : ndarray
        frequency resolution for the x axes
    nfft : complex ndarray
        generated FFT of the signal
    """
    nfft = np.fft.fftshift(np.fft.fft(x))/len(x)
    df = np.fft.fftshift(np.fft.fftfreq(n=len(x),d=1/fs))
    return df,nfft

def rfft(x,fs):
    """
    return the complex normalized one-sided FFT of the signal

    Parameters
    ----------
    x : input signal
    fs : sampling frequency
    
    Returns
    -------
    df : ndarray
        frequency resolution for the x axes
    nfft : complex ndarray
        generated FFT of the signal
    """
    nfft = np.fft.rfft(x)/(len(x)/2)
    df = np.fft.rfftfreq(n=len(x),d=1/fs)
    return df,nfft
    

