
import numpy as np

def gen_sine_real(fs,fin,vpp,N):
    """
    Generate sine wave with only real component

    Parameters
    ----------
    fs : sampling freqeuncy
    fin : freqeuncy of generated signal
    vpp : amplitude of generated signal
    N : number of discrete samples of the generated signal

    Returns
    -------
    dt : ndarray
        time resolution of the generated signal
    x : ndarray
        generated signal
    """
    T = 1/fs
    dt = np.arange(0.0,N*T, T)
    return dt,vpp*np.exp(1j*2*np.pi*fin*dt).real

def gen_sine_compl(fs,fin,vpp,N):
    """
    Generate sine wave with only real component

    Parameters
    ----------
    fs : sampling freqeuncy
    fin : freqeuncy of generated signal
    vpp : amplitude of generated signal
    N : number of discrete samples of the generated signal

    Returns
    -------
    dt : ndarray
        time resolution of the generated signal
    x : ndarray
        generated complex sinusoid signal
    """
    T = 1/fs
    dt = np.arange(0.0,N*T, T)
    return dt,vpp*np.exp(1j*2*np.pi*fin*dt)