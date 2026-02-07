import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from scipy.fft import fft, fftfreq,ifft
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from scipy.io import wavfile

def readwav4(fname):
    fs, data = wavfile.read(fname)
    # Convert audio to float in [-1, 1] range if integer
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float32)

    #if not 4 channel give error
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("WAV file must have exactly 4 channels")

    return fs, data.T

def bandpass(sig, fs, f0, bw=2000):
    low = (f0 - bw/2) / (fs/2)
    high = (f0 + bw/2) / (fs/2)
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, sig)


def tdelay(sig, ref, fs, maxd=0.2, c=1500):
    maxlag=int((maxd / c) * fs)

    n=len(sig) + len(ref)
    SIG=fft(sig, n)
    REF=fft(ref, n)

    R=SIG*np.conj(REF)
    R=R/( np.abs(R) + 1e-12)

    corr=np.real(ifft(R))
    corr=np.concatenate((corr[-len(ref)+1:], corr[:len(sig)]))

    mid=len(corr)//2
    window=corr[mid-maxlag:mid+maxlag]

    lag=np.argmax(window)-maxlag
    return lag/fs


def doa(signals, fs, positions, c=1500.0):
    ref = signals[0]
    delays = [0.0]
    for i in range(1, signals.shape[0]):
        delays.append(tdelay(signals[i], ref, fs))
    delays = np.array(delays)

    r_ref=positions[0]
    A=positions[1:] - r_ref
    b=c*delays[1:]

    u, *_=np.linalg.lstsq(A, b, rcond=None)
    u=u/np.linalg.norm(u)

    el=180+np.degrees(np.arctan2(u[1], u[0]))
    az=-np.degrees(np.arcsin(u[2]))
    return az, el, u

if __name__ == "__main__":
    d=0.01
    positions =  np.array([
        [0.0,      0.0,      0.0],
        [d,        0.0,      0.0],
        [0.5*d,    np.sqrt(3)/2*d, 0.0],
        [0.5*d,    np.sqrt(3)/6*d, np.sqrt(2/3)*d]
    ])
    
    
    fs, signals = readwav4("hydrophones_tetrahedral_30kHz.wav")

    signals = np.array([
        bandpass(ch, fs, 30000) for ch in signals
    ])

    az, el, u = doa(signals, fs, positions)
    print(f"Azimuth: {az:.2f}°, Elevation: {el:.2f}°")

    
