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

    #if 4 channel give error
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("WAV file must have exactly 4 channels")
    
    return fs, data.T

def bandpass(sig, fs, f0, bw=2000):
    low = (f0 - bw/2) / (fs/2)
    high = (f0 + bw/2) / (fs/2)
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, sig)


def tdelay(sig, ref, fs, maxd=0.2, c=1500):
    maxlag = int((maxd / c) * fs)

    n = len(sig) + len(ref)
    SIG = fft(sig, n)
    REF = fft(ref, n)

    R = SIG * np.conj(REF)
    R =R/(np.abs(R) + 1e-12)

    corr=np.real(ifft(R))
    corr=np.concatenate((corr[-len(ref)+1:], corr[:len(sig)]))

    mid=len(corr)//2
    window=corr[mid-maxlag:mid+maxlag]

    i = np.argmax(window)

    # breaking into sub-samples usng parabolic interpolation 
    if 0 < i < len(window)-1:
        y0,y1,y2=window[i-1],window[i], window[i+1]
        denom= (y0 - 2*y1 + y2)
        frac= 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        frac = 0.0

    lag=(i-maxlag+frac)
    return lag/fs

def doa(signals, fs, positions, c=1500.0):
    pairs= []
    delays= []
    A =[]

    for i in range(4):
        for j in range(i+1, 4):
            dt = tdelay(signals[i], signals[j], fs)
            delays.append(dt)
            A.append(positions[j] - positions[i])

    A=np.array(A)
    b=c*np.array(delays)

    # Least squares solve
    u, *_ =np.linalg.lstsq(A, b, rcond=None)

    # Normalize direction vector
    u/=np.linalg.norm(u)

    az =np.degrees(np.arctan2(u[1], u[0]))
    el=np.degrees(np.arcsin(u[2]))

    return az, el, u


if __name__ == "__main__":
    d=0.01
    positions =  np.array([
        [0.0,      0.0,      0.0],
        [d,        0.0,      0.0],
        [0.5*d,    np.sqrt(3)/2*d, 0.0],
        [0.5*d,    np.sqrt(3)/6*d, np.sqrt(2/3)*d]
    ])
    
    
    fs,signals=readwav4("hydrophones_tetrahedral_30kHz.wav")

    signals=np.array([
        bandpass(ch,fs,30000) for ch in signals
    ])

    az,el,u=doa(signals,fs,positions)
    print(f"Azimuth: {az:.2f}°, Elevation: {el:.2f}°")

    
