import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

import pywt

dt = 0.001
fs = 1/dt
nq_f = fs/2.0 
t = np.arange(0, 1, dt)
N = t.size

freqs = np.linspace(1,nq_f,500)

sig = (np.exp(2*np.pi*1j*(N/2**2)*t) * ((0/8<=t) & (t<1/8)) +
       np.exp(2*np.pi*1j*(N/2**3)*t) * ((1/8<=t) & (t<2/8)) +
       np.exp(2*np.pi*1j*(N/2**4)*t) * ((2/8<=t) & (t<3/8)) +
       np.exp(2*np.pi*1j*(N/2**5)*t) * ((3/8<=t) & (t<4/8)) +
       np.exp(2*np.pi*1j*(N/2**6)*t) * ((4/8<=t) & (t<5/8)) +
       np.exp(2*np.pi*1j*(-N/2**2)*t) * ((5/8<=t) & (t<6/8)) +
       np.exp(2*np.pi*1j*(-N/2**3)*t) * ((6/8<=t) & (t<7/8)) +
       np.exp(2*np.pi*1j*(-N/2**4)*t) * ((7/8<=t) & (t<8/8))).astype('float64')
sig = pywt.data.demo_signal('MishMash', t.size)
sig = chirp(t, f0=10, f1=500, t1=2, method='linear')
N2 = N // 2

j = 1 if N2*2 == N else 0
f = np.concatenate((np.arange(0, N2+1), np.arange(-N2+1-j, 0))) / N  # 周波数
MST = np.zeros((N2+1, N))  # 正の周波数
SIG = np.fft.fft(sig, N)  # FFT

print(f)

const = [1, 2, 4, 8]

for j, k in enumerate(const):
    gamma = k * np.var(sig)  # パラメータ γ
    for i in range(1, N2):
        SIGs = np.roll(SIG, -i+1)  # circularly shift the spectrum SIG
        W = (gamma / f[i]) * 2 * np.pi * f  # scale Gaussian
        G = np.exp(-W**2 / 2)  # W in Fourier domain
        G = G[:N]  # truncate G to match SIGs
        MST[i] = np.fft.ifft(SIGs * G)  # compute the complex values of MST

    plt.figure()
    plt.imshow(np.abs(MST), aspect='auto', origin='lower', cmap = 'jet') # plot the magnitude of MST origin='lower'
    plt.colorbar()
    plt.suptitle('gamma = {0}'.format(gamma))

plt.figure()
plt.xlabel("Time[s]")
plt.ylabel("Signal")
plt.plot(sig)

plt.show()
