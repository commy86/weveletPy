import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

def theta(f, number):
    theta = np.linspace(0, 2*f*np.pi,number)
    return theta

f1, f10, f100 = 1, 10, 100
number = 1001

x = np.linspace(0, 1, number)
z = np.sin(theta(f1, number)) + np.sin(theta(f10, number)) + np.sin(theta(f100, number))
y = np.linspace(1,100, 100)

taps = 100

z2 = sig.cwt(z, sig.ricker, y)

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1)

ax1.text(0, 0, "CWT", fontsize="large")
ax1draw = ax1.matshow(z2, cmap=plt.cm.gray)

minmax = np.linspace(np.min(z2), np.max(z2),100)
#cax = ax1.matshow(ax1draw, interpolation='nearest')
plt.colorbar(ax1draw)

ax2.plot(x, z)
ax2.text(0, 0, "Input data of CWT")

ax3.plot(x, np.sin(theta(f1, number)))
ax3.text(0, 0, "1Hz")

ax4.plot(x, np.sin(theta(f10, number)))
ax4.text(0, 0, "10Hz")

ax5.plot(x, np.sin(theta(f100, number)))
ax5.text(0, 0, "100Hz")

plt.show()