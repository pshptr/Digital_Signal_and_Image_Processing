import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

def signal(x):
    return np.cos(5 * x) + np.sin(6 * x)

N = 1024
x = np.linspace(0, 50, N)
y = signal(x)

plt.figure(figsize=(14, 12)) 

plt.subplot(4, 1, 1)
plt.plot(x, y)
plt.title('Signal')
plt.xlabel('x')
plt.ylabel('y')

Y = fft(y)
frequency = np.fft.fftfreq(len(y), d=(x[1]-x[0]))

plt.subplot(4, 1, 2)
plt.plot(frequency, Y.real)
plt.title('FFT result')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Real Part')

plt.subplot(4, 1, 3)
plt.plot(frequency, np.abs(Y))
plt.title('Magnitude Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

y_reconstructed = ifft(Y)

plt.subplot(4, 1, 4)
plt.plot(x, y_reconstructed.real)
plt.title('Reconstructed signal')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()

plt.show()
