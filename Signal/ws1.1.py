import numpy as np
import matplotlib.pyplot as plt
import cmath

def fft(x):
    N = len(x)
    if N <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [cmath.exp(-2j * np.pi * k / N) for k in range(N // 2)]
    return [even[k] + T[k] * odd[k] for k in range(N // 2)] + [even[k] - T[k] * odd[k] for k in range(N // 2)]

def ifft(X):
    N = len(X)
    if N <= 1:
        return X
    even = ifft(X[0::2])
    odd = ifft(X[1::2])
    T = [cmath.exp(2j * np.pi * k / N) for k in range(N // 2)]
    return [(even[k] + T[k] * odd[k]) / 2 for k in range(N // 2)] + [(even[k] - T[k] * odd[k]) / 2 for k in range(N // 2)]


def signal(x):
    return np.cos(5 * x) + np.sin(6 * x)

N = 1024 
x = np.linspace(0, 50, N)
y = signal(x)

plt.figure(figsize=(14, 12))

#signal
plt.subplot(4, 1, 1)
plt.plot(x, y)
plt.title('Original signal')
plt.xlabel('x')
plt.ylabel('y')

#fft
Y = fft(y)

#frequency
sample_rate = len(y)
frequency = np.array([k * sample_rate / N if k <= N / 2 else (k - N) * sample_rate / N for k in range(N)])

#real part
plt.subplot(4, 1, 2)
plt.plot(frequency, [val.real for val in Y])
plt.title('FFT result')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Real Part')

#magnitude
plt.subplot(4, 1, 3)
plt.plot(frequency, [np.abs(val) for val in Y])
plt.title('Magnitude spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

#ifft
y_reconstructed = ifft(Y)

#signal
plt.subplot(4, 1, 4)
plt.plot(x, [val.real for val in y_reconstructed])
plt.title('Reconstructed Signal')
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()

plt.show()
