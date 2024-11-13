import numpy as np
import matplotlib.pyplot as plt


def FFT(x):
    N = len(x)
    if N <= 1:
        return x
    even = FFT(x[0::2])
    odd = FFT(x[1::2])
    W = [np.exp(-2j * np.pi * k / N) for k in range(N // 2)]
    return [even[k] + W[k] * odd[k] for k in range(N // 2)] + [even[k] - W[k] * odd[k] for k in range(N // 2)]


def blackman_window(numtaps):
    n = np.arange(numtaps)
    window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (numtaps - 1)) + 0.08 * np.cos(4 * np.pi * n / (numtaps - 1))
    return window


def firwin(numtaps, cutoff):
    taps = np.sinc(2 * cutoff * (np.arange(numtaps) - (numtaps - 1) / 2)) * blackman_window(numtaps)
    return taps


N = 1024

x = np.linspace(0, 2*np.pi, N)
y_original = np.cos(2*x) + np.sin(5*x)

x1 = np.linspace(0, 2*np.pi, 32)
y_original1 = np.cos(2*x1) + np.sin(5*x1)

y_noise = np.cos(2*x) + np.sin(5*x) + np.cos(100*x)

y_noise1 = np.cos(2*x1) + np.sin(5*x1) + np.cos(100*x1)

count = 128
cutoff = 5 / count
b = firwin(count, cutoff)

y_filtered = np.convolve(y_noise, b, mode='same')

freq_response = (np.fft.fft(b))

plt.figure(figsize=(12, 6))

plt.subplot(4, 2, 1)
plt.plot(x, y_original)
plt.title('Исходный график')

plt.subplot(4, 2, 2)
#plt.plot(x, np.fft.fft(y_original))
plt.stem(range(1024), np.fft.fft(y_original))
plt.title('Исходный график АЧХ')

plt.subplot(4, 2, 3)
plt.plot(x, y_noise)
plt.title('График с шумом')

plt.subplot(4, 2, 4)
#plt.plot(x, FFT(y_noise))
plt.stem(range(1024), np.fft.fft(y_noise))
plt.title('График с шумом АЧХ')

plt.subplot(4, 2, 5)
plt.plot(x, y_filtered)
plt.title('График после использования КИХ фильтра')

plt.subplot(4, 2, 6)
#plt.plot(x1, FFT(y_filtered))
plt.stem(range(1024), np.fft.fft(y_filtered))
plt.title('График после использования КИХ фильтра АЧХ')

plt.subplot(4, 2, 8)
plt.plot(np.abs(freq_response))
plt.title('АЧХ')

plt.tight_layout()
plt.show()