import numpy as np
import matplotlib.pyplot as plt

def haar_wavelet_transform(signal, level):
    coeffs = []
    for _ in range(level):
        length = len(signal)
        if length == 1:
            coeffs.append(signal)
            break

        half_length = length // 2
        low_pass = np.zeros(half_length)
        high_pass = np.zeros(half_length)

        for i in range(half_length):
            low_pass[i] = (signal[2*i] + signal[2*i + 1]) / np.sqrt(2)
            high_pass[i] = (signal[2*i] - signal[2*i + 1]) / np.sqrt(2)

        coeffs.append(high_pass)
        signal = low_pass

    coeffs.append(signal)
    coeffs.reverse()
    print("Коэффициенты вейвлет-преобразования:")
    print(coeffs)
    return coeffs


def inverse_haar_wavelet_transform(coeffs):
    if not isinstance(coeffs, list):
        raise ValueError("Coefficients must be provided as a list")

    level = len(coeffs) - 1

    signal = coeffs[0]
    for i in range(1, level + 1):
        length = len(coeffs[i])
        reconstructed_signal = np.zeros(length * 2)
        for j in range(length):
            reconstructed_signal[2*j] = (signal[j] + coeffs[i][j]) / np.sqrt(2)
            reconstructed_signal[2*j + 1] = (signal[j] - coeffs[i][j]) / np.sqrt(2)
        signal = reconstructed_signal
    print("Восстановленная функция:")
    print(signal)
    return signal


N = 1024
period = 4 * np.pi
X = np.linspace(0, period, N)

Y = np.cos(X*X) # [1,2 3, 4]
print("Исходная функция:")
print(Y)
level = int(np.log2(N))

coeffs = haar_wavelet_transform(Y, level)
reconstructed_signal = inverse_haar_wavelet_transform(coeffs)


fig, ax = plt.subplots(3, 1, figsize=(12, 12))

ax[0].set_title('Исходная функция y = cos(5x) + sin(6x)')
ax[0].plot(X, Y, 'c')
ax[0].grid()

ax[1].set_title('Сигнал после ВП (уровни)')
for i, coeff in enumerate(coeffs):
    ax[1].plot(X[:len(coeff)], coeff, label=f'Level {i+1}')
ax[1].legend()
ax[1].grid()

ax[2].set_title('Восстановленный сигнал')
ax[2].plot(X, reconstructed_signal, 'c')
ax[2].grid()


plt.tight_layout()
plt.show()
