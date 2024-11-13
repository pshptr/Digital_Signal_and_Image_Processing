import math
import numpy
from scipy.linalg import dft
from scipy.fft import ifft, fft
from matplotlib import pyplot

# Функция для вычисления быстрого преобразования Фурье (FFT)
def count_fft(a):
    a_len = len(a)

    # Базовый случай: если длина массива меньше или равна 1, возвращаем его же
    if a_len <= 1:
        return a

    # Рекурсивное вычисление FFT для четных и нечетных элементов
    bEven = count_fft(a[0::2])
    bOdd = count_fft(a[1::2])

    # Вычисление преобразования Фурье
    y1 = []
    y2 = []
    for i in range(a_len // 2):
        y1.append(bEven[i] + numpy.exp(-2j * numpy.pi * i / a_len) * bOdd[i])
        y2.append(bEven[i] - numpy.exp(-2j * numpy.pi * i / a_len) * bOdd[i])

    return y1 + y2

# Функция для вычисления обратного быстрого преобразования Фурье (IFFT)
def count_ifft(A, N):
    a_len = len(A)
    a = numpy.zeros(a_len, dtype=complex)

    # Вычисление IFFT
    for i in range(a_len):
        for k in range(a_len):
            a[i] += A[k] * numpy.exp(2j * numpy.pi * k * i / a_len)
        a[i] /= N
    return a

# Функция для вычисления линейной свертки
def get_linear_convolution(y, z):
    convolution_len: int = len(y) + len(z) - 1
    convolution = [0] * convolution_len

    y = numpy.array(y)
    z = numpy.array(z)

    # Вычисление свертки
    for i in range(convolution_len):
        for j in range(max(0, i - len(z) + 1), min(i + 1, len(y))):
            convolution[i] += y[j] * z[i - j]

    return convolution

# Функция для вычисления циклической свертки
def get_cycle_convolution(y, z):
    convolution_len: int = len(y)
    convolution = [0] * convolution_len

    y = numpy.array(y)
    z = numpy.array(z)

    # Вычисление циклической свертки
    for i in range(convolution_len):
        for j in range(convolution_len):
            k = (i - j) % convolution_len
            convolution[i] += y[j] * z[k]

    return convolution

# Функция для вычисления свертки с использованием библиотеки numpy
def get_library_convolution(y, z):
    return numpy.convolve(y, z)

# Функция для вычисления свертки с использованием FFT и IFFT
def get_furie_convolution(y, z, N):
    convolution_len: int = len(y)
    convolution = [0] * convolution_len

    # Применение FFT к обоим сигналам
    y_after_fft = count_fft(y)
    z_after_fft = count_fft(z)

    # Вычисление свертки с использованием FFT
    for i in range(convolution_len):
        convolution[i] = y_after_fft[i] * z_after_fft[i]

    # Применение IFFT к результату
    return count_ifft(convolution, N)

# Функция для вычисления корреляции
def get_correlation(y, z):
    correlation_len: int = len(y)
    correlation = [0] * correlation_len

    y = numpy.array(y)
    z = numpy.array(z)

    # Вычисление корреляции
    for i in range(correlation_len):
        for j in range(correlation_len):
            if i + j >= correlation_len:
                continue
            correlation[i] += y[j] * z[i + j]

    return correlation

# Функция для вычисления корреляции с использованием библиотеки numpy
def get_library_correlation(y, z, N):
    return numpy.correlate(y, z, mode='full')[:N][::-1]

N: int = 16
figure, axes = pyplot.subplots(2, 4)

# Генерация сигналов
def get_signals():
    x = numpy.linspace(0, N - 1, N)
    y = numpy.sin(3 * x)
    z = numpy.cos(x)
    double_x = numpy.linspace(0, 2 * N - 2, 2 * N - 1)
    return x, y, z, double_x

# Функция для отображения графика
def print_graph(x, y, coords: list[int], title: str) -> None:
    axes[coords[0], coords[1]].plot(x, y)
    axes[coords[0], coords[1]].set_title(title)

# Основная функция программы
def main():
    # Сигналы
    x, y, z, double_x = get_signals()

    # Свертки
    linear_convolution = get_linear_convolution(y, z)
    cycle_convolution = get_cycle_convolution(y, z)
    library_convolution = get_library_convolution(y, z)
    furie_convolution = get_furie_convolution(y, z, N).real

    # Корреляции
    correlation = get_correlation(y, z)
    library_correlation = get_library_correlation(y, z, N)

    # Оригинальные сигналы
    print_graph(x, y, [0, 0], 'Original y:')
    print_graph(x, z, [0, 1], 'Original z:')

    # Корреляции
    print_graph(x, correlation, [0, 2], 'Correlation:')
    print_graph(x, library_correlation, [0, 3], 'Library correlation:')

    # Свертки
    print_graph(double_x, linear_convolution, [1, 0], 'Linear convolution:')
    print_graph(x, cycle_convolution, [1, 1], 'Cycle convolution:')
    print_graph(double_x, library_convolution, [1, 2], 'Library convolution:')
    print_graph(x, furie_convolution, [1, 3], 'Furie convolution:')

    pyplot.gcf().canvas.manager.set_window_title('Convolutions and correlations')
    pyplot.tight_layout()
    pyplot.show()

# Запуск основной функции программ
if __name__ == '__main__':
    main()
