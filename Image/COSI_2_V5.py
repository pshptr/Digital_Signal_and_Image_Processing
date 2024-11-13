import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Функция для отображения изображения
def display_image(image, title="Image"):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


# Функция для вычисления площади по формуле Гаусса
def gaussian_area(contour):
    x_coords = [point[0][0] for point in contour]
    y_coords = [point[0][1] for point in contour]
    n = len(contour)

    area = 0.0
    for i in range(n - 1):
        area += x_coords[i] * y_coords[i + 1] - y_coords[i] * x_coords[i + 1]
    area += x_coords[-1] * y_coords[0] - y_coords[-1] * x_coords[0]

    return abs(area) / 2.0


# Функция для вычисления признаков объекта
def calculate_features(contour):
    area = gaussian_area(contour)
    perimeter = len(contour)

    x_coords = [point[0][0] for point in contour]
    y_coords = [point[0][1] for point in contour]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)
    aspect_ratio = width / height if height > 0 else 0

    return [area, perimeter, aspect_ratio]


# Функция для обработки всех изображений в директории
def process_all_images(input_folder, output_folder):
    files = [f for f in os.listdir(input_folder) if f.endswith('.jpg')]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Красный, зеленый, синий для кластеров

    for file in files:
        # Загрузка изображения
        image_path = os.path.join(input_folder, file)
        image = cv2.imread(image_path)

        # Преобразование и маскирование
        mean_brightness = np.mean(image, axis=2)
        mask = image > (mean_brightness[:, :, None] * 0.6)
        mask = ~mask.all(axis=2)
        result_image = np.zeros_like(image)
        result_image[mask] = image[mask]

        # Преобразование в оттенки серого, размытие и бинаризация
        gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, binary = cv2.threshold(blurred, 75, 150, cv2.THRESH_BINARY)

        # Морфологические операции и контуры
        kernel = np.ones((15, 15), np.uint8)
        eroded = cv2.erode(binary, kernel, iterations=5)
        dilated = cv2.dilate(eroded, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Создание пустого изображения для отображения результатов
        result_image = np.zeros_like(image)

        # Вычисление признаков для кластеризации на каждой картинке отдельно
        features = []
        for contour in contours:
            features.append(calculate_features(contour))
            print(f"Площадь фигуры: {features[-1][0]}")  # Вывод площади в консоль

        # Применение KMeans для 3 кластеров на текущем изображении
        if len(features) >= 3:  # Проверка на количество объектов для кластеризации
            kmeans = KMeans(n_clusters=3, random_state=0)
            labels = kmeans.fit_predict(features)
        else:
            labels = [0] * len(features)  # Если фигур меньше 3, присваиваем один класс всем

        # Отрисовка контуров по классам
        for i, contour in enumerate(contours):
            color = colors[labels[i]]
            cv2.drawContours(result_image, [contour], -1, color, cv2.FILLED)

        # # Обведение контуров красным цветом
        # cv2.drawContours(result_image, contours, -1, (0, 0, 255), 2)

        # Сохранение и вывод результата
        output_path = os.path.join(output_folder, 'processed_' + file)
        cv2.imwrite(output_path, result_image)
        display_image(image, f"Original: {file}")
        display_image(result_image, f"Processed: {file}")


# Указываем папки для входных и выходных данных
input_folder = 'input_images'
output_folder = 'output_images'

# Проверка существования выходной папки
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Обработка всех изображений в директории
process_all_images(input_folder, output_folder)