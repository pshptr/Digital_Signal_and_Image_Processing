import cv2
import numpy as np
import os  # Импортируем os для работы с файловой системой


# Функция для вычисления угла поворота
def get_rotation_angle(contour):
    rect = cv2.minAreaRect(contour)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle


# Функция для поворота объекта
def rotate_the_shape(image, contour, angle):
    rect = cv2.minAreaRect(contour)
    center = tuple(map(int, rect[0]))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


# Основная функция обработки изображения
def process_single_image(image_path, output_folder, min_area=50, max_area=5000):
    # Загружаем изображение
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}.")
        return

    # Бинаризация для выделения белых цифр на черном фоне
    _, binary = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    # Нахождение контуров
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация контуров по площади
    filtered_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

    print(f"Обнаружено {len(filtered_contours)} цифр (после фильтрации) на изображении {image_path}")

    for i, contour in enumerate(filtered_contours):
        # Расчет угла поворота
        angle = get_rotation_angle(contour)
        print(f"Цифра {i + 1}: Угол поворота = {angle:.2f}°")

        # Поворот цифры
        rotated = rotate_the_shape(image, contour, angle)

        # Сохранение повернутой цифры
        crop_path = f"{output_folder}/rotated_digit_{i + 1}.jpg"
        cv2.imwrite(crop_path, rotated)

        print(f"Цифра {i + 1} повернута и сохранена в {crop_path}")

    # Сохранение обработанного изображения (для контроля)
    processed_path = f"{output_folder}/processed.jpg"
    cv2.imwrite(processed_path, binary)
    print(f"Обработанное изображение сохранено в {processed_path}")


# Путь к изображению
image_path = 'input_images/processed_image.jpg'  # Укажите путь к вашей картинке

# Путь для сохранения результатов
output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Обработка одной картинки
process_single_image(image_path, output_folder)