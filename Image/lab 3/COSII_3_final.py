import cv2
import numpy as np
import os


# Функция для масштабирования изображения
def up_scale(image, scale_percent=20):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


# Функция для создания маски зеленого цвета
def mask_green(image):
    lower_green = np.array([35, 50, 50])  # Нижний порог зеленого
    upper_green = np.array([85, 255, 255])  # Верхний порог зеленого
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    return cv2.bitwise_and(image, image, mask=mask)


# Функция для применения медианного фильтра
def apply_median_filter(image, ksize=7):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.medianBlur(gray_image, ksize)


# Функция для применения минимального фильтра
def apply_minimal_filter(image, ksize1=3, ksize2=3):
    kernel = np.ones((ksize1, ksize2), np.uint8)
    return cv2.erode(image, kernel)


# Функция для выделения контуров с использованием оператора Собеля
def apply_Sobel_operator(image):
    if len(image.shape) == 3:  # Если RGB, переводим в оттенки серого
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    return cv2.convertScaleAbs(sobel_combined)


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


# Функция для обработки изображения и поиска контуров
def process_and_analyze_image(image_path, output_folder, min_area=50, max_area=5000):
    # Загружаем оригинальное изображение
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}.")
        return

    # Масштабирование
    processed_image = up_scale(original_image)

    # Создание маски для зеленых объектов
    processed_image = mask_green(processed_image)

    # Применение медианного фильтра
    processed_image = apply_median_filter(processed_image, ksize=7)

    # Применение минимального фильтра
    processed_image = apply_minimal_filter(processed_image)

    # Выделение контуров с использованием оператора Собеля
    sobel_image = apply_Sobel_operator(processed_image)
    _, binary = cv2.threshold(sobel_image, 100, 255, cv2.THRESH_BINARY)

    # Сохранение обработанного изображения
    processed_filename = os.path.join(output_folder, "processed_" + os.path.basename(image_path))
    cv2.imwrite(processed_filename, binary)
    print(f"Обработанное изображение сохранено как {processed_filename}")

    # Нахождение контуров
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Фильтрация контуров по площади
    filtered_contours = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]

    print(f"Обнаружено {len(filtered_contours)} цифр (после фильтрации) на обработанном изображении.")

    for i, contour in enumerate(filtered_contours):
        # Расчет угла поворота
        angle = get_rotation_angle(contour)
        print(f"Цифра {i + 1}: Угол поворота = {angle:.2f}°")

        # Поворот цифры
        rotated = rotate_the_shape(original_image, contour, angle)

        # Сохранение повернутой цифры
        rotated_filename = os.path.join(output_folder, f"rotated_digit_{i + 1}.jpg")
        cv2.imwrite(rotated_filename, rotated)
        print(f"Цифра {i + 1} сохранена как {rotated_filename}")


# Основной блок вызова функции
if __name__ == "__main__":
    # Путь к изображению (задать свой путь к изображению)
    image_path = 'input_images/5.jpg'  # Укажите путь к вашей картинке

    # Путь для сохранения результатов
    output_folder = 'output_images'

    # Создание выходной папки, если она не существует
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Обработка изображения
    process_and_analyze_image(image_path, output_folder)