import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Функция для отображения изображения
def display_image(image, title="Image"):
    # plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Функция для обработки всех изображений в директории
def process_all_images(input_folder, output_folder):
    # Получаем список файлов в директории
    files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg'))]

    for file in files:
        # 1. Загрузка изображения
        image_path = os.path.join(input_folder, file)
        image = cv2.imread(image_path)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 2. Определение диапазона для голубого цвета (фигуры)
        lower_blue = np.array([102, 62, 62])   # Нижняя граница голубого цвета #90, 50, 50
        upper_blue = np.array([130, 255, 255])  # Верхняя граница голубого цвета
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # 3. Определение диапазона для зеленого цвета (цифры)
        lower_green = np.array([35, 40, 40])   # Нижняя граница зеленого цвета
        upper_green = np.array([85, 255, 255])  # Верхняя граница зеленого цвета
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # 4. Комбинируем маски для голубого (фигуры) и зеленого (цифры)
        combined_mask = cv2.bitwise_or(blue_mask, green_mask)

        # 5. Применение морфологических операций для удаления шумов и артефактов
        kernel = np.ones((2, 2), np.uint8)
        combined_mask_cleaned = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask_cleaned = cv2.morphologyEx(combined_mask_cleaned, cv2.MORPH_OPEN, kernel)

        # 6. Нахождение контуров голубых фигур и зеленых цифр
        contours, _ = cv2.findContours(combined_mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 7. Создаем черную маску для фона (оставляем только фигуры и цифры)
        image_with_black_background = np.zeros_like(image)  # Создаем черное изображение того же размера
        image_with_black_background[combined_mask_cleaned == 255] = image[combined_mask_cleaned == 255]

        # 8. Применение оператора Собеля для выделения контуров
        gray_image = cv2.cvtColor(image_with_black_background, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)

        # # Нормализуем изображение
        # sobel_combined = np.uint8(sobel_combined / np.max(sobel_combined) * 255)

        # 9. Обводим контуры фигур и цифр красным цветом
        contour_image = image_with_black_background.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)  # Красный контур

        # 10. Сохранение результатов
        output_path = os.path.join(output_folder, 'processed_' + file)
        cv2.imwrite(output_path, contour_image)

        # Вывод оригинала и обработанного изображения
        display_image(image, f"Original: {file}")
        display_image(contour_image, f"Processed: {file}")

# Указываем папки для входных и выходных данных
input_folder = 'input_images'  # Папка с исходными изображениями
output_folder = 'output_images'  # Папка для сохранения обработанных изображений

# Проверка существования выходной папки
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Обработка всех изображений в директории
process_all_images(input_folder, output_folder)