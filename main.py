import cv2
import numpy as np
import face_recognition
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import logging
import os

# Импортируем функции из fan_landmark.py
from fan_landmark import initialize_fan, get_landmarks_fan

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
handler = logging.FileHandler('face_swap.log', encoding='utf-8')  # Указание кодировки utf-8
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def detect_face_landmarks(image, method='face_recognition', fa=None):
    if method == 'face_recognition':
        return face_recognition.face_landmarks(image)
    elif method == 'fan':
        if fa is None:
            logger.error("Модель FAN не инициализирована.")
            return []
        return get_landmarks_fan(fa, image)
    else:
        raise ValueError(f"Неизвестный метод обнаружения ключевых точек: {method}")


def load_source_image(method='face_recognition', fa=None):
    Tk().withdraw()
    source_image_path = askopenfilename(title='Выберите изображение лица')
    if not source_image_path:
        logger.error("No file selected.")
        print("No file selected.")
        return None, None
    try:
        source_image = face_recognition.load_image_file(source_image_path)
        source_face_landmarks = detect_face_landmarks(source_image, method=method, fa=fa)
        if len(source_face_landmarks) == 0:
            logger.error("No faces found in the source image")
            print("No faces found in the source image")
            return None, None
        return source_image, source_face_landmarks[0]
    except Exception as e:
        logger.error("Ошибка при загрузке исходного изображения", exc_info=True)
        print(f"Ошибка при загрузке исходного изображения: {e}")
        return None, None

def prepare_source_face(source_image, source_face_landmarks):
    try:
        required_keys = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip', 
                         'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
        for key in required_keys:
            if key not in source_face_landmarks:
                logger.error(f"Отсутствуют ключевые точки: {key}")
                return None, None, None

        logger.info("Подготовка исходного лица...")
        source_points = np.array(
            source_face_landmarks['chin'] +
            source_face_landmarks['left_eyebrow'] +
            source_face_landmarks['right_eyebrow'] +
            source_face_landmarks['nose_bridge'] +
            source_face_landmarks['nose_tip'] +
            source_face_landmarks['left_eye'] +
            source_face_landmarks['right_eye'] +
            source_face_landmarks['top_lip'] +
            source_face_landmarks['bottom_lip']
        ).astype(np.int32)  # Преобразование к целочисленному типу

        if source_points.size == 0:
            logger.error("Набор ключевых точек пуст.")
            return None, None, None

        logger.info(f"Количество точек: {len(source_points)}")
        logger.debug(f"Координаты точек: {source_points}")

        source_mask = np.zeros(source_image.shape[:2], dtype=np.uint8)
        convex_hull = cv2.convexHull(source_points)
        if convex_hull is None or convex_hull.size == 0:
            logger.error("convexHull вернул пустой результат.")
            return None, None, None

        cv2.fillConvexPoly(source_mask, convex_hull, 255)
        source_face = cv2.bitwise_and(source_image, source_image, mask=source_mask)
        source_rect = cv2.boundingRect(convex_hull)
        (sub_x, sub_y, sub_w, sub_h) = source_rect
        source_face = source_image[sub_y:sub_y+sub_h, sub_x:sub_x+sub_w]
        source_mask = source_mask[sub_y:sub_y+sub_h, sub_x:sub_x+sub_w]
        logger.info("Исходное лицо подготовлено успешно.")
        return source_face, source_mask, source_points
    except Exception as e:
        logger.error("Ошибка при подготовке исходного лица", exc_info=True)
        print(f"Ошибка при подготовке исходного лица: {e}")
        return None, None, None


def detect_faces_haar(frame, face_cascade):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Применяем CLAHE для улучшения контрастности
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    except Exception as e:
        logger.error("Ошибка при обнаружении лиц с помощью Haar Cascade", exc_info=True)
        return []

def get_available_cameras():
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            cameras.append(index)
        cap.release()
        index += 1
    return cameras
