import face_alignment
import numpy as np
import cv2
import logging

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('fan_landmark.log', encoding='utf-8')  # Указание кодировки utf-8
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def initialize_fan():
    """
    Инициализирует модель FAN.
    
    :return: Объект face_alignment.FaceAlignment
    """
    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', flip_input=False)
        return fa
    except Exception as e:
        logger.error("Ошибка при инициализации модели FAN", exc_info=True)
        raise e

def get_landmarks_fan(fa, image):
    """
    Обнаруживает ключевые точки лица с использованием FAN.
    
    :param fa: Инициализированный объект face_alignment.FaceAlignment
    :param image: Изображение в формате NumPy массива (RGB).
    :return: Список словарей с ключевыми точками.
    """
    try:
        preds = fa.get_landmarks(image)
        landmarks = []
        if preds is not None:
            logger.info(f"FAN обнаружил {len(preds)} лицо(а).")
            for pred in preds:
                landmarks_dict = {
                    'chin': list(map(tuple, pred[0:17])),
                    'left_eyebrow': list(map(tuple, pred[17:22])),
                    'right_eyebrow': list(map(tuple, pred[22:27])),
                    'nose_bridge': list(map(tuple, pred[27:31])),
                    'nose_tip': list(map(tuple, pred[31:36])),
                    'left_eye': list(map(tuple, pred[36:42])),
                    'right_eye': list(map(tuple, pred[42:48])),
                    'top_lip': list(map(tuple, pred[48:54])),
                    'bottom_lip': list(map(tuple, pred[54:60]))
                }
                landmarks.append(landmarks_dict)
        else:
            logger.warning("FAN не обнаружил лиц в изображении.")
        return landmarks
    except Exception as e:
        logger.error("Ошибка при обнаружении ключевых точек с помощью FAN", exc_info=True)
        return []
