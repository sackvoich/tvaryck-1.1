import face_alignment
import numpy as np
import cv2
import logging
import torch  # Добавляем импорт torch для проверки доступности CUDA

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
        device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
        if device == 'cuda':
            logger.info("CUDA доступен. Инициализация модели на GPU.")
        else:
            logger.info("CUDA недоступен. Инициализация модели на CPU.")
        
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, 
            device=device, 
            flip_input=False
        )
        return fa
    except Exception as e:
        logger.error("Ошибка при инициализации модели FAN", exc_info=True)
        raise e

def get_landmarks_fan(fa, image, source=False):
    """
    Обнаруживает ключевые точки лица с использованием FAN.
    
    :param fa: Инициализированный объект face_alignment.FaceAlignment
    :param image: Изображение в формате NumPy массива (RGB).
    :param source: Флаг, указывающий, является ли изображение исходным.
    :return: Список словарей с ключевыми точками.
    """
    try:
        preds = fa.get_landmarks(image)
        landmarks = []
        if preds is not None:
            if source:
                logger.info(f"Исходное изображение: обнаружено {len(preds)} лицо(а).")
            else:
                logger.info(f"Видеопоток: обнаружено {len(preds)} лицо(а).")
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
            if source:
                logger.warning("FAN не обнаружил лиц в исходном изображении.")
            else:
                logger.warning("FAN не обнаружил лиц в кадре видеопотока.")
        return landmarks
    except Exception as e:
        logger.error("Ошибка при обнаружении ключевых точек с помощью FAN", exc_info=True)
        return []
