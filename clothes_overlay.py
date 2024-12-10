import cv2
import numpy as np
import logging
import face_recognition  # Импортируем face_recognition
from fan_landmark import get_landmarks_fan # Импорт функции

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
handler = logging.FileHandler('clothes_overlay.log', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_clothes_image(image_path, fa):
    try:
        clothes_image = face_recognition.load_image_file(image_path)
        clothes_landmarks = get_landmarks_fan(fa, clothes_image, source=True)
        if len(clothes_landmarks) == 0:
            return None, None
        return clothes_image, clothes_landmarks[0]
    except Exception as e:
        logger.error("Ошибка при загрузке изображения одежды", exc_info=True)
        return None, None

def create_clothes_mask(image, body_keypoints):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # Определяем индексы ключевых точек для плеч и бедер
    # Индексы зависят от используемой модели. Для MediaPipe Pose это:
    # 11 - левое плечо, 12 - правое плечо, 23 - левое бедро, 24 - правое бедро
    torso_indices = [11, 12, 23, 24]  # Убедитесь, что индексы соответствуют вашей модели

    if len(body_keypoints) < max(torso_indices) + 1:
        logger.warning("Недостаточно ключевых точек тела для создания маски одежды.")
        return None

    points = body_keypoints[torso_indices].astype(np.int32)

    # Создаём выпуклую оболочку вокруг плеч и бедер
    convex_hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, convex_hull, 255)

    return mask


def overlay_clothes(frame, clothes_image, clothes_mask, clothes_keypoints, body_keypoints):
    try:
        logger.info("Starting clothes overlay.")
        # Определяем индексы ключевых точек для торса (плечи, бедра)
        torso_indices = [11, 12, 23, 24]  # Индексы в MediaPipe для левого плеча, правого плеча, левого бедра, правого бедра

        # Извлекаем ключевые точки торса из тела и одежды
        body_torso_keypoints = body_keypoints[torso_indices]
        clothes_torso_keypoints = clothes_keypoints[torso_indices]

        logger.debug(f"Body Torso Keypoints: {body_torso_keypoints}")
        logger.debug(f"Clothes Torso Keypoints: {clothes_torso_keypoints}")

        # Проверяем наличие ключевых точек
        if len(body_torso_keypoints) < 4 or len(clothes_torso_keypoints) < 4:
            logger.warning("Недостаточно ключевых точек для наложения одежды на торс.")
            return frame

        M, _ = cv2.findHomography(clothes_torso_keypoints, body_torso_keypoints)

        if M is None:
            logger.warning("Homography matrix is None. Skipping clothes overlay.")
            return frame

        transformed_clothes = cv2.warpPerspective(clothes_image, M, (frame.shape[1], frame.shape[0]))
        transformed_mask = cv2.warpPerspective(clothes_mask, M, (frame.shape[1], frame.shape[0]))

        logger.debug(f"Transformed clothes shape: {transformed_clothes.shape}")
        logger.debug(f"Transformed mask shape: {transformed_mask.shape}")

        # Преобразуем маску в трехканальную
        transformed_mask_3ch = cv2.cvtColor(transformed_mask, cv2.COLOR_GRAY2BGR)

        # Смешиваем изображения с помощью маски
        frame = np.where(transformed_mask_3ch, transformed_clothes, frame)

        logger.info("Clothes overlay completed successfully.")
        return frame

    except Exception as e:
        logger.error(f"Ошибка при наложении одежды: {e}", exc_info=True)
        return frame