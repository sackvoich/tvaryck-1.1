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

# Some updates (needed to fix)

# import cv2
# import numpy as np
# import logging
# import face_recognition  # Импортируем face_recognition
# from fan_landmark import get_landmarks_fan # Импорт функции

# # Настройка логирования
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.ERROR)
# handler = logging.FileHandler('clothes_overlay.log', encoding='utf-8')
# formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# def load_clothes_image(image_path, fa):
#     try:
#         clothes_image = face_recognition.load_image_file(image_path)
#         clothes_landmarks = get_landmarks_fan(fa, clothes_image, source=True)
#         if len(clothes_landmarks) == 0:
#             return None, None
#         return clothes_image, clothes_landmarks[0]
#     except Exception as e:
#         logger.error("Ошибка при загрузке изображения одежды", exc_info=True)
#         return None, None

# def create_clothes_mask(image, body_keypoints):
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     # Новые индексы ключевых точек торса
#     torso_indices = [11, 13, 15, 12, 14, 16, 23, 24]  # Добавлены локти и запястья
    
#     if len(body_keypoints) < max(torso_indices) + 1:
#         logger.warning("Недостаточно ключевых точек тела для создания маски одежды.")
#         return None
    
#     points = body_keypoints[torso_indices].astype(np.int32)
    
#     # Интерполяция между плечом и локтем, локтем и запястьем
#     interpolated_points = []
#     num_interpolated = 2  # Количество промежуточных точек между двумя основными
    
#     # Левый плечо -> левый локоть -> левый запястье
#     left_shoulder = points[0]
#     left_elbow = points[1]
#     left_wrist = points[2]
    
#     interpolated_points.extend([left_shoulder])
#     interpolated_points.extend(interpolate_points(left_shoulder, left_elbow, num_points=num_interpolated))
#     interpolated_points.extend([left_elbow])
#     interpolated_points.extend(interpolate_points(left_elbow, left_wrist, num_points=num_interpolated))
#     interpolated_points.extend([left_wrist])
    
#     # Правый плечо -> правый локоть -> правый запястье
#     right_shoulder = points[3]
#     right_elbow = points[4]
#     right_wrist = points[5]
    
#     interpolated_points.extend([right_shoulder])
#     interpolated_points.extend(interpolate_points(right_shoulder, right_elbow, num_points=num_interpolated))
#     interpolated_points.extend([right_elbow])
#     interpolated_points.extend(interpolate_points(right_elbow, right_wrist, num_points=num_interpolated))
#     interpolated_points.extend([right_wrist])
    
#     # Добавление бедер
#     left_hip = points[6]
#     right_hip = points[7]
#     interpolated_points.extend([left_hip, right_hip])
    
#     interpolated_points = np.array(interpolated_points, dtype=np.int32)
    
#     # Создаём выпуклую оболочку вокруг всех точек
#     convex_hull = cv2.convexHull(interpolated_points)
#     cv2.fillConvexPoly(mask, convex_hull, 255)
    
#     return mask

# def overlay_clothes(frame, clothes_image, clothes_mask, clothes_keypoints, body_keypoints):
#     try:
#         logger.info("Starting clothes overlay.")
#         # Новые индексы ключевых точек торса
#         torso_indices = [11, 13, 15, 12, 14, 16, 23, 24]
    
#         # Извлекаем ключевые точки торса из тела и одежды
#         body_torso_keypoints = body_keypoints[torso_indices]
#         clothes_torso_keypoints = clothes_keypoints[torso_indices]
    
#         logger.debug(f"Body Torso Keypoints: {body_torso_keypoints}")
#         logger.debug(f"Clothes Torso Keypoints: {clothes_torso_keypoints}")
    
#         # Проверяем наличие ключевых точек
#         if len(body_torso_keypoints) < 8 or len(clothes_torso_keypoints) < 8:
#             logger.warning("Недостаточно ключевых точек для наложения одежды на торс.")
#             return frame
    
#         # Вычисляем матрицу гомографии, используя все ключевые точки
#         M, _ = cv2.findHomography(clothes_torso_keypoints, body_torso_keypoints, cv2.RANSAC, 5.0)
    
#         if M is None:
#             logger.warning("Homography matrix is None. Skipping clothes overlay.")
#             return frame
    
#         # Трансформируем изображение одежды и маску
#         transformed_clothes = cv2.warpPerspective(clothes_image, M, (frame.shape[1], frame.shape[0]))
#         transformed_mask = cv2.warpPerspective(clothes_mask, M, (frame.shape[1], frame.shape[0]))
    
#         logger.debug(f"Transformed clothes shape: {transformed_clothes.shape}")
#         logger.debug(f"Transformed mask shape: {transformed_mask.shape}")
    
#         # Преобразуем маску в трехканальную
#         transformed_mask_3ch = cv2.cvtColor(transformed_mask, cv2.COLOR_GRAY2BGR)
    
#         # Создаем маску для наложения
#         clothes_area = transformed_mask_3ch.astype(bool)
    
#         # Накладываем одежду на кадр
#         frame[clothes_area] = transformed_clothes[clothes_area]
    
#         logger.info("Clothes overlay completed successfully.")
#         return frame
    
#     except Exception as e:
#         logger.error(f"Ошибка при наложении одежды: {e}", exc_info=True)
#         return frame


# def interpolate_points(p1, p2, num_points=2):
#     """
#     Линейная интерполяция между двумя точками.
    
#     :param p1: Первая точка (x, y)
#     :param p2: Вторая точка (x, y)
#     :param num_points: Количество промежуточных точек
#     :return: Список промежуточных точек
#     """
#     return [((p1[0] + i * (p2[0] - p1[0]) / (num_points + 1)),
#              (p1[1] + i * (p2[1] - p1[1]) / (num_points + 1))) for i in range(1, num_points + 1)]
