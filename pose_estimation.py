import cv2
import mediapipe as mp
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)
handler = logging.FileHandler('pose_estimation.log', encoding='utf-8')
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class PoseEstimator:
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def get_keypoints(self, image):
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # MediaPipe ожидает RGB
            results = self.pose.process(image)

            if results.pose_landmarks:
                keypoints = np.array([[lmk.x * image.shape[1], lmk.y * image.shape[0]] for lmk in results.pose_landmarks.landmark])
                logger.info(f"Pose keypoints detected: {keypoints.shape}")
                return keypoints
            else:
                logger.warning("No pose landmarks detected.")
                return None

        except Exception as e:
            logger.error(f"Ошибка при определении поз: {e}", exc_info=True)
            return None

# Some updates (needed to fix)
# import cv2
# import mediapipe as mp
# import numpy as np
# import logging

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.ERROR)
# handler = logging.FileHandler('pose_estimation.log', encoding='utf-8')
# formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)

# class PoseEstimator:
#     def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
#         self.mp_pose = mp.solutions.pose
#         self.pose = self.mp_pose.Pose(
#             static_image_mode=static_image_mode,
#             model_complexity=model_complexity,
#             min_detection_confidence=min_detection_confidence,
#             min_tracking_confidence=min_tracking_confidence
#         )

#     def get_keypoints(self, image):
#         try:
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # MediaPipe ожидает RGB
#             results = self.pose.process(image)

#             if results.pose_landmarks:
#                 keypoints = np.array([[lmk.x * image.shape[1], lmk.y * image.shape[0]] for lmk in results.pose_landmarks.landmark])
#                 logger.info(f"Pose keypoints detected: {keypoints.shape}")
#                 return keypoints
#             else:
#                 logger.warning("No pose landmarks detected.")
#                 return None

#         except Exception as e:
#             logger.error(f"Ошибка при определении поз: {e}", exc_info=True)
#             return None
