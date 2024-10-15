import cv2
import numpy as np
import face_recognition
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def load_source_image():
    Tk().withdraw()
    source_image_path = askopenfilename(title='Выберите изображение лица')
    if not source_image_path:
        print("No file selected.")
        return None, None
    source_image = face_recognition.load_image_file(source_image_path)
    source_face_landmarks = face_recognition.face_landmarks(source_image)
    if len(source_face_landmarks) == 0:
        print("No faces found in the source image")
        return None, None
    return source_image, source_face_landmarks[0]

def prepare_source_face(source_image, source_face_landmarks):
    source_points = np.array(
        source_face_landmarks['chin'] +
        source_face_landmarks['left_eyebrow'] +
        source_face_landmarks['right_eyebrow'] +
        source_face_landmarks['nose_bridge'] +
        source_face_landmarks['nose_tip'] +
        source_face_landmarks['left_eye'] +
        source_face_landmarks['right_eye'] +
        source_face_landmarks['top_lip'] +
        source_face_landmarks['bottom_lip'] +
        source_face_landmarks['left_eye'] +
        source_face_landmarks['right_eye']
    )
    source_mask = np.zeros(source_image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(source_mask, cv2.convexHull(source_points), 255)
    source_face = cv2.bitwise_and(source_image, source_image, mask=source_mask)
    source_rect = cv2.boundingRect(cv2.convexHull(source_points))
    (sub_x, sub_y, sub_w, sub_h) = source_rect
    source_face = source_image[sub_y:sub_y+sub_h, sub_x:sub_x+sub_w]
    source_mask = source_mask[sub_y:sub_y+sub_h, sub_x:sub_x+sub_w]
    return source_face, source_mask, source_points

def detect_faces_haar(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

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

if __name__ == "__main__":
    source_image, source_face_landmarks = load_source_image()
    if source_image is not None and source_face_landmarks is not None:
        source_face, source_mask, source_points = prepare_source_face(source_image, source_face_landmarks)

    video_capture = cv2.VideoCapture(0)
    
    # Загружаем каскад Хаара для обнаружения лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Используем каскады Хаара для быстрого предварительного обнаружения лиц
        faces = detect_faces_haar(frame, face_cascade)

        if source_face is not None and source_mask is not None:
            for (x, y, w, h) in faces:
                # Вырезаем область лица
                face_region = frame[y:y+h, x:x+w]
                
                # Применяем нейросетевой метод только к обнаруженной области
                face_landmarks_list = face_recognition.face_landmarks(face_region)
                
                for face_landmarks in face_landmarks_list:
                    # Корректируем координаты ключевых точек
                    for feature in face_landmarks:
                        for i in range(len(face_landmarks[feature])):
                            face_landmarks[feature][i] = (face_landmarks[feature][i][0] + x, face_landmarks[feature][i][1] + y)
                    
                    target_points = np.array(
                        face_landmarks['chin'] +
                        face_landmarks['left_eyebrow'] +
                        face_landmarks['right_eyebrow'] +
                        face_landmarks['nose_bridge'] +
                        face_landmarks['nose_tip'] +
                        face_landmarks['left_eye'] +
                        face_landmarks['right_eye'] +
                        face_landmarks['top_lip'] +
                        face_landmarks['bottom_lip'] +
                        face_landmarks['left_eye'] +
                        face_landmarks['right_eye']
                    )

                    # Проверяем, находятся ли точки лица в пределах кадра
                    if not (0 <= np.min(target_points[:, 0]) < frame.shape[1] and
                            0 <= np.min(target_points[:, 1]) < frame.shape[0] and
                            0 <= np.max(target_points[:, 0]) < frame.shape[1] and
                            0 <= np.max(target_points[:, 1]) < frame.shape[0]):
                        continue

                    M, _ = cv2.findHomography(source_points, target_points)
                    transformed_face = cv2.warpPerspective(source_face, M, (frame.shape[1], frame.shape[0]))
                    transformed_mask = cv2.warpPerspective(source_mask, M, (frame.shape[1], frame.shape[0]))

                    center_point = (int(target_points[:,0].mean()), int(target_points[:,1].mean()))
                    center_x = max(0, min(center_point[0], frame.shape[1] - 1))
                    center_y = max(0, min(center_point[1], frame.shape[0] - 1))
                    center_point_corrected = (center_x, center_y)

                    frame = cv2.seamlessClone(transformed_face, frame, transformed_mask, center_point_corrected, cv2.NORMAL_CLONE)

        cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)        
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            source_image, source_face_landmarks = load_source_image()
            if source_image is not None and source_face_landmarks is not None:
                source_face, source_mask, source_points = prepare_source_face(source_image, source_face_landmarks)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
