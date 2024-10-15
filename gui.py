import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer
import cv2
import numpy as np
import face_recognition

# Импортируем функции из main.py
from main import load_source_image, prepare_source_face, detect_faces_haar

class FaceSwapApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Swap Application")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

        self.load_source_button = QPushButton("Load Source Image")
        self.load_source_button.clicked.connect(self.load_source)
        self.layout.addWidget(self.load_source_button)

        self.start_button = QPushButton("Start Face Swap")
        self.start_button.clicked.connect(self.start_face_swap)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Face Swap")
        self.stop_button.clicked.connect(self.stop_face_swap)
        self.layout.addWidget(self.stop_button)

        self.source_image = None
        self.source_face_landmarks = None
        self.source_face = None
        self.source_mask = None
        self.source_points = None

        self.video_capture = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def load_source(self):
        self.source_image, self.source_face_landmarks = load_source_image()
        if self.source_image is not None and self.source_face_landmarks is not None:
            self.source_face, self.source_mask, self.source_points = prepare_source_face(self.source_image, self.source_face_landmarks)
            print("Source image loaded successfully")

    def start_face_swap(self):
        if self.source_image is None:
            print("Please load a source image first")
            return
        self.timer.start(30)  # Update every 30 ms (approx. 33 fps)

    def stop_face_swap(self):
        self.timer.stop()

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = detect_faces_haar(frame, self.face_cascade)

        if self.source_face is not None and self.source_mask is not None:
            for (x, y, w, h) in faces:
                face_region = frame[y:y+h, x:x+w]
                face_landmarks_list = face_recognition.face_landmarks(face_region)
                
                for face_landmarks in face_landmarks_list:
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

                    if not (0 <= np.min(target_points[:, 0]) < frame.shape[1] and
                            0 <= np.min(target_points[:, 1]) < frame.shape[0] and
                            0 <= np.max(target_points[:, 0]) < frame.shape[1] and
                            0 <= np.max(target_points[:, 1]) < frame.shape[0]):
                        continue

                    M, _ = cv2.findHomography(self.source_points, target_points)
                    transformed_face = cv2.warpPerspective(self.source_face, M, (frame.shape[1], frame.shape[0]))
                    transformed_mask = cv2.warpPerspective(self.source_mask, M, (frame.shape[1], frame.shape[0]))

                    center_point = (int(target_points[:,0].mean()), int(target_points[:,1].mean()))
                    center_x = max(0, min(center_point[0], frame.shape[1] - 1))
                    center_y = max(0, min(center_point[1], frame.shape[0] - 1))
                    center_point_corrected = (center_x, center_y)

                    frame = cv2.seamlessClone(transformed_face, frame, transformed_mask, center_point_corrected, cv2.NORMAL_CLONE)

        self.display_image(frame)

    def display_image(self, image):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        self.video_capture.release()
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = FaceSwapApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()