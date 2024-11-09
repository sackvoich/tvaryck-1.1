import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QComboBox,
    QMessageBox, QHBoxLayout
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer
import cv2
import numpy as np
import face_recognition
import logging
import traceback

from main import load_source_image, prepare_source_face, detect_faces_haar, get_available_cameras, detect_face_landmarks
from fan_landmark import initialize_fan

# Настройка логирования
logger = logging.getLogger()
logger.setLevel(logging.ERROR)
file_handler = logging.FileHandler('face_swap_gui.log', encoding='utf-8')
file_formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(levelname)s:%(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

def exception_hook(exctype, value, tb):
    logger.error("Непойманное исключение", exc_info=(exctype, value, tb))
    QMessageBox.critical(None, "Непойманное исключение",
                         f"Произошла непойманная ошибка:\n{''.join(traceback.format_exception(exctype, value, tb))}")
    sys.exit(1)

sys.excepthook = exception_hook

class FaceSwapApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Swap Application")
        self.setGeometry(100, 100, 1600, 900)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.central_widget.setStyleSheet("background-color: #f5f5f5;")  # Задаем фон

        # Заголовок приложения
        self.title_label = QLabel("Face Swap Application")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        self.layout.addWidget(self.title_label)

        # Верхняя часть: видео и индикатор
        self.video_layout = QVBoxLayout()
        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")
        self.video_label.setMinimumSize(1440, 810)
        self.video_layout.addWidget(self.video_label)

        self.overlay_status_indicator = QLabel()
        self.overlay_status_indicator.setFixedSize(20, 20)
        self.overlay_status_indicator.setStyleSheet("background-color: red; border-radius: 10px;")
        self.overlay_status_indicator.setToolTip("Overlay Inactive")
        self.overlay_status_indicator.setAlignment(Qt.AlignTop | Qt.AlignRight)

        self.video_layout.addWidget(self.overlay_status_indicator, alignment=Qt.AlignRight | Qt.AlignTop)
        self.layout.addLayout(self.video_layout)

        # Выбор камеры
        self.camera_select = QComboBox()
        self.camera_select.setStyleSheet("padding: 10px; font-size: 16px;")
        self.layout.addWidget(self.camera_select)
        self.populate_camera_select()

        # Выбор метода обнаружения ключевых точек
        self.method_layout = QHBoxLayout()
        self.method_label = QLabel("Выберите метод обнаружения ключевых точек лица:")
        self.method_label.setStyleSheet("font-size: 16px;")
        self.method_select = QComboBox()
        self.method_select.addItem("Face Recognition", "face_recognition")
        self.method_select.addItem("FAN", "fan")
        self.method_select.setCurrentIndex(0)
        self.method_select.setStyleSheet("padding: 10px; font-size: 16px;")  # Обновленный стиль
        self.method_layout.addWidget(self.method_label)
        self.method_layout.addWidget(self.method_select)
        self.layout.addLayout(self.method_layout)

        # Кнопки управления
        self.buttons_layout = QHBoxLayout()
        button_style = "QPushButton { padding: 10px; font-size: 16px; background-color: #0078D7; color: white; border: none; border-radius: 5px; } QPushButton:hover { background-color: #005A9E; }"

        self.load_source_button = QPushButton("Load Source Image")
        self.load_source_button.setStyleSheet(button_style)
        self.load_source_button.clicked.connect(self.load_source)
        self.buttons_layout.addWidget(self.load_source_button)

        self.start_button = QPushButton("Start Face Swap")
        self.start_button.setStyleSheet(button_style)
        self.start_button.clicked.connect(self.start_face_swap)
        self.buttons_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Face Swap")
        self.stop_button.setStyleSheet(button_style)
        self.stop_button.clicked.connect(self.stop_face_swap)
        self.buttons_layout.addWidget(self.stop_button)

        self.restart_button = QPushButton("Restart Overlay")
        self.restart_button.setStyleSheet(button_style)
        self.restart_button.clicked.connect(self.restart_overlay)
        self.buttons_layout.addWidget(self.restart_button)

        self.layout.addLayout(self.buttons_layout)

        # Инициализация переменных
        self.source_image = None
        self.source_face_landmarks = None
        self.source_face = None
        self.source_mask = None
        self.source_points = None

        self.video_capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.overlay_active = False

        # Инициализация модели FAN
        self.fan = None
        if self.method_select.currentData() == 'fan':
            try:
                self.fan = initialize_fan()
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось инициализировать FAN: {e}")

        self.method_select.currentIndexChanged.connect(self.change_method)
        self.current_method = self.method_select.currentData()

    def populate_camera_select(self):
        try:
            cameras = get_available_cameras()
            if not cameras:
                QMessageBox.warning(self, "Предупреждение", "Не обнаружено ни одной камеры.")
            for camera in cameras:
                self.camera_select.addItem(f"Camera {camera}", camera)
            self.camera_select.currentIndexChanged.connect(self.change_camera)
        except Exception as e:
            logger.error("Ошибка при получении списка камер", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Не удалось получить список камер: {e}")

    def change_camera(self):
        try:
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            camera_index = self.camera_select.currentData()
            if camera_index is not None:
                self.video_capture = cv2.VideoCapture(camera_index)
                if not self.video_capture.isOpened():
                    logger.error(f"Не удалось открыть камеру {camera_index}")
                    QMessageBox.critical(self, "Ошибка", f"Не удалось открыть камеру {camera_index}.")
                    self.video_capture = None
        except Exception as e:
            logger.error("Ошибка при смене камеры", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Не удалось сменить камеру: {e}")

    def change_method(self):
        try:
            self.current_method = self.method_select.currentData()
            if self.current_method == 'fan' and self.fan is None:
                self.fan = initialize_fan()
            elif self.current_method != 'fan' and self.fan is not None:
                del self.fan
                self.fan = None
            QMessageBox.information(self, "Метод изменен", f"Выбран метод: {self.method_select.currentText()}")
        except Exception as e:
            logger.error("Ошибка при изменении метода обнаружения", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Не удалось изменить метод обнаружения: {e}")

    def load_source(self):
        try:
            selected_method = self.current_method
            self.source_image, self.source_face_landmarks = load_source_image(method=selected_method, fa=self.fan, source=True)
            if self.source_image is not None and self.source_face_landmarks is not None:
                self.source_face, self.source_mask, self.source_points = prepare_source_face(
                    self.source_image, self.source_face_landmarks)
                if self.source_face is None or self.source_mask is None or self.source_points is None:
                    QMessageBox.critical(self, "Ошибка", "Не удалось подготовить исходное лицо.")
                    return
                self.overlay_active = True
                self.update_overlay_status()
                QMessageBox.information(self, "Информация", "Исходное изображение успешно загружено.")
            else:
                QMessageBox.warning(self, "Предупреждение", "Не удалось загрузить исходное изображение.")
        except Exception as e:
            logger.error("Ошибка при загрузке исходного изображения", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить исходное изображение: {e}")

    def start_face_swap(self):
        try:
            if self.source_image is None:
                QMessageBox.warning(self, "Предупреждение", "Пожалуйста, загрузите исходное изображение сначала.")
                return
            if self.video_capture is None:
                self.change_camera()
                if self.video_capture is None:
                    return
            if not self.video_capture.isOpened():
                QMessageBox.critical(self, "Ошибка", "Камера не открыта.")
                return
            self.overlay_active = True
            self.update_overlay_status()
            self.timer.start(30)
        except Exception as e:
            logger.error("Ошибка при запуске наложения лица", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить наложение лица: {e}")

    def stop_face_swap(self):
        try:
            if self.timer.isActive():
                self.timer.stop()
                self.overlay_active = False
                self.update_overlay_status()
                QMessageBox.information(self, "Информация", "Наложение лица остановлено.")
        except Exception as e:
            logger.error("Ошибка при остановке наложения лица", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Не удалось остановить наложение лица: {e}")

    def restart_overlay(self):
        try:
            if self.source_image is None:
                QMessageBox.warning(self, "Предупреждение", "Пожалуйста, загрузите исходное изображение сначала.")
                return
            if self.video_capture is None or not self.video_capture.isOpened():
                self.change_camera()
                if self.video_capture is None:
                    return
            self.overlay_active = True
            self.update_overlay_status()
            QMessageBox.information(self, "Информация", "Наложение лица перезапущено.")
        except Exception as e:
            logger.error("Ошибка при перезапуске наложения лица", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Не удалось перезапустить наложение лица: {e}")

    def update_frame(self):
        try:
            if self.video_capture is None or not self.video_capture.isOpened():
                return

            ret, frame = self.video_capture.read()
            if not ret:
                logger.error("Не удалось прочитать кадр из видеопотока.")
                return

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.current_method == 'fan':
                face_landmarks_list = detect_face_landmarks(frame_rgb, method=self.current_method, fa=self.fan, source=False)
            else:
                face_landmarks_list = detect_face_landmarks(frame_rgb, method=self.current_method, source=False)

            if self.source_face is not None and self.source_mask is not None and self.overlay_active:
                try:
                    for face_landmarks in face_landmarks_list:
                        target_points = np.array(
                            face_landmarks['chin'] +
                            face_landmarks['left_eyebrow'] +
                            face_landmarks['right_eyebrow'] +
                            face_landmarks['nose_bridge'] +
                            face_landmarks['nose_tip'] +
                            face_landmarks['left_eye'] +
                            face_landmarks['right_eye'] +
                            face_landmarks['top_lip'] +
                            face_landmarks['bottom_lip']
                        )

                        if not (0 <= np.min(target_points[:, 0]) < frame_rgb.shape[1] and
                                0 <= np.min(target_points[:, 1]) < frame_rgb.shape[0] and
                                0 <= np.max(target_points[:, 0]) < frame_rgb.shape[1] and
                                0 <= np.max(target_points[:, 1]) < frame_rgb.shape[0]):
                            continue

                        try:
                            M, _ = cv2.findHomography(self.source_points, target_points)
                            if M is None:
                                logger.warning("Не удалось найти гомографию для наложения лица.")
                                continue

                            transformed_face = cv2.warpPerspective(
                                self.source_face, M, (frame_rgb.shape[1], frame_rgb.shape[0])
                            )
                            transformed_mask = cv2.warpPerspective(
                                self.source_mask, M, (frame_rgb.shape[1], frame_rgb.shape[0])
                            )

                            if transformed_face.size == 0 or transformed_mask.size == 0:
                                logger.warning("Трансформированное лицо или маска имеют нулевой размер.")
                                continue

                            center_point = (int(target_points[:, 0].mean()), int(target_points[:, 1].mean()))
                            center_x = max(0, min(center_point[0], frame_rgb.shape[1] - 1))
                            center_y = max(0, min(center_point[1], frame_rgb.shape[0] - 1))
                            center_point_corrected = (center_x, center_y)

                            if not (0 <= center_x < frame_rgb.shape[1] and 0 <= center_y < frame_rgb.shape[0]):
                                logger.warning("Центр области наложения вне границ кадра.")
                                continue

                            if cv2.countNonZero(transformed_mask) == 0:
                                logger.warning("Маска не имеет ненулевых пикселей.")
                                continue

                            try:
                                frame_rgb = cv2.seamlessClone(
                                    transformed_face, frame_rgb, transformed_mask, center_point_corrected, cv2.NORMAL_CLONE
                                )
                            except cv2.error as e:
                                logger.warning(f"Ошибка в cv2.seamlessClone: {e}")
                                continue

                        except Exception as e:
                            logger.error("Ошибка при трансформации и наложении лица", exc_info=True)
                            continue

                except Exception as e:
                    logger.error("Ошибка при обработке лиц в кадре", exc_info=True)
                    self.overlay_active = False
                    self.update_overlay_status()
                    QMessageBox.critical(self, "Ошибка", f"Произошла ошибка при обработке лиц: {e}")

            # Растягиваем изображение до пропорций 16:9
            frame_rgb = cv2.resize(frame_rgb, (1440, 810), interpolation=cv2.INTER_LINEAR)
            self.display_image(frame_rgb)
        except Exception as e:
            logger.error("Непредвиденная ошибка в методе update_frame", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Произошла непредвиденная ошибка: {e}")
            self.overlay_active = False
            self.update_overlay_status()

    def display_image(self, image):
        try:
            h, w, ch = image.shape
            bytes_per_line = ch * w
            qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            logger.error("Ошибка при отображении изображения", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Не удалось отобразить изображение: {e}")

    def update_overlay_status(self):
        try:
            if self.overlay_active:
                self.overlay_status_indicator.setStyleSheet("background-color: green; border-radius: 10px;")
                self.overlay_status_indicator.setToolTip("Overlay Active")
            else:
                self.overlay_status_indicator.setStyleSheet("background-color: red; border-radius: 10px;")
                self.overlay_status_indicator.setToolTip("Overlay Inactive")
        except Exception as e:
            logger.error("Ошибка при обновлении индикатора состояния наложения", exc_info=True)

    def closeEvent(self, event):
        try:
            if self.video_capture:
                self.video_capture.release()
        except Exception as e:
            logger.error("Ошибка при закрытии видео захвата", exc_info=True)
        super().closeEvent(event)

def main():
    app = QApplication(sys.argv)
    window = FaceSwapApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
