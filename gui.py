import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QComboBox,
    QMessageBox, QHBoxLayout, QCheckBox, QButtonGroup, QRadioButton, QGroupBox, QFrame,
    QSpacerItem, QSizePolicy
)
from PySide6.QtGui import QPixmap, QImage, QMovie
from PySide6.QtCore import Qt, QTimer, QSize
import cv2
import numpy as np
import face_recognition
import logging
import traceback
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os
import torch

from main import load_source_image, prepare_source_face, detect_faces_haar, get_available_cameras, detect_face_landmarks
from fan_landmark import initialize_fan
from pose_estimation import PoseEstimator  # Импорт обертки для MediaPipe
from clothes_overlay import load_clothes_image, create_clothes_mask, overlay_clothes

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
        self.setGeometry(100, 100, 1600, 700)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)  # Основной QHBoxLayout для разделения на левую и правую части
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.central_widget.setStyleSheet("background-color: #f5f5f5;")

        # Заголовок приложения
        self.title_label = QLabel("Face Swap Application")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #333;")
        self.layout.addWidget(self.title_label)

        # --- Левая часть: Видео ---
        self.video_area_layout = QVBoxLayout()  # QVBoxLayout для левой части окна (видео + индикатор)
        # **Настройка политики размера для video_area_layout, чтобы занимала доступное пространство**
        self.video_area_layout.setSizeConstraint(QVBoxLayout.SetMinAndMaxSize) # Устанавливаем ограничение на размер
        self.video_area_layout.setContentsMargins(0, 0, 20, 0) # Добавляем отступ справа

        self.video_layout = QVBoxLayout()  # Содержимое video_area_layout - видеолейбл и индикатор
        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 10px;")
        # **Установка политики размера и соотношения сторон для video_label**
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Растягивается в обе стороны
        self.video_label.setMinimumSize(640, 360) # Минимальный размер, чтобы не схлопывалось
        self.video_label.setMaximumSize(1920, 1080) # Максимальный размер, ограничиваем сверху
        self.video_layout.addWidget(self.video_label)

        # Индикатор статуса наложения (текст + иконка) - остается в правом верхнем углу
        self.overlay_status_layout = QHBoxLayout()
        self.overlay_status_indicator = QLabel()
        self.overlay_status_indicator.setFixedSize(20, 20)
        self.overlay_status_indicator.setToolTip("Overlay Inactive")
        self.overlay_status_text = QLabel("Наложение: Неактивно")
        self.overlay_status_text.setStyleSheet("color: #666; font-size: 14px;")
        self.overlay_status_layout.addWidget(self.overlay_status_indicator)
        self.overlay_status_layout.addWidget(self.overlay_status_text)
        self.overlay_status_layout.setAlignment(Qt.AlignTop | Qt.AlignRight)

        self.video_layout.addLayout(self.overlay_status_layout)
        self.video_area_layout.addLayout(self.video_layout)  # Добавляем video_layout в video_area_layout

        # --- Правая часть: Элементы управления ---
        self.controls_panel_layout = QVBoxLayout()  # QVBoxLayout для правой части окна (все элементы управления)
        self.controls_panel_layout.setAlignment(Qt.AlignTop)  # Выравнивание элементов управления по верху
        # **Настройка политики размера для controls_panel_layout, фиксированная ширина, растягивается по вертикали**
        self.controls_panel_layout.setSizeConstraint(QVBoxLayout.SetMinAndMaxSize) # Устанавливаем ограничение на размер
        self.controls_panel_layout.setContentsMargins(20, 0, 0, 0) # Добавляем отступ слева

        # Индикатор инициализации FAN (остается без изменений)
        self.fan_init_indicator = QLabel()
        self.fan_init_indicator.setFixedSize(20, 20)
        self.fan_init_indicator.clear()

        # Кнопки управления и индикаторы загрузки (стиль кнопок остается без изменений)
        self.buttons_layout = QHBoxLayout()
        button_style = "QPushButton { padding: 8px 15px; font-size: 16px; background-color: #0078D7; color: white; border: none; border-radius: 5px; min-width: 120px; } QPushButton:hover { background-color: #005A9E; }"  # Меньше padding и min-width

        # Кнопка "Load Face Image" и индикатор
        self.load_face_layout = QVBoxLayout()
        self.load_face_button = QPushButton("Загрузить лицо")
        self.load_face_button.setStyleSheet(button_style)
        # **Политика размера для кнопок загрузки, растягиваются по горизонтали**
        self.load_face_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.load_face_button.clicked.connect(self.load_source)
        self.load_face_indicator = QLabel()
        self.load_face_indicator.setFixedSize(20, 20)
        self.load_face_indicator.clear()
        self.load_face_layout.addWidget(self.load_face_button)
        self.load_face_layout.addWidget(self.load_face_indicator, alignment=Qt.AlignCenter)

        # Кнопка "Load Clothes Image" и индикатор
        self.load_clothes_layout = QVBoxLayout()
        self.load_clothes_button = QPushButton("Загрузить одежду")
        self.load_clothes_button.setStyleSheet(button_style)
        # **Политика размера для кнопок загрузки, растягиваются по горизонтали**
        self.load_clothes_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.load_clothes_button.clicked.connect(self.load_clothes_image)
        self.load_clothes_indicator = QLabel()
        self.load_clothes_indicator.setFixedSize(20, 20)
        self.load_clothes_indicator.clear()
        self.load_clothes_layout.addWidget(self.load_clothes_button)
        self.load_clothes_layout.addWidget(self.load_clothes_indicator, alignment=Qt.AlignCenter)

        self.start_button = QPushButton("Старт")
        self.start_button.setStyleSheet(button_style)
        # **Политика размера для кнопок Старт/Стоп, растягиваются по горизонтали**
        self.start_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.start_button.clicked.connect(self.start_face_swap)
        self.buttons_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Стоп")
        self.stop_button.setStyleSheet(button_style)
        # **Политика размера для кнопок Старт/Стоп, растягиваются по горизонтали**
        self.stop_button.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        self.stop_button.clicked.connect(self.stop_face_swap)
        self.buttons_layout.addWidget(self.stop_button)

        # Radio buttons initialization
        self.overlay_mode_group = QButtonGroup(self)
        self.overlay_mode_label = QLabel("Режим наложения:")

        self.no_overlay_radio = QRadioButton("Без наложения")
        self.face_overlay_radio = QRadioButton("Только лицо")
        self.clothes_overlay_radio = QRadioButton("Только одежда")
        self.both_overlays_radio = QRadioButton("Лицо и одежда")

        self.overlay_mode_group.addButton(self.no_overlay_radio)
        self.overlay_mode_group.addButton(self.face_overlay_radio)
        self.overlay_mode_group.addButton(self.clothes_overlay_radio)
        self.overlay_mode_group.addButton(self.both_overlays_radio)

        # --- Группа "Выбор источника видео" ---
        self.camera_group = QGroupBox("Источник видео")
        camera_group_layout = QVBoxLayout()  # Создаем временный QVBoxLayout для камеры
        self.camera_select = QComboBox()
        self.camera_select.setStyleSheet("padding: 10px; font-size: 16px;")
        # **Политика размера для ComboBox, растягивается по горизонтали**
        self.camera_select.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        camera_group_layout.addWidget(self.camera_select)  # Добавляем выпадающий список в QVBoxLayout
        self.camera_group.setLayout(camera_group_layout)  # Устанавливаем layout для groupbox
        self.controls_panel_layout.addWidget(self.camera_group)  # Добавляем в controls_panel_layout

        self.populate_camera_select()

        # --- Группа "Выбор метода обнаружения лиц" ---
        self.method_group = QGroupBox("Метод обнаружения лиц")
        method_group_layout = QHBoxLayout()  # Создаем временный QHBoxLayout для метода
        self.method_label = QLabel("Метод:")
        self.method_label.setStyleSheet("font-size: 16px;")
        self.method_select = QComboBox()
        self.method_select.addItem("Face Recognition", "face_recognition")
        self.method_select.addItem("FAN", "fan")
        self.method_select.setCurrentIndex(0)
        self.method_select.setStyleSheet("padding: 10px; font-size: 16px;")
        # **Политика размера для ComboBox, растягивается по горизонтали**
        self.method_select.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        method_group_layout.addWidget(self.method_label)  # Добавляем label
        method_group_layout.addWidget(self.method_select)  # Добавляем combobox
        method_group_layout.addWidget(self.fan_init_indicator)  # Добавляем индикатор FAN
        self.method_group.setLayout(method_group_layout)  # Устанавливаем layout для groupbox
        self.controls_panel_layout.addWidget(self.method_group)  # Добавляем в controls_panel_layout

        # --- Группа "Загрузка изображений" ---
        self.load_images_group = QGroupBox("Загрузка изображений")
        self.load_images_layout = QHBoxLayout()  # Горизонтальный layout для кнопок загрузки
        # **Политика размера для load_images_layout, растягивается по горизонтали**
        self.load_images_layout.setSizeConstraint(QHBoxLayout.SetMinAndMaxSize)

        # Добавляем load_face_layout и load_clothes_layout в горизонтальный layout
        self.load_images_layout.addLayout(self.load_face_layout)
        self.load_images_layout.addLayout(self.load_clothes_layout)
        # Добавляем spacer, чтобы кнопки были слева
        self.load_images_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.load_images_group.setLayout(self.load_images_layout)
        self.controls_panel_layout.addWidget(self.load_images_group)  # Добавляем в controls_panel_layout

        # --- Группа "Управление наложением" ---
        self.control_group = QGroupBox("Управление наложением")
        self.control_layout = QHBoxLayout()  # Горизонтальный layout для кнопок Старт/Стоп и режима наложения
        # **Политика размера для control_layout, растягивается по горизонтали**
        self.control_layout.setSizeConstraint(QHBoxLayout.SetMinAndMaxSize)

        # Кнопки "Старт" и "Стоп" (уже в QHBoxLayout)
        self.control_layout.addLayout(self.buttons_layout)
        # Spacer между кнопками и режимом наложения
        self.control_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        # --- Группа "Режим наложения" (теперь в horizontal layout группы "Управление наложением") ---
        # Используем QVBoxLayout для радиокнопок внутри horizontal layout группы "Управление наложением"
        self.overlay_mode_v_layout = QVBoxLayout()

        # Группа радиокнопок (уже инициализированы)
        self.overlay_mode_v_layout.addWidget(self.overlay_mode_label)  # Label "Режим наложения"
        self.overlay_mode_v_layout.addWidget(self.no_overlay_radio)
        self.overlay_mode_v_layout.addWidget(self.face_overlay_radio)
        self.overlay_mode_v_layout.addWidget(self.clothes_overlay_radio)
        self.overlay_mode_v_layout.addWidget(self.both_overlays_radio)

        self.control_layout.addLayout(self.overlay_mode_v_layout)  # Добавляем QVBoxLayout с радиокнопками в QHBoxLayout группы "Управление наложением"
        self.control_group.setLayout(self.control_layout)
        self.controls_panel_layout.addWidget(self.control_group)  # Добавляем в controls_panel_layout

        # Добавляем левую и правую части в основной QHBoxLayout
        self.layout.addLayout(self.video_area_layout, stretch=2)  # Левая часть (видео), занимает 2/3 пространства
        self.layout.addLayout(self.controls_panel_layout, stretch=1)  # Правая часть (элементы управления), занимает 1/3 пространства

        self.fan = None
        self.source_image = None
        self.source_face_landmarks = None
        self.source_face = None
        self.source_mask = None
        self.source_points = None

        self.clothes_image = None
        self.clothes_mask = None
        self.clothes_keypoints = None
        self.clothes_overlay_active = False

        self.pose_estimator = PoseEstimator()

        self.video_capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.no_overlay_radio.setChecked(True)

        # Инициализация модели FAN (с индикацией)
        self.init_fan_model()

        self.method_select.currentIndexChanged.connect(self.change_method)
        self.current_method = self.method_select.currentData()
    def init_fan_model(self):
        """Инициализация FAN с индикацией."""
        if self.method_select.currentData() == 'fan':
            movie = self.load_movie_icon("loading.gif") # Загружаем GIF иконку
            if movie: # Проверяем, что иконка загрузилась успешно
                self.fan_init_indicator.setMovie(movie) # Показываем "загрузку"
                movie.start()
                QApplication.processEvents() # Даем GUI обновиться и показать индикатор
            try:
                self.fan = initialize_fan()
                success_icon = self.load_static_icon("success.png") # Загружаем статическую иконку успеха
                if success_icon: # Проверяем, что иконка загрузилась успешно
                    self.fan_init_indicator.setPixmap(success_icon) # Показываем "успех"
                if movie and movie.state() == QMovie.Running: # Проверяем, что movie не None и запущен
                    movie.stop() # Останавливаем GIF, если вдруг был запущен
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось инициализировать FAN: {e}")
                error_icon = self.load_static_icon("error.png") # Загружаем статическую иконку ошибки
                if error_icon: # Проверяем, что иконка загрузилась успешно
                    self.fan_init_indicator.setPixmap(error_icon) # Показываем "ошибку"
                if movie and movie.state() == QMovie.Running: # Проверяем, что movie не None и запущен
                    movie.stop() # Останавливаем GIF, если вдруг был запущен
        else:
            self.fan_init_indicator.clear() # Очищаем индикатор, если метод не FAN


    def load_movie_icon(self, filename):
        """Загружает иконку-GIF из файла или None в случае ошибки."""
        filepath = os.path.join("icons", filename) # Полный путь к файлу
        if not os.path.exists(filepath): # Проверяем, существует ли файл
            logger.error(f"Файл иконки не найден: {filepath}")
            return None
        movie = QMovie(filepath)
        if not movie.isValid(): # Проверяем, что GIF загрузился корректно
            logger.error(f"Ошибка загрузки GIF иконки: {filepath}")
            return None
        movie.setScaledSize(QSize(20, 20))
        return movie

    def load_static_icon(self, filename):
        """Загружает статическую иконку из файла или None в случае ошибки."""
        filepath = os.path.join("icons", filename) # Полный путь к файлу
        if not os.path.exists(filepath): # Проверяем, существует ли файл
            logger.error(f"Файл иконки не найден: {filepath}")
            return None
        pixmap = QPixmap(filepath)
        if pixmap.isNull(): # Проверяем, что PNG/Pixmap загрузился корректно
            logger.error(f"Ошибка загрузки статической иконки: {filepath}")
            return None
        return pixmap.scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)

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
            if self.current_method == 'fan':
                self.init_fan_model() # Инициализируем FAN при выборе метода FAN
            elif self.current_method != 'fan' and self.fan is not None:
                del self.fan
                self.fan = None
                self.fan_init_indicator.clear() # Очищаем индикатор, если метод не FAN
            QMessageBox.information(self, "Метод изменен", f"Выбран метод: {self.method_select.currentText()}")
        except Exception as e:
            logger.error("Ошибка при изменении метода обнаружения", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Не удалось изменить метод обнаружения: {e}")

    def load_source(self):
        try:
            loading_movie = self.load_movie_icon("loading.gif") # Загружаем GIF иконку
            if loading_movie: # Проверяем, что иконка загрузилась успешно
                self.load_face_indicator.setMovie(loading_movie) # Показываем "загрузку"
                loading_movie.start()
                QApplication.processEvents() # Обновляем GUI
            selected_method = self.current_method
            self.source_image, self.source_face_landmarks = load_source_image(method=selected_method, fa=self.fan, source=True)
            if self.source_image is not None and self.source_face_landmarks is not None:
                self.source_face, self.source_mask, self.source_points = prepare_source_face(
                    self.source_image, self.source_face_landmarks)
                if self.source_face is None or self.source_mask is None or self.source_points is None:
                    QMessageBox.critical(self, "Ошибка", "Не удалось подготовить исходное лицо.")
                    error_icon = self.load_static_icon("error.png") # Загружаем статическую иконку ошибки
                    if error_icon: # Проверяем, что иконка загрузилась успешно
                        self.load_face_indicator.setPixmap(error_icon) # Показываем "ошибку"
                    if loading_movie and loading_movie.state() == QMovie.Running: # Проверяем, что loading_movie не None и запущен
                        loading_movie.stop()
                    return
                self.overlay_active = True
                self.update_overlay_status()
                QMessageBox.information(self, "Информация", "Исходное изображение успешно загружено.")
                success_icon = self.load_static_icon("success.png") # Загружаем статическую иконку успеха
                if success_icon: # Проверяем, что иконка загрузилась успешно
                    self.load_face_indicator.setPixmap(success_icon) # Показываем "успех"
                if loading_movie and loading_movie.state() == QMovie.Running: # Проверяем, что loading_movie не None и запущен
                    loading_movie.stop()
            else:
                QMessageBox.warning(self, "Предупреждение", "Не удалось загрузить исходное изображение.")
                self.load_face_indicator.clear() # Очищаем индикатор (или можно показать "ошибку", если хочешь)
                if loading_movie and loading_movie.state() == QMovie.Running: # Проверяем, что loading_movie не None и запущен
                    loading_movie.stop()
        except Exception as e:
            logger.error("Ошибка при загрузке исходного изображения", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить исходное изображение: {e}")
            error_icon = self.load_static_icon("error.png") # Загружаем статическую иконку ошибки
            if error_icon: # Проверяем, что иконка загрузилась успешно
                self.load_face_indicator.setPixmap(error_icon) # Показываем "ошибку"
            if loading_movie and loading_movie.state() == QMovie.Running: # Проверяем, что loading_movie не None и запущен
                loading_movie.stop()

    def start_face_swap(self):
        try:
            selected_mode = self.overlay_mode_group.checkedButton().text()  # Получаем выбранный режим
            logger.info(f"Выбран режим наложения: {selected_mode}")

            if selected_mode == "Face Only" and self.source_image is None:
                QMessageBox.warning(self, "Предупреждение", "Пожалуйста, загрузите изображение лица.")
                return

            if selected_mode == "Clothes Only" and self.clothes_image is None:
                QMessageBox.warning(self, "Предупреждение", "Пожалуйста, загрузите изображение одежды.")
                return

            if selected_mode == "Both" and (self.source_image is None or self.clothes_image is None):
                QMessageBox.warning(self, "Предупреждение", "Пожалуйста, загрузите изображение лица и одежды.")
                return

            if self.video_capture is None:
                self.change_camera()
                if self.video_capture is None:
                    return

            if not self.video_capture.isOpened():
                QMessageBox.critical(self, "Ошибка", "Камера не открыта.")
                return

            self.overlay_active = selected_mode in ["Face Only", "Both"]  # Активируем наложение лица
            self.clothes_overlay_active = selected_mode in ["Clothes Only", "Both"]  # Активируем наложение одежды

            logger.info(f"overlay_active: {self.overlay_active}, clothes_overlay_active: {self.clothes_overlay_active}")

            self.timer.start(30)

        except Exception as e:
            logger.error("Ошибка при запуске наложения", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить наложение: {e}")


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
                logger.warning("Video capture not opened.")
                return

            ret, frame = self.video_capture.read()
            if not ret:
                logger.error("Не удалось прочитать кадр из видеопотока.")
                return

            # Получаем ключевые точки тела из изображения
            keypoints = self.pose_estimator.get_keypoints(frame)
            if keypoints is not None:
                logger.info(f"Detected {len(keypoints)} keypoints.")
            else:
                logger.warning("No keypoints detected.")

            # Преобразуем кадр в RGB после MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Обнаружение ключевых точек лица
            if self.current_method == 'fan':
                face_landmarks_list = detect_face_landmarks(frame_rgb, method=self.current_method, fa=self.fan, source=False)
            else:
                face_landmarks_list = detect_face_landmarks(frame_rgb, method=self.current_method, source=False)

            # Наложение лица, если нужно
            if self.face_overlay_radio.isChecked() or self.both_overlays_radio.isChecked():
                if self.source_face is not None and self.source_mask is not None:
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
                            logger.warning("Target points out of frame boundaries. Skipping face overlay.")
                            continue

                        try:
                            M, _ = cv2.findHomography(self.source_points, target_points)
                            if M is None:
                                logger.warning("Homography matrix is None. Skipping face overlay.")
                                continue

                            transformed_face = cv2.warpPerspective(
                                self.source_face, M, (frame_rgb.shape[1], frame_rgb.shape[0])
                            )
                            transformed_mask = cv2.warpPerspective(
                                self.source_mask, M, (frame_rgb.shape[1], frame_rgb.shape[0])
                            )

                            if transformed_face.size == 0 or transformed_mask.size == 0:
                                logger.warning("Transformed face or mask is empty. Skipping face overlay.")
                                continue

                            center_point = (int(target_points[:, 0].mean()), int(target_points[:, 1].mean()))
                            center_x = max(0, min(center_point[0], frame_rgb.shape[1] - 1))
                            center_y = max(0, min(center_point[1], frame_rgb.shape[0] - 1))
                            center_point_corrected = (center_x, center_y)

                            if not (0 <= center_x < frame_rgb.shape[1] and 0 <= center_y < frame_rgb.shape[0]):
                                logger.warning("Center point out of frame boundaries. Skipping face overlay.")
                                continue

                            if cv2.countNonZero(transformed_mask) == 0:
                                logger.warning("Transformed mask is empty. Skipping face overlay.")
                                continue

                            try:
                                frame_rgb = cv2.seamlessClone(
                                    transformed_face, frame_rgb, transformed_mask, center_point_corrected, cv2.NORMAL_CLONE
                                )
                                logger.info("Face overlay applied successfully.")
                            except cv2.error as e:
                                logger.error(f"OpenCV error during seamlessClone: {e}", exc_info=True)
                                continue

                        except Exception as e:
                            logger.error(f"Exception during face overlay: {e}", exc_info=True)
                            continue

            # Наложение одежды, если нужно
            if self.clothes_overlay_radio.isChecked() or self.both_overlays_radio.isChecked():
                if self.clothes_image is not None and self.clothes_mask is not None and self.clothes_keypoints is not None and keypoints is not None:
                    logger.info("Applying clothes overlay.")
                    frame_rgb = overlay_clothes(frame_rgb, self.clothes_image, self.clothes_mask, self.clothes_keypoints, keypoints)
                else:
                    logger.warning("Clothes overlay parameters missing. Skipping clothes overlay.")

            # Проверка, что frame_rgb не пустой перед изменением размера
            if frame_rgb is not None and frame_rgb.size > 0:
                frame_rgb = cv2.resize(frame_rgb, (1440, 810), interpolation=cv2.INTER_LINEAR)
                self.display_image(frame_rgb)
            else:
                logger.error("frame_rgb is empty. Skipping resize and display.")

        except Exception as e:
            logger.error("Непредвиденная ошибка в методе update_frame", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Произошла непредвиденная ошибка: {e}")
            self.overlay_active = False
            self.clothes_overlay_active = False

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
                self.overlay_status_indicator.setPixmap(self.load_static_icon("active.png")) # Иконка "активно"
                self.overlay_status_text.setText("Наложение: Активно") # Текст "Активно"
                self.overlay_status_text.setStyleSheet("color: green; font-size: 14px;") # Зеленый цвет текста
                self.overlay_status_indicator.setToolTip("Overlay Active")
            else:
                self.overlay_status_indicator.setPixmap(self.load_static_icon("inactive.png")) # Иконка "неактивно"
                self.overlay_status_text.setText("Наложение: Неактивно") # Текст "Неактивно"
                self.overlay_status_text.setStyleSheet("color: red; font-size: 14px;") # Красный цвет текста
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

    def load_clothes_image(self):
        try:
            loading_movie = self.load_movie_icon("loading.gif") # Загружаем GIF иконку
            if loading_movie: # Проверяем, что иконка загрузилась успешно
                self.load_clothes_indicator.setMovie(loading_movie) # Показываем "загрузку"
                loading_movie.start()
                QApplication.processEvents() # Обновляем GUI
            else:
                logger.error("Не удалось загрузить иконку загрузки для одежды.")
                # Handle the case where the loading movie couldn't be loaded, maybe set a static error icon?
                error_icon = self.load_static_icon("error.png") # Загружаем статическую иконку ошибки
                if error_icon: # Проверяем, что иконка загрузилась успешно
                    self.load_clothes_indicator.setPixmap(error_icon) # Показываем "ошибку"
                return # Exit the function early if loading movie failed

            Tk().withdraw()
            clothes_image_path = askopenfilename(title='Выберите изображение одежды')
            if not clothes_image_path:
                print("Файл не выбран.")
                self.load_clothes_indicator.clear() # Очищаем индикатор при отмене выбора файла
                if loading_movie: # Check if loading_movie is valid before trying to stop
                    loading_movie.stop()
                return

            self.clothes_image = face_recognition.load_image_file(clothes_image_path)
            logger.info(f"Clothes Image Loaded: {self.clothes_image.shape}")

            self.clothes_keypoints = self.pose_estimator.get_keypoints(self.clothes_image)
            if self.clothes_keypoints is not None:
                logger.info(f"Clothes Keypoints Detected: {self.clothes_keypoints.shape}")
                self.clothes_mask = create_clothes_mask(self.clothes_image, self.clothes_keypoints)
                if self.clothes_mask is None:
                    QMessageBox.warning(self, "Предупреждение", "Не удалось создать маску для одежды.")
                    error_icon = self.load_static_icon("error.png") # Показываем "ошибку"
                    if error_icon: # Проверяем, что иконка загрузилась успешно
                        self.load_clothes_indicator.setPixmap(error_icon) # Показываем "ошибку"
                    if loading_movie: # Check if loading_movie is valid before trying to stop
                        loading_movie.stop()
                    return

                QMessageBox.information(self, "Информация", "Изображение одежды успешно загружено.")
                success_icon = self.load_static_icon("success.png") # Показываем "успех"
                if success_icon: # Проверяем, что иконка загрузилась успешно
                    self.load_clothes_indicator.setPixmap(success_icon) # Показываем "успех"
                if loading_movie: # Check if loading_movie is valid before trying to stop
                    loading_movie.stop()
            else:
                QMessageBox.warning(self, "Предупреждение", "Не удалось определить ключевые точки тела на изображении одежды.")
                error_icon = self.load_static_icon("error.png") # Показываем "ошибку"
                if error_icon: # Проверяем, что иконка загрузилась успешно
                    self.load_clothes_indicator.setPixmap(error_icon) # Показываем "ошибку"
                if loading_movie: # Check if loading_movie is valid before trying to stop
                    loading_movie.stop()

        except Exception as e:
            logger.error("Ошибка при загрузке изображения одежды", exc_info=True)
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить изображение одежды: {e}")
            error_icon = self.load_static_icon("error.png") # Показываем "ошибку"
            if error_icon: # Проверяем, что иконка загрузилась успешно
                self.load_clothes_indicator.setPixmap(error_icon) # Показываем "ошибку"
            if loading_movie: # Check if loading_movie is valid before trying to stop
                loading_movie.stop()


    def toggle_clothes_overlay(self, state):
        self.clothes_overlay_active = state == Qt.Checked

def main():
    app = QApplication(sys.argv)
    window = FaceSwapApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()