import os
import logging
import cv2
import sys
import time
import app
from PyQt6.QtWidgets import *
from PyQt6.QtCore import QFileSystemWatcher
from styles import STYLE
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap 
from logging.handlers import RotatingFileHandler
from ultralytics import YOLO
from detector import Detector
from config import (
    CAMERA_INDEX,
    DATASET_DIR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IMAGE_PREFIX,
    MODEL_PATH,
    WINDOW_NAME,
    TIME,
)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Object Detector")
        self.setGeometry(100, 100, FRAME_WIDTH, FRAME_HEIGHT)
        self.detector = app.detector
        self.cap = app.cap
        self.setup_ui()
        self.timer = QTimer()
        self.timer.setInterval(30) # 30 ms for ~33 FPS
        self.timer.timeout.connect(self.process_frame)
        self.timer.start()
        self.watcher = QFileSystemWatcher()
        self.watcher.addPath("styles.py")
        self.watcher.fileChanged.connect(self.reload_styles)

    def reload_styles(self):
        import importlib
        import styles
        importlib.reload(styles)
        QApplication.instance().setStyleSheet(styles.STYLE)
        
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        main_layout.addWidget(self.make_header())
        main_layout.addWidget(self.make_content(), stretch=1)
        main_layout.addWidget(self.make_footer())

    def make_header(self):
        header = QWidget()
        header.setFixedHeight(50)
        header.setStyleSheet('background: #1a1a2e')
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 0, 16, 0)

        title = QLabel("AI Object Detector")
        title.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #00ff88; font-size: 14px;")

        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.fps_label)
        return header
    
    def make_content(self):
        #qlabel can display a video frame as a pixmap
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background: black;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return self.video_label

    def make_footer(self):
        footer = QWidget()
        footer.setFixedHeight(40)
        footer.setStyleSheet("background: #16213e;")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(16, 0, 16, 0)

        self.object_label = QLabel("Detected: None")
        self.object_label.setStyleSheet("color: #00ff88; font-size: 14px;")

        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setStyleSheet("color: #00ff88; font-size: 14px;")

        layout.addWidget(self.object_label)
        layout.addStretch()
        layout.addWidget(self.confidence_label)
        return footer
    def process_frame(self):
        # while true replacement
        start = time.time()
        ret, frame = self.cap.read()
        if not ret:
            logging.warning("Frame lost, retrying...")
            self.cap.release()
            self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            return
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = cv2.flip(frame, 1)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced_lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        frame = self.detector.detect(frame)
        #update fps label
        fps = 1.0 / (time.time() - start + 1e-6)
        self.fps_label.setText(f"FPS: {fps:.1f}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(img))
    
    def update_detection(self, name, confidence):
        self.object_label.setText(f'Detected: {name}')
        self.confidence_label.setText(f'Confidence: {confidence:.1%}')
    
    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_S:
            if not os.path.exists(DATASET_DIR):
                os.makedirs(DATASET_DIR)
            count = len([f for f in os.listdir(DATASET_DIR) if f.lower().endswith('.jpg')]) + 1
            file_name = f"{IMAGE_PREFIX}{count}_{TIME}.jpg"
            cv2.imwrite(os.path.join(DATASET_DIR, file_name), self.cap.read()[1])
            logging.info(f"Saved {DATASET_DIR}/{file_name}")
    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        cv2.destroyAllWindows()

app_qt = QApplication(sys.argv)
app_qt.setStyleSheet(STYLE)
window = MainWindow()
window.show()
sys.exit(app_qt.exec())


"""

while True:
    ret, frame = cap.read()
    if not ret:
        logging.warning("Frame lost, retrying...")
        cap.release()
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        continue    
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    frame = cv2.flip(frame, 1)

    # Run YOLO on the current frame and draw detections on the same frame window.
    #enhance image quality by adjusting brightness and contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    frame = detector.detect(frame)
    
    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if not os.path.exists(DATASET_DIR):
            os.makedirs(DATASET_DIR)
        count = len([f for f in os.listdir(DATASET_DIR) if f.lower().endswith('.jpg')]) + 1
        file_name = f"{IMAGE_PREFIX}{count}{TIME}.jpg"
        cv2.imwrite(os.path.join(DATASET_DIR, file_name), frame)
        logging.info(f"Saved {DATASET_DIR}/{file_name}")
    cv2.putText(frame, "Press 's' to save frame, '+'/'-' to adjust resolution, 'q' to quit", (10, FRAME_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow(WINDOW_NAME, frame)
    if key == ord('q'):
        break
        sys.exit(0)
    
    



cap.release()
cv2.destroyAllWindows()
"""