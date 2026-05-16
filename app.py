import os
import logging
import cv2
import sys
from PyQt6.QtWidgets import (QApplication, QLabel, QLineEdit, QMainWindow, QHBoxLayout, QPushButton, QWidget)
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

# Set up logging

level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=level,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
console_handler = logging.StreamHandler()
console_handler.setLevel(level)

# save logging to files (debug, info, warning, error, critical)

os.makedirs("logs/", exist_ok=True)
file_handler = RotatingFileHandler(
    "logs/app.log",
    maxBytes=1_000_000,
    backupCount=3
)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logging.getLogger().addHandler(file_handler)

# Load the YOLO model and initialize the detector
logging.info("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print(model.names)
logging.info("Model loaded successfully.")
logging.info("Initializing detector...")
detector = Detector(model)

#launch the camera and start capturing the video, camera feed should be inverted

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
# PyQt6 renders the video feed in the application window, so we do not
# create an OpenCV HighGUI window here. That avoids crashes on headless or
# headless-style OpenCV builds that do not include GUI support.
