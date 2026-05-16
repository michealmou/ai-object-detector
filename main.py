import os
import logging
import cv2
import sys

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
logging.info("Model loaded successfully.")
logging.info("Initializing detector...")
detector = Detector(model)

#launch the camera and start capturing the video, camera feed should be inverted

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

#convert the video into frame by frame

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