import os
import cv2
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

model = YOLO(MODEL_PATH)
detector = Detector(model)
#launch the camera and start capturing the video, camera feed should be inverted
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)


#convert the video into frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame lost, retrying...")
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
        print(f"Saved {DATASET_DIR}/{file_name}")

    if key == ord('q'):
        break

#release
cap.release()
cv2.destroyAllWindows()

