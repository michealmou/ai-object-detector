import os
import cv2
import time
import pyttsx3
import threading
from ultralytics import YOLO
from voice import Voice
from detector import detector
from config import (
    ALLOWED_CLASSES,
    CAMERA_INDEX,
    COOLDOWN_TIME,
    DATASET_DIR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IMAGE_PREFIX,
    MODEL_PATH,
    WINDOW_NAME,
    TIME,
)

model = YOLO(MODEL_PATH)
#launch the camera and start capturing the video, camera feed should be inverted
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)


last_seen = {}
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
    results = model(frame, verbose=False, stream=True)

    
    count = {}
    for r in results:
        for box in r.boxes:

        # Bounding box coordinates
        # x1,y1 = top-left
        # x2,y2 = bottom-right
            x1, y1, x2, y2 = box.xyxy[0]

        # Confidence score (0 to 1)
            confidence = float(box.conf[0])

        # Class ID (number)
            class_id = int(box.cls[0])

        # Convert class ID to actual object name
            class_name = model.names[class_id]
            
                
            current_time = time.time()
            if class_name not in ALLOWED_CLASSES:
                continue
            last_time = last_seen.get(class_name, 0)
            if current_time - last_time > COOLDOWN_TIME:
                print(f"Detected {class_name} with confidence {confidence:.2f}")
                d = detector(model, frame)
                d.log_detection(class_name, confidence)
                threading.Thread(target=Voice().speak, args=(f"{class_name} detected",)).start()
                last_seen[class_name] = current_time
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 6)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    
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

