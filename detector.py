import os
import threading
import time

import cv2

from config import ALLOWED_CLASSES, COOLDOWN_TIME
from voice import Voice


class Detector:
    def __init__(self, model):
        self.model = model
        self.last_seen = {}

    def detect(self, frame):
        results = self.model(frame, verbose=False, stream=True)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                current_time = time.time()
                if class_name not in ALLOWED_CLASSES:
                    continue

                last_time = self.last_seen.get(class_name, 0)
                if current_time - last_time > COOLDOWN_TIME:
                    print(f"Detected {class_name} with confidence {confidence:.2f}")
                    self.log_detection(class_name, confidence)
                    threading.Thread(target=Voice().speak, args=(f"{class_name} detected",)).start()
                    self.last_seen[class_name] = current_time

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 6)
                cv2.putText(
                    frame,
                    f"{class_name} {confidence:.2f}",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        return frame

    def log_detection(self, class_name, confidence):
        if not os.path.exists('logs/'):
            os.makedirs('logs/')

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open("logs/detection_log.txt", "a") as f:
            f.write(f"{timestamp} - Detected {class_name} with confidence {confidence:.2f}\n")


detector = Detector