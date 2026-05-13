import time
import os
import threading 

class detector:
    def __init__(self, model, frame):
        self.model = model
        self.frame = frame

    def detect(self):
        results = self.model(self.frame, verbose=False)
        return results
    def log_detection(self, class_name, confidence):
        if os.path.exists('logs/'):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open("logs/detection_log.txt", "a") as f:
                f.write(f"{timestamp} - Detected {class_name} with confidence {confidence:.2f}\n")
        else:
            os.makedirs('logs/')
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open("logs/detection_log.txt", "a") as f:
                f.write(f"{timestamp} - Detected {class_name} with confidence {confidence:.2f}\n")