import os
import cv2
import threading
import psutil
import time
import logging
from ultralytics import YOLO

from config import COOLDOWN_TIME, FRAME_HEIGHT, FRAME_WIDTH
from voice import Voice
logger = logging.getLogger(__name__)

class ErrorHandling:
    def init_camera(self, camera_index):
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                raise Exception("Could not open camera")
            return cap
        except Exception as e:      
            logger.error(f"Error initializing camera {camera_index}: {e}")
            return None
        
    def model_file_not_found(self, model_path):
        if not os.path.exists(model_path):
            logger.error(f"Error: Model file not found at {model_path}")
            logger.info("Download from: https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt")
            return False
        return True
    def text_to_speech_error(self, message):
        try:
            threading.Thread(target=Voice().speak, args=(message,), daemon=True).start()
            return True
        except Exception as e:
            logger.error(f"Voice failed (continuing anyway): {e}")
            return False
    
    def dataset_directory_permission_error(self, dataset_dir, filepath, frame):
        try:
            if not os.path.exists(dataset_dir):
                os.makedirs(dataset_dir)
            cv2.imwrite(filepath, frame)
            return True
        except PermissionError:
            logger.error(f"Permission denied: Unable to save image to {dataset_dir}. Please check your permissions.")
            return False
        except OSError as e:
            logger.error(f"Could not save image to {dataset_dir}: {e}")
            return False
    def log_file_error(self, class_name, confidence):
        try:
            if not os.path.exists('logs/'):
                os.makedirs('logs/')
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open("logs/detection_log.txt", "a") as f:
                f.write(f"{timestamp} - Detected {class_name} with confidence {confidence:.2f}\n")
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")
    def invalid_frame_error(self, frame):
        if frame is None:
            logger.warning("Invalid frame captured, skipping...")
            return False
        if frame.size == 0:
            logger.warning("Empty frame captured, skipping...")
            return False
        return True
    def window_display_error(self, window_name, frame):
        try:
            cv2.imshow(window_name, frame)
            return True
        except Exception as e:
            logger.error(f"Error displaying window: {e} (running in headless mode?)")
            return False
    def configuration_error(self):   
        valid = True
        if FRAME_WIDTH <= 0 or FRAME_HEIGHT <= 0 or FRAME_WIDTH > 3840 or FRAME_HEIGHT > 2160:
            logger.error("Error: Invalid frame dimensions in configuration. Please check FRAME_WIDTH and FRAME_HEIGHT.")
            valid = False
        if COOLDOWN_TIME < 0:
            logger.error("Error: Invalid cooldown time in configuration. COOLDOWN_TIME must be non-negative.")
            valid = False
        return valid
    def keyboard_interrupt(self, cap):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nKeyboard interrupt received, exiting gracefully...")
        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()

    def resource_exhaustion(self):
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > 500:  # > 500MB
            logger.warning(f"Warning: High memory usage detected ({memory_mb:.1f} MB)")
            return False
        return True
