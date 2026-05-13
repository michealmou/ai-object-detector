import time
MODEL_PATH = 'yolov8n.pt'
CAMERA_INDEX = 0
WINDOW_NAME = 'AI Detector'
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ALLOWED_CLASSES = ['person', 'apple']
COOLDOWN_TIME = 3
DATASET_DIR = 'dataset'
IMAGE_PREFIX = 'object'
TIME = time.strftime("%Y%m%d-%H%M%S")
