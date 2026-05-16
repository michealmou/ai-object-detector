# AI Object Detector

A real-time object detection system using YOLOv8, OpenCV, and Python. Detects objects from a webcam feed, provides text-to-speech announcements, saves detected frames, and maintains comprehensive logs.

## Features

- **Real-time detection** using YOLOv8 nano model
- **Configurable classes** (person, apple, bottle by default)
- **Cooldown-based announcements** via text-to-speech (prevents spam)
- **Frame capture** on keystroke (save to `dataset/`)
- **FPS monitoring** with exponential smoothing
- **Structured logging** to console and `logs/app.log`
- **Error handling** for camera failures, model loading, file I/O
- **Resource monitoring** (memory usage detection)
- **Comprehensive test suite** with error simulation

## Requirements

- Python 3.8+
- Webcam (or camera device at index 0, configurable)
- 500 MB+ disk space (for YOLOv8 model download)
- Text-to-speech capable system (or runs headless with warnings)

## Installation

1. **Clone or download the repository:**
   ```bash
   git clone <repo-url>
   cd ai-object-detector
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the YOLOv8 model (first run will auto-download):**
   - The script will fetch `yolov8n.pt` (~6.3 MB) on first execution.
   - Or manually download from: https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
   - Place in project root as `yolov8n.pt`.

## Configuration

Edit `config.py` to customize:

```python
MODEL_PATH = 'yolov8n.pt'           # Path to YOLO model
CAMERA_INDEX = 0                    # Camera device (0 = default)
WINDOW_NAME = 'AI Detector'         # Display window title
FRAME_WIDTH = 1280                  # Resize width
FRAME_HEIGHT = 720                  # Resize height
ALLOWED_CLASSES = ['person', 'apple', 'bottle']  # Objects to detect
COOLDOWN_TIME = 3                   # Seconds between announcements
DATASET_DIR = 'dataset'             # Save location for captured frames
```

## Usage

### Run the detector:

```bash
python main.py
```

### Keyboard controls (during execution):

- **`s`** – Save current frame to `dataset/`
- **`q`** – Quit application
- **`+` / `-`** – Adjust resolution (future feature)

### Real-time output:

- **Console**: FPS counter, detection logs, errors
- **Window**: Live feed with bounding boxes and confidence scores
- **Logs**: Persistent records in `logs/app.log`
- **Audio**: Text-to-speech announcements (if TTS available)

## Logging

### Configuration

Logging is initialized in `main.py`:
- **Console handler** (INFO level): shows important events
- **File handler** (DEBUG level): detailed traces to `logs/app.log`

### Enable DEBUG logging:

```bash
$env:LOG_LEVEL='DEBUG'; python main.py   # PowerShell
LOG_LEVEL=DEBUG python main.py            # Linux/macOS
```

### Log structure

```
2026-05-16 00:50:58,902 - INFO - Detected person with confidence 0.88
2026-05-16 00:50:58,963 - INFO - Model loaded successfully.
```

## Testing

### Run the full test suite:

```bash
python test_errorhandling.py
```

Tests cover:
- Camera initialization
- Model file existence
- Frame validation (None, empty, valid)
- Window display
- Configuration validation
- Dataset directory and file save
- Log file writing
- Text-to-speech
- Resource/memory monitoring

All test results are logged to `logs/app.log` and printed to console.

## Project Structure

```
ai-object-detector/
├── main.py                      # Entry point; camera loop + logging setup
├── detector.py                  # Detector class; YOLO inference + FPS calculation
├── config.py                    # Configuration constants
├── errorhandling.py             # Error handling utilities
├── voice.py                     # Text-to-speech wrapper (pyttsx3)
├── utils.py                     # Utility functions (currently empty)
├── test_errorhandling.py        # Test suite for error handlers
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── logs/
│   ├── app.log                  # Application and test logs (rotating)
│   └── detection_log.txt        # Detection event history
├── dataset/                     # Saved frames from 's' key press
├── guides/
│   ├── DETECTOR_REFACTOR_GUIDE.txt
│   ├── ERROR_HANDLING_GUIDE.txt
│   ├── LOGGING_MIGRATION_GUIDE.txt
│   ├── GITHUB_ISSUES.md
│   └── STEP_BY_STEP_GUIDE.txt
└── yolov8n.pt                   # YOLOv8 nano model (auto-downloaded)
```

## Real-Time Processing Fundamentals

### Frame Capture

A frame is one still image from a video stream. OpenCV captures frames in a loop:

```python
ret, frame = cap.read()  # ret = success, frame = image data
```

### Pixel and Array Concepts

- A **pixel** is one dot in an image; each has RGB (or BGR in OpenCV) values.
- OpenCV stores frames as NumPy arrays: `shape = (height, width, 3)`.
- Values range from 0–255 (`uint8`).

Check frame properties:

```python
print(frame.shape)      # e.g., (720, 1280, 3)
print(frame.dtype)      # uint8
```

### RGB vs BGR

OpenCV uses **BGR** by default (not RGB). Convert if needed:

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

### Processing Loop

1. Capture frame
2. Resize/preprocess
3. Run YOLO inference
4. Draw boxes and labels
5. Display frame
6. Check for user input

Efficient loops maintain smooth, real-time performance.

## Troubleshooting

### Camera not found

- Ensure camera is connected and not in use by another app.
- Check `config.py`: `CAMERA_INDEX = 0` (may be 1, 2, etc. on multi-camera systems).
- Run `python test_errorhandling.py` to test camera initialization.

### No TTS audio

- Text-to-speech may fail on headless systems (SSH, Docker).
- System continues without audio (non-fatal error).
- Check console for `[Voice failed...]` messages.

### High memory usage

- Adjust `FRAME_WIDTH` and `FRAME_HEIGHT` to smaller values.
- Monitor with `LOG_LEVEL=DEBUG` to see memory warnings.

### Model download fails

- Manually download `yolov8n.pt` and place in project root.
- Or set `MODEL_PATH` to an existing local model.

### Encoding errors in logs (Windows)

- Non-ASCII symbols may cause `UnicodeEncodeError` on Windows.
- Logs now use ASCII markers: `[PASS]`, `[FAIL]`, `[WARN]`.
- Console output should display correctly.

## Performance Notes

- **FPS**: Typical 20–30 FPS on modern hardware (GPU recommended for higher throughput).
- **Model size**: YOLOv8 nano (~6.3 MB) is optimized for real-time inference.
- **Logging overhead**: DEBUG logs are written asynchronously to minimize frame drops.

## Contributing

- See `guides/` folder for refactoring guides and issue tracking.
- Test your changes with `python test_errorhandling.py`.
- Update logs via `LOG_LEVEL=DEBUG` before committing.

## References

- **OpenCV**: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- **YOLOv8**: https://docs.ultralytics.com/
- **NumPy**: https://numpy.org/doc/stable/user/
- **pyttsx3**: https://github.com/nateshmbhat/pyttsx3

## License

[Add your license here, e.g., MIT]

## Authors

AI Object Detector Project Team