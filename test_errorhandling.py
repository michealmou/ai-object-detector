import cv2
import numpy as np
import os
import tempfile
from errorhandling import ErrorHandling
from config import (
    CAMERA_INDEX,
    DATASET_DIR,
    COOLDOWN_TIME,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)

print("=" * 60)
print("REAL-TIME ERROR HANDLING TESTS")
print("=" * 60)

object1 = ErrorHandling()

# ========== TEST 1: CAMERA INITIALIZATION ==========
print("\n[TEST 1] Camera Initialization")
print("-" * 60)
cap = object1.init_camera(CAMERA_INDEX)
if cap is not None and cap.isOpened():
    print("✓ Camera initialized successfully.")
    cap.release()
else:
    print("✗ Camera initialization failed (is camera connected?).")

# ========== TEST 2: MODEL FILE CHECK ==========
print("\n[TEST 2] Model File Existence")
print("-" * 60)
print("Checking for yolov8n.pt...")
if object1.model_file_not_found("yolov8n.pt"):
    print("✓ Model file exists.")
else:
    print("✗ Model file missing (download it first).")

# ========== TEST 3: INVALID FRAME ERROR ==========
print("\n[TEST 3] Frame Validation")
print("-" * 60)

# Test with None frame
print("Testing None frame...")
result = object1.invalid_frame_error(None)
if not result:
    print("✓ Correctly detected None frame.")
else:
    print("✗ Failed to detect None frame.")

# Test with empty frame
print("Testing empty frame...")
empty_frame = np.array([], dtype=np.uint8)
result = object1.invalid_frame_error(empty_frame)
if not result:
    print("✓ Correctly detected empty frame.")
else:
    print("✗ Failed to detect empty frame.")

# Test with valid frame
print("Testing valid frame...")
valid_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
result = object1.invalid_frame_error(valid_frame)
if result:
    print("✓ Correctly validated valid frame.")
else:
    print("✗ Failed to validate valid frame.")

# ========== TEST 4: WINDOW DISPLAY ERROR ==========
print("\n[TEST 4] Window Display")
print("-" * 60)
print("Testing window display (will briefly show a black window)...")
try:
    result = object1.window_display_error("Test Window", valid_frame)
    if result:
        print("✓ Window displayed successfully.")
        cv2.destroyAllWindows()
    else:
        print("✗ Window display failed (headless mode?).")
except Exception as e:
    print(f"✗ Window display error: {e}")

# ========== TEST 5: CONFIGURATION VALIDATION ==========
print("\n[TEST 5] Configuration Validation")
print("-" * 60)
print("Checking config values...")
print(f"  Frame Width: {FRAME_WIDTH}")
print(f"  Frame Height: {FRAME_HEIGHT}")
print(f"  Cooldown Time: {COOLDOWN_TIME}")
result = object1.configuration_error()
if result:
    print("✓ Configuration is valid.")
else:
    print("✗ Configuration has invalid values.")

# ========== TEST 6: DATASET DIRECTORY & FILE SAVE ==========
print("\n[TEST 6] Dataset Directory & File Save")
print("-" * 60)
print(f"Testing save to: {DATASET_DIR}")
try:
    # Create temp directory to test safely
    with tempfile.TemporaryDirectory() as tmpdir:
        test_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        test_filepath = os.path.join(tmpdir, "test_image.jpg")
        
        result = object1.dataset_directory_permission_error(tmpdir, test_filepath, test_frame)
        if result and os.path.exists(test_filepath):
            file_size = os.path.getsize(test_filepath)
            print(f"✓ Image saved successfully ({file_size} bytes).")
        else:
            print("✗ Failed to save image.")
except Exception as e:
    print(f"✗ File save error: {e}")

# ========== TEST 7: LOG FILE ERROR ==========
print("\n[TEST 7] Log File Writing")
print("-" * 60)
print("Testing log file write...")
try:
    object1.log_file_error("person", 0.95)
    if os.path.exists("logs/detection_log.txt"):
        with open("logs/detection_log.txt", "r") as f:
            content = f.read()
            if "person" in content and "0.95" in content:
                print("✓ Log file written successfully.")
                print(f"  Log entry: {content.strip().split(chr(10))[-1]}")
            else:
                print("✗ Log file exists but content is incorrect.")
    else:
        print("✗ Log file was not created.")
except Exception as e:
    print(f"✗ Log file error: {e}")

# ========== TEST 8: TEXT-TO-SPEECH ERROR ==========
print("\n[TEST 8] Text-to-Speech")
print("-" * 60)
print("Testing voice announcement (you may hear 'test message detected')...")
try:
    result = object1.text_to_speech_error("test message detected")
    if result:
        print("✓ Voice thread started successfully.")
    else:
        print("✗ Voice failed (no audio device or TTS not available).")
except Exception as e:
    print(f"✗ Voice error: {e}")

# ========== TEST 9: RESOURCE MONITORING ==========
print("\n[TEST 9] Memory/Resource Monitoring")
print("-" * 60)
print("Checking memory usage...")
try:
    result = object1.resource_exhaustion()
    if result:
        print("✓ Memory usage is normal.")
    else:
        print("⚠ Warning: High memory usage detected.")
except Exception as e:
    print(f"✗ Resource monitoring error: {e}")

# ========== SUMMARY ==========
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("All error handlers have been tested.")
print("Check output above for ✓ (pass) or ✗ (fail) markers.")
print("=" * 60)
