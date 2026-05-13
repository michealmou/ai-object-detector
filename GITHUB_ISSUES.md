# AI Object Detector GitHub Issues

This file breaks the roadmap into individual GitHub issues you can create in order. Each issue includes a clear goal and acceptance criteria so it is easy to track progress in GitHub Projects or a milestone board.

## Epic 1: Environment and Foundations

### 1. Set up the Python development environment
Goal: Prepare a clean Python workspace for development.

Acceptance criteria:
- Python 3.11+ is installed and verified.
- A virtual environment is created and activated.
- Core dependencies are installed.
- VS Code extensions for Python development are installed.

### 2. Establish the initial project structure
Goal: Create the first version of the app layout and supporting files.

Acceptance criteria:
- The project contains `main.py`, `detector.py`, `voice.py`, `utils.py`, `config.py`, `requirements.txt`, and `assets/`.
- Imports resolve cleanly between modules.
- The structure is documented in the README.

### 3. Build Python foundations through mini projects
Goal: Practice core Python concepts before starting the detector.

Acceptance criteria:
- A calculator script works.
- A number guessing game works.
- A timer utility works.
- A basic to-do list script works.

## Epic 2: Computer Vision Basics

### 4. Open and display the webcam feed
Goal: Confirm webcam capture works with OpenCV.

Acceptance criteria:
- The webcam opens successfully.
- Live video is displayed in a window.
- The stream can be closed cleanly.

### 5. Add screenshot saving and frame drawing helpers
Goal: Learn basic frame manipulation and image output.

Acceptance criteria:
- A frame can be saved as an image.
- Rectangles can be drawn on a frame.
- Text can be drawn on a frame.

### 6. Document image and video fundamentals
Goal: Capture key concepts needed for real-time processing.

Acceptance criteria:
- Notes explain frames, pixels, arrays, and RGB/BGR.
- The README links to the OpenCV and NumPy docs.

## Epic 3: Object Detection Core

### 7. Learn YOLO detection output structure
Goal: Understand how detections are represented.

Acceptance criteria:
- The app can print detected boxes, labels, and confidence scores.
- Coordinates and class IDs are explained in comments or docs.

### 8. Run a pretrained YOLO model on the webcam feed
Goal: Add real-time object detection.

Acceptance criteria:
- A YOLO model loads successfully.
- Webcam frames are processed with inference.
- Detections are displayed in real time.

### 9. Draw bounding boxes and labels on detected objects
Goal: Make detections visible on the video stream.

Acceptance criteria:
- Bounding boxes render correctly.
- Labels and confidence values render correctly.
- Output remains readable during live playback.

### 10. Add object counting and detection filtering
Goal: Make the detector smarter and less noisy.

Acceptance criteria:
- The app can count detections by class.
- The app can filter detections to selected classes.
- Only matching objects trigger annotations or alerts.

### 11. Add cooldown logic for repeated detections
Goal: Prevent repeated actions from firing too often.

Acceptance criteria:
- Repeated detections are rate-limited.
- Cooldown timing is configurable.
- The behavior is predictable during rapid object movement.

## Epic 4: Voice, Logging, and Storage

### 12. Add text-to-speech announcements
Goal: Make the application speak detected labels aloud.

Acceptance criteria:
- pyttsx3 is integrated.
- Detected objects can be spoken aloud.
- Speech spam is prevented with throttling or cooldowns.

### 13. Save detection logs and timestamps
Goal: Record detection activity for later review.

Acceptance criteria:
- Detection events are written to a log.
- Timestamps are included.
- Confidence scores and labels are stored.

### 14. Save screenshots from detection events
Goal: Persist important detections as images.

Acceptance criteria:
- A screenshot can be captured on demand or on detection.
- Saved files go to a predictable location.
- Filenames include useful metadata such as timestamps.

## Epic 5: Code Quality and Architecture

### 15. Refactor the app into modular components
Goal: Separate detection, speech, utilities, and configuration.

Acceptance criteria:
- Detection logic lives outside the main entrypoint.
- Voice logic lives in its own module.
- Configuration values are centralized.
- The codebase is easier to test and maintain.

### 16. Add error handling and debugging support
Goal: Make the project easier to diagnose and less brittle.

Acceptance criteria:
- Common runtime errors are handled gracefully.
- Tracebacks are easier to understand.
- Debugging instructions are documented for VS Code.

### 17. Add structured logging
Goal: Replace ad hoc prints with maintainable logging.

Acceptance criteria:
- The logging module is used for key events.
- Log levels are meaningful.
- Logs support debugging without cluttering the UI.

### 18. Optimize performance for smoother real-time inference
Goal: Improve FPS and responsiveness.

Acceptance criteria:
- FPS is measured and displayed.
- Frame resizing is implemented where appropriate.
- Inference performance is reviewed and improved.

## Epic 6: User Interface and Distribution

### 19. Build an optional desktop GUI
Goal: Provide a more polished user experience.

Acceptance criteria:
- The app has start and stop controls.
- Detection history is visible.
- Confidence information is displayed.

### 20. Add portfolio-ready project documentation
Goal: Make the project easy to understand and present.

Acceptance criteria:
- README includes installation and usage steps.
- Screenshots are included.
- A demo section is added.
- The setup instructions are current.

### 21. Prepare Git and GitHub workflow tasks
Goal: Track the project professionally with version control.

Acceptance criteria:
- The repository is initialized and committed.
- Branching workflow is documented.
- Basic GitHub publishing steps are complete.

## Epic 7: Advanced Expansions

### 22. Train a custom detection model
Goal: Extend the project beyond pretrained detection.

Acceptance criteria:
- A training workflow is documented.
- Dataset requirements are identified.
- The custom model can be evaluated in the app.

### 23. Add hand tracking as an optional module
Goal: Expand the app with a second vision capability.

Acceptance criteria:
- MediaPipe is integrated or documented.
- Hand landmarks can be detected.
- The module is isolated from core detection logic.

### 24. Add a web or API interface
Goal: Expose detections through a backend service.

Acceptance criteria:
- A minimal Flask or FastAPI service is defined.
- Detection output can be queried or streamed.
- The backend is separated from the desktop workflow.

### 25. Add an external bot integration
Goal: Send detection events to another platform.

Acceptance criteria:
- Detection events can be forwarded externally.
- The bot integration is optional and configurable.
- The feature does not block the core app.

## Suggested Order

1. 1-3: Environment and Python basics
2. 4-6: Webcam and image processing fundamentals
3. 7-11: YOLO detection and control logic
4. 12-14: Voice, logging, and screenshots
5. 15-18: Refactor, debug, and optimize
6. 19-21: GUI, documentation, and GitHub polish
7. 22-25: Optional advanced expansions