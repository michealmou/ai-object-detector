# ai-object-detector

This project uses OpenCV and NumPy to process webcam video in real time.

## Real-Time Processing Fundamentals Guide

### 1) What is a frame?
- A frame is one still image from a video stream.
- A webcam feed is just many frames shown quickly, one after another.
- If your app runs at 30 FPS (frames per second), it processes about 30 frames every second.

In OpenCV, a typical loop looks like this:

```python
ret, frame = cap.read()
```

- `ret` is `True` if a frame was captured successfully.
- `frame` is the image data for that moment.

### 2) What is a pixel?
- A pixel is one tiny dot in an image.
- Every frame is made of many pixels.
- In color images, each pixel stores color values in 3 channels.

Example: one pixel value may look like `[B, G, R]`.

### 3) Why arrays matter (NumPy)
- OpenCV stores images as NumPy arrays.
- A color frame usually has shape `(height, width, 3)`.
- That means:
	- first axis: rows (y)
	- second axis: columns (x)
	- third axis: color channels

Quick checks:

```python
print(type(frame))      # numpy.ndarray
print(frame.shape)      # e.g. (480, 640, 3)
print(frame.dtype)      # usually uint8
```

- `uint8` means each channel value is from 0 to 255.

### 4) RGB vs BGR
- Most imaging libraries and plotting tools expect RGB order.
- OpenCV uses BGR order by default.
- So red in RGB is not red if interpreted as BGR.

Convert BGR to RGB when needed:

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

Convert RGB back to BGR for OpenCV display/drawing:

```python
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
```

### 5) Mental model for real-time detection
1. Capture one frame.
2. Optionally resize/preprocess it.
3. Run model inference on array data.
4. Draw boxes/labels on the frame.
5. Show frame.
6. Repeat quickly.

If this loop is efficient, video appears smooth and detection feels real time.

## Official Docs
- OpenCV Python tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- NumPy user guide: https://numpy.org/doc/stable/user/