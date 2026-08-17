# YOLO Fall Detection

Real-time fall detection using a YOLOv8 model and a webcam feed. When a fall is detected, the app shows a flashing on-screen alert and keeps a running count of falls.

## Features

- Real-time detection from webcam using YOLOv8 (`ultralytics`)
- Live info panel with FPS, detection count, and total falls
- Flashing "FALL DETECTED" alert with a cooldown to avoid duplicate alerts
- Optional recording of the annotated output to a video file
- Reset stats or quit with a keypress

## Requirements

- Python 3.8+
- A webcam
- Dependencies listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Place your trained YOLOv8 fall detection model (`.pt` file) in the project folder and update `MODEL_PATH` in `yolo_based_fall_detection.py` if needed .
2. Run the script:

```bash
python yolo_based_fall_detection.py
```

3. Controls while running:
   - `Q` — quit
   - `R` — reset statistics

The annotated video session is saved to `output_video.mp4` by default.

## Configuration

Key settings are in the `main()` function of `yolo_based_fall_detection.py`:

- `MODEL_PATH` — path to the YOLOv8 `.pt` model
- `CAMERA_INDEX` — webcam index (0 = default)
- `CONFIDENCE_THRESHOLD` — minimum detection confidence
- `COOLDOWN_FRAMES` — frames to wait before registering another fall

## Project Files

- `yolo_based_fall_detection.py` — main detection script
- `fall_detection.pt` — YOLOv8 model weights
- `output_video.mp4` — sample output video
- `requirements.txt` — Python dependencies
