# 🎭 Face Mask Detection using YOLOv8

A production-ready, modular face-mask detection system built with **Ultralytics YOLOv8** and **OpenCV**. The system detects multiple faces in a single image or video and classifies each as **with_mask** or **without_mask** in real time.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [How the System Works Internally](#how-the-system-works-internally)
3. [Folder Structure](#folder-structure)
4. [Dataset Preparation](#dataset-preparation)
5. [Environment Setup](#environment-setup)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Inference](#inference)
   - [Single Image](#single-image-inference)
   - [Webcam (Real-time)](#webcam-real-time-inference)
   - [Video File](#video-file-inference)
9. [ONNX Export & Deployment](#onnx-export--deployment)
10. [FastAPI Backend](#fastapi-backend)
11. [Performance Optimization](#performance-optimization)
12. [Sample Output](#sample-output)
13. [Future Improvements](#future-improvements)

---

## Project Overview

| Property | Value |
|---|---|
| Task | Object Detection |
| Classes | `with_mask`, `without_mask` |
| Model | YOLOv8n / YOLOv8s (Ultralytics) |
| Framework | PyTorch + OpenCV |
| Deployment | ONNX, FastAPI |

---

## How the System Works Internally

### 1. CNN Feature Extraction

YOLOv8 uses a **CSPDarknet** backbone — a deep convolutional neural network (CNN). Each convolutional layer learns spatial filters that detect increasingly abstract features:

```
Early layers  → edges, corners, colour gradients
Middle layers → eyes, noses, mouth shapes
Deep layers   → full faces, mask textures
```

### 2. Multi-Scale Detection (PANet Neck)

The **PANet neck** fuses feature maps from multiple backbone stages (shallow + deep). This lets the model detect both small and large faces in the same image without rescaling.

### 3. Bounding-Box Prediction

The **detection head** divides the input into a grid and predicts, for each of ~8 400 anchor points:
- **(cx, cy)** — centre of the bounding box
- **(w, h)** — width and height
- **class probabilities** — `with_mask` vs `without_mask`

### 4. Confidence Score

```
confidence = objectness_score × class_probability
```

Only predictions above the `--conf` threshold are kept. **Non-Maximum Suppression (NMS)** then removes duplicate overlapping boxes using the `--iou` threshold.

### 5. Why Object Detection > Classification Here

| Aspect | Classification | Object Detection |
|---|---|---|
| Multiple faces | ❌ One label per image | ✅ One box per face |
| Face location | ❌ Unknown | ✅ Exact pixel coords |
| Crowd scenes | ❌ Fails | ✅ Handles many people |
| Real-time video | Slow | Fast (single forward pass) |

---

## Folder Structure

```
face-mask-detection-1/
├── README.md                    ← This file
├── requirements.txt             ← Python dependencies
│
├── data/
│   ├── data.yaml                ← YOLO dataset config
│   ├── README.md                ← Dataset format guide
│   ├── images/
│   │   ├── train/               ← Training images (.jpg/.png)
│   │   └── val/                 ← Validation images
│   ├── labels/
│   │   ├── train/               ← YOLO annotation .txt files
│   │   └── val/
│   └── sample_labels/
│       └── example_image.txt    ← Example annotation
│
├── src/
│   ├── __init__.py
│   ├── utils.py                 ← Shared helpers (draw_boxes, get_device)
│   ├── train.py                 ← Model training script
│   ├── evaluate.py              ← Validation & metrics
│   ├── inference_image.py       ← Single-image inference
│   ├── inference_webcam.py      ← Real-time webcam detection
│   ├── inference_video.py       ← Video file processing
│   └── export_onnx.py           ← ONNX export + inference
│
├── api/
│   ├── __init__.py
│   └── app.py                   ← FastAPI REST backend
│
├── scripts/
│   └── check_gpu.py             ← GPU availability check
│
└── outputs/                     ← Saved annotated images/videos
```

---

## Dataset Preparation

### YOLO Annotation Format

Each image requires a `.txt` annotation file with the same base name, placed in the `labels/` directory:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are **normalised to [0, 1]** relative to the image dimensions.

**Example** (`data/sample_labels/example_image.txt`):
```
0 0.2500 0.3750 0.1800 0.2600
1 0.7200 0.4100 0.2000 0.2800
```

- Line 1: `with_mask` (class 0), centred at 25% × 37.5%, size 18% × 26% of image
- Line 2: `without_mask` (class 1), centred at 72% × 41%, size 20% × 28% of image

### data.yaml

```yaml
path: .
train: images/train
val:   images/val
nc: 2
names:
  0: with_mask
  1: without_mask
```

### Train / Validation Split

Recommended **80 / 20** split:
- Place 80 % of images + labels → `images/train` & `labels/train`
- Place 20 % of images + labels → `images/val` & `labels/val`

### Recommended Public Datasets

- [Kaggle Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) (~850 images, XML annotations → convert with `xml_to_yolo.py`)
- [Roboflow Face Mask Dataset](https://universe.roboflow.com/joseph-nelson/face-mask) (pre-converted to YOLO format)

---

## Environment Setup

### 1. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify GPU

```bash
python scripts/check_gpu.py
```

Expected output on a GPU machine:
```
PyTorch version : 2.x.x+cu118
CUDA available  : True
GPU count       : 1
GPU name        : NVIDIA GeForce RTX 3080
Device selected : cuda:0
```

---

## Training

```bash
# Default: YOLOv8 nano, 50 epochs, batch=16, imgsz=640
python src/train.py

# Custom settings
python src/train.py --model yolov8s.pt --epochs 100 --batch 32 --imgsz 640
```

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `--model` | `yolov8n.pt` | Base checkpoint (downloads automatically) |
| `--epochs` | `50` | Training epochs |
| `--batch` | `16` | Images per gradient step |
| `--imgsz` | `640` | Input resolution (square) |

### Loss Components

| Loss | Description |
|---|---|
| `box_loss` | IoU regression — penalises inaccurate bounding box coordinates |
| `cls_loss` | Cross-entropy — penalises wrong class predictions |
| `dfl_loss` | Distribution Focal Loss — improves edge localisation |

### Output

Best weights saved to:
```
runs/detect/face_mask/weights/best.pt
```

---

## Evaluation

```bash
python src/evaluate.py --weights runs/detect/face_mask/weights/best.pt
```

Expected output:
```
── Validation Results ─────────────────────────────────────
  mAP@0.50        : 0.9370
  mAP@0.50:0.95   : 0.7120
  Precision (mean): 0.9210
  Recall    (mean): 0.9080
───────────────────────────────────────────────────────────
```

---

## Inference

### Single Image Inference

```bash
python src/inference_image.py --source path/to/image.jpg
```

Expected console output:
```
[inference_image] Image  : path/to/image.jpg
[inference_image] Device : cpu
[inference_image] 3 detection(s) found:
  #1   with_mask      conf=0.94  box=[112, 45, 210, 180]
  #2   with_mask      conf=0.87  box=[305, 60, 398, 192]
  #3   without_mask   conf=0.81  box=[498, 38, 590, 175]
[inference_image] Output saved → outputs/image_result.jpg
```

### Webcam (Real-time) Inference

```bash
python src/inference_webcam.py
# Press 'q' to quit
```

FPS is displayed in the top-left corner of the live window.

### Video File Inference

```bash
python src/inference_video.py --source path/to/video.mp4
# Saves to outputs/video_result.mp4
```

---

## ONNX Export & Deployment

### Export to ONNX

```bash
python src/export_onnx.py --weights runs/detect/face_mask/weights/best.pt
# Saved as runs/detect/face_mask/weights/best.onnx
```

### Run ONNX Inference

```bash
python src/export_onnx.py \
  --onnx runs/detect/face_mask/weights/best.onnx \
  --infer --source path/to/image.jpg
```

---

## FastAPI Backend

### Start the Server

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs: `http://localhost:8000/docs`

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/info` | Model metadata |
| `POST` | `/predict` | Upload image → JSON detections |
| `POST` | `/predict/image` | Upload image → annotated JPEG |

### Example Request

```bash
# JSON response
curl -X POST "http://localhost:8000/predict" \
     -F "file=@/path/to/image.jpg" | python -m json.tool

# Annotated image
curl -X POST "http://localhost:8000/predict/image" \
     -F "file=@/path/to/image.jpg" --output result.jpg
```

### Example JSON Response

```json
{
  "num_detections": 2,
  "detections": [
    {
      "class_id": 0,
      "class_name": "with_mask",
      "confidence": 0.94,
      "box": {"x1": 112, "y1": 45, "x2": 210, "y2": 180}
    },
    {
      "class_id": 1,
      "class_name": "without_mask",
      "confidence": 0.81,
      "box": {"x1": 498, "y1": 38, "x2": 590, "y2": 175}
    }
  ]
}
```

---

## Performance Optimization

### Model Size vs Speed Trade-off

| Model | Params | mAP@0.5 (COCO) | CPU FPS (est.) | GPU FPS (est.) |
|---|---|---|---|---|
| YOLOv8n | 3.2 M | 37.3 | ~25 | ~200 |
| YOLOv8s | 11.2 M | 44.9 | ~12 | ~120 |
| YOLOv8m | 25.9 M | 50.2 | ~5  | ~80  |

**Recommendation**: Use `yolov8n.pt` for real-time CPU inference. Switch to `yolov8s.pt` for higher accuracy when a GPU is available.

### Additional Tips

1. **Lower `--imgsz`** (e.g., `416`) — reduces computation at the cost of small-face accuracy.
2. **Increase `--conf`** threshold — fewer false positives, slightly lower recall.
3. **Use ONNX + onnxruntime** — typically 1.5–2× faster than PyTorch on CPU.
4. **Frame skipping** in `inference_webcam.py` — process every other frame at high camera FPS.
5. **Half precision** (`model.predict(half=True)`) — 2× speedup on supported GPUs.

---

## Sample Output

```
Bounding boxes:
  ┌─────────────────────────────────────────────────────┐
  │  [with_mask 0.94]  [with_mask 0.87]                 │
  │  ┌─────────┐       ┌─────────┐                      │
  │  │  😷 ✓   │       │  😷 ✓   │  [without_mask 0.81] │
  │  └─────────┘       └─────────┘  ┌─────────┐        │
  │                                  │  😶 ✗   │        │
  │                                  └─────────┘        │
  └─────────────────────────────────────────────────────┘
  Green box = with_mask  |  Red box = without_mask
```

---

## Future Improvements

- [ ] **Mask-type classification** — surgical mask vs N95 vs cloth mask
- [ ] **Mask colour detection** — add colour classification head
- [ ] **Alert system** — send notification when `without_mask` is detected
- [ ] **Multi-camera support** — handle multiple RTSP streams simultaneously
- [ ] **Docker containerisation** — `Dockerfile` for one-command deployment
- [ ] **Distance estimation** — approximate face distance using bounding-box size
- [ ] **Web dashboard** — React/Vue frontend consuming the FastAPI backend
- [ ] **Model quantisation** — INT8 for embedded/edge deployment (Raspberry Pi, Jetson Nano)
- [ ] **Incorrect mask detection** — third class for improperly worn masks