"""
app.py — FastAPI backend for face-mask detection inference.

Usage:
    # From the project root:
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /            → health check
    GET  /info        → model metadata
    POST /predict     → upload an image, receive detection results (JSON)
    POST /predict/image → upload an image, receive annotated image (JPEG)

─── How a request flows through the system ─────────────────────────
1. Client sends a multipart POST request with an image file.
2. FastAPI reads the file bytes into memory.
3. The bytes are decoded into a BGR NumPy array via OpenCV.
4. YOLOv8 performs inference (same pipeline as inference_image.py).
5. Results are serialised to JSON (class, confidence, bounding box)
   and returned to the client.
6. Optionally, the annotated image is JPEG-encoded and returned as
   an image/jpeg response for direct browser display.

─── Sample curl request ────────────────────────────────────────────
    curl -X POST "http://localhost:8000/predict" \
         -F "file=@/path/to/image.jpg" | python -m json.tool

─── Sample JSON response ───────────────────────────────────────────
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
"""

import io
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from ultralytics import YOLO

# Allow imports from src/ when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils import draw_boxes, get_device  # noqa: E402  (after sys.path insert)


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = str(
    Path(__file__).parent.parent / "runs" / "detect" / "face_mask" / "weights" / "best.pt"
)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640


# ── Application setup ────────────────────────────────────────────────────────

app = FastAPI(
    title="Face Mask Detection API",
    description=(
        "Real-time face-mask detection powered by YOLOv8. "
        "Upload an image to receive bounding-box detections."
    ),
    version="1.0.0",
)

# Load the model once at startup (not per request) for performance.
_device = get_device()
try:
    _model = YOLO(DEFAULT_WEIGHTS)
    _class_names = list(_model.names.values())
    _model_loaded = True
except Exception as _load_err:
    _model_loaded = False
    _load_err_msg = str(_load_err)


# ── Helper ───────────────────────────────────────────────────────────────────

def _bytes_to_bgr(file_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes to a BGR NumPy array."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes — unsupported format?")
    return img


def _run_inference(img: np.ndarray) -> list[dict[str, Any]]:
    """Run YOLOv8 inference and return a list of detection dicts."""
    results = _model.predict(
        source=img,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        device=_device,
        verbose=False,
    )
    detections = []
    boxes = results[0].boxes
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = round(float(box.conf[0]), 4)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": _class_names[cls_id],
                    "confidence": conf,
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )
    return detections


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    """Return API status."""
    return {"status": "ok", "model_loaded": _model_loaded}


@app.get("/info", summary="Model information")
def model_info():
    """Return model metadata."""
    if not _model_loaded:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {_load_err_msg}")
    return {
        "classes": _class_names,
        "device": _device,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "img_size": IMG_SIZE,
    }


@app.post("/predict", summary="Detect face masks — returns JSON")
async def predict(file: UploadFile = File(..., description="Image file (JPEG/PNG)")):
    """
    Upload an image and receive detection results as JSON.

    Returns a JSON object with:
    - num_detections: total number of detected faces
    - detections: list of {class_id, class_name, confidence, box}
    """
    if not _model_loaded:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {_load_err_msg}")

    contents = await file.read()
    try:
        img = _bytes_to_bgr(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    detections = _run_inference(img)
    return {"num_detections": len(detections), "detections": detections}


@app.post(
    "/predict/image",
    response_class=Response,
    summary="Detect face masks — returns annotated JPEG",
)
async def predict_image(file: UploadFile = File(..., description="Image file (JPEG/PNG)")):
    """
    Upload an image and receive it back annotated with bounding boxes.

    The response Content-Type is image/jpeg.
    """
    if not _model_loaded:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {_load_err_msg}")

    contents = await file.read()
    try:
        img = _bytes_to_bgr(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results = _model.predict(
        source=img,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        device=_device,
        verbose=False,
    )
    annotated = draw_boxes(img, results[0].boxes, _class_names)

    _, buffer = cv2.imencode(".jpg", annotated)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")
