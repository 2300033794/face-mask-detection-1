"""
inference_webcam.py — Real-time face-mask detection from a webcam.

Usage:
    python src/inference_webcam.py
    python src/inference_webcam.py --weights runs/detect/face_mask/weights/best.pt

Controls:
    Press 'q' to quit the live window.

─── How it works ───────────────────────────────────────────────────
1. OpenCV captures frames from the default camera (device index 0).
2. Each frame is passed through the YOLOv8 model for inference.
3. Detected bounding boxes are drawn on the frame.
4. FPS is computed as 1 / (time for one inference iteration) and
   displayed in the top-left corner.
5. The annotated frame is shown in a live OpenCV window.

─── Performance tip ────────────────────────────────────────────────
Use the yolov8n (nano) model for maximum FPS on CPU.
With a modern GPU yolov8s can easily exceed 60 FPS.
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import draw_boxes, get_device


DEFAULT_WEIGHTS = str(
    Path(__file__).parent.parent / "runs" / "detect" / "face_mask" / "weights" / "best.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time face-mask detection via webcam."
    )
    parser.add_argument(
        "--weights", default=DEFAULT_WEIGHTS, help="Path to model weights (.pt)."
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold."
    )
    parser.add_argument(
        "--iou", type=float, default=0.45, help="IoU threshold for NMS."
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size."
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera device index."
    )
    return parser.parse_args()


def run_webcam(
    weights: str = DEFAULT_WEIGHTS,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    camera: int = 0,
) -> None:
    """Open webcam and perform continuous face-mask detection."""
    device = get_device()
    model = YOLO(weights)
    class_names = list(model.names.values())

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {camera}.")

    print(f"[webcam] Starting real-time detection. Press 'q' to quit.")
    print(f"[webcam] Device : {device}  |  Model : {weights}\n")

    try:
        while True:
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                print("[webcam] Failed to read frame — camera disconnected?")
                break

            # ── Inference ────────────────────────────────────────────────
            results = model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

            # ── Annotate ─────────────────────────────────────────────────
            annotated = draw_boxes(frame, results[0].boxes, class_names)

            # ── FPS overlay ──────────────────────────────────────────────
            fps = 1.0 / (time.perf_counter() - t_start)
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Face Mask Detection — press 'q' to quit", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[webcam] Quit signal received.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[webcam] Resources released.")


if __name__ == "__main__":
    args = parse_args()
    run_webcam(
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        camera=args.camera,
    )
