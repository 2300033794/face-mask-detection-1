"""
inference_image.py — Run face-mask detection on a single image.

Usage:
    python src/inference_image.py --source path/to/image.jpg
    python src/inference_image.py --source path/to/image.jpg --weights runs/detect/face_mask/weights/best.pt

The script:
1. Loads the trained YOLOv8 model.
2. Runs inference (forward pass through the CNN).
3. Applies Non-Maximum Suppression (NMS) to remove duplicate boxes.
4. Draws colour-coded bounding boxes + class labels on the image.
5. Saves the annotated image to the outputs/ directory.
6. Prints a summary of detections to the console.

─── How confidence scores work ──────────────────────────────────────
The model head outputs, for each candidate anchor:
  • objectness score    : probability that any object is present
  • class probabilities : softmax distribution over all classes

The final confidence = objectness × class_probability.
Only boxes with confidence ≥ --conf threshold are kept.
Overlapping boxes are further filtered by NMS using --iou threshold.

Expected sample output (console)
---------------------------------
    [inference_image] Image  : samples/test.jpg
    [inference_image] Device : cpu
    [inference_image] 3 detection(s) found:
      #1  with_mask     conf=0.94  box=[112, 45, 210, 180]
      #2  with_mask     conf=0.87  box=[305, 60, 398, 192]
      #3  without_mask  conf=0.81  box=[498, 38, 590, 175]
    [inference_image] Output saved → outputs/test_result.jpg
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils import draw_boxes, get_device


DEFAULT_WEIGHTS = str(
    Path(__file__).parent.parent / "runs" / "detect" / "face_mask" / "weights" / "best.pt"
)
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect face masks in a single image."
    )
    parser.add_argument(
        "--source", required=True, help="Path to input image."
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
    return parser.parse_args()


def run_inference(
    source: str,
    weights: str = DEFAULT_WEIGHTS,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> str:
    """
    Perform mask-detection inference on one image.

    Returns the path to the saved output image.
    """
    device = get_device()
    model = YOLO(weights)
    class_names = model.names  # {0: 'with_mask', 1: 'without_mask'}

    print(f"\n[inference_image] Image  : {source}")
    print(f"[inference_image] Device : {device}")

    # ── Run inference ────────────────────────────────────────────────
    # YOLOv8 internally:
    #   1. Resizes the image to imgsz × imgsz
    #   2. Passes it through the CSPDarknet backbone to extract features
    #   3. PANet neck fuses multi-scale feature maps
    #   4. Detection head decodes anchor offsets and class scores
    #   5. NMS removes redundant overlapping boxes
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes

    # ── Console summary ───────────────────────────────────────────────
    num_detections = len(boxes) if boxes is not None else 0
    print(f"[inference_image] {num_detections} detection(s) found:")
    if boxes is not None:
        for i, box in enumerate(boxes, start=1):
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            coords = [int(v) for v in box.xyxy[0].tolist()]
            print(
                f"  #{i:<3} {class_names[cls_id]:<14} "
                f"conf={conf_val:.2f}  box={coords}"
            )

    # ── Draw annotations ──────────────────────────────────────────────
    img = cv2.imread(source)
    annotated = draw_boxes(img, boxes, list(class_names.values()))

    # ── Save output ───────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(source).stem
    out_path = str(OUTPUT_DIR / f"{stem}_result.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"[inference_image] Output saved → {out_path}\n")

    return out_path


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        source=args.source,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
