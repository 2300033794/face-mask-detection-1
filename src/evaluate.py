"""
evaluate.py — Validate a trained YOLOv8 face-mask model.

Usage:
    python src/evaluate.py --weights runs/detect/face_mask/weights/best.pt

The script runs the model over the validation split defined in data/data.yaml
and prints precision, recall, and mAP metrics to the console.

─── Metrics Explained ──────────────────────────────────────────────
• Precision   : Of all bounding boxes predicted as class X, what
                fraction were actually class X?
                High precision → few false positives.

• Recall      : Of all ground-truth instances of class X, what
                fraction did the model detect?
                High recall → few missed detections.

• mAP@0.5     : Mean Average Precision at IoU ≥ 0.50.
                A predicted box counts as a true positive only when
                its overlap with the ground-truth box exceeds 50 %.

• mAP@0.5:0.95: Stricter metric averaged over IoU thresholds
                0.50, 0.55, …, 0.95.  Better reflects real-world
                localisation quality.

Expected sample output
----------------------
    Class          Images  Instances      P      R  mAP50  mAP50-95
    all               200        340  0.921  0.908  0.937     0.712
    with_mask         200        210  0.945  0.931  0.958     0.741
    without_mask      200        130  0.896  0.885  0.916     0.683
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


DATA_YAML = str(Path(__file__).parent.parent / "data" / "data.yaml")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained YOLOv8 face-mask model."
    )
    parser.add_argument(
        "--weights",
        default="runs/detect/face_mask/weights/best.pt",
        help="Path to trained model weights (.pt file).",
    )
    parser.add_argument(
        "--data",
        default=DATA_YAML,
        help="Path to data.yaml dataset config.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Inference image size."
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold."
    )
    parser.add_argument(
        "--iou", type=float, default=0.5, help="IoU threshold for NMS."
    )
    return parser.parse_args()


def evaluate(args) -> None:
    """Run validation and print per-class metrics."""
    model = YOLO(args.weights)

    print(
        f"\n[evaluate] Running validation:"
        f"\n  weights = {args.weights}"
        f"\n  data    = {args.data}"
        f"\n  imgsz   = {args.imgsz}"
        f"\n  conf    = {args.conf}"
        f"\n  iou     = {args.iou}\n"
    )

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=True,
    )

    # Pretty-print key summary numbers
    print("\n── Validation Results ─────────────────────────────────────")
    print(f"  mAP@0.50        : {metrics.box.map50:.4f}")
    print(f"  mAP@0.50:0.95   : {metrics.box.map:.4f}")
    print(f"  Precision (mean): {metrics.box.mp:.4f}")
    print(f"  Recall    (mean): {metrics.box.mr:.4f}")
    print("───────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
