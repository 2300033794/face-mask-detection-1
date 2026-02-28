"""
train.py — Train a YOLOv8 model for face-mask detection.

Usage:
    python src/train.py [--epochs 50] [--batch 16] [--imgsz 640] [--model yolov8n.pt]

The script:
1. Loads a pretrained YOLOv8 nano (or other variant) checkpoint.
2. Fine-tunes it on the face-mask dataset described in data/data.yaml.
3. Saves the best weights to runs/detect/face_mask/weights/best.pt.

─── Training Parameters ───────────────────────────────────────────
• epochs   : Number of full passes through the training data.
             More epochs → better accuracy (up to a point).
• batch    : Images per gradient-update step.
             Larger batches → more stable gradients but more VRAM.
• imgsz    : Input resolution fed to the network (square crop).
             640 × 640 is the YOLOv8 default; use 416 to save memory.

─── Loss Components ────────────────────────────────────────────────
YOLOv8 minimises a composite loss on each training step:

  box_loss  — IoU-based regression loss. Penalises inaccurate
              predicted bounding-box coordinates vs ground-truth.

  cls_loss  — Cross-entropy classification loss. Penalises wrong
              class predictions (with_mask vs without_mask).

  dfl_loss  — Distribution Focal Loss. Improves localisation by
              modelling box-edge positions as distributions rather
              than single values, giving sharper boundary estimates.

─── Evaluation Metric (mAP) ────────────────────────────────────────
mAP (mean Average Precision) is the standard COCO detection metric:

  • For each class, precision–recall curve is plotted over a range
    of confidence thresholds.
  • The area under that curve is the Average Precision (AP).
  • mAP@0.5   : average AP at IoU threshold 0.50.
  • mAP@0.5:0.95 : average AP over IoU thresholds 0.50–0.95 (step 0.05).

Higher mAP → better detection performance.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


# ── Default hyperparameters ──────────────────────────────────────────────────

DATA_YAML = str(Path(__file__).parent.parent / "data" / "data.yaml")
PROJECT_DIR = str(Path(__file__).parent.parent / "runs" / "detect")
RUN_NAME = "face_mask"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for face-mask detection."
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 checkpoint to start from (e.g. yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="Batch size per training step."
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Input image size (square)."
    )
    parser.add_argument(
        "--data",
        default=DATA_YAML,
        help="Path to data.yaml dataset config file.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override: 'cpu', '0', '0,1', etc. Auto-detected if omitted.",
    )
    return parser.parse_args()


def train(args) -> None:
    """
    Fine-tune a YOLOv8 model on the face-mask dataset.

    How it works internally
    -----------------------
    1. YOLOv8's backbone (CSPDarknet) acts as a convolutional feature extractor.
       Each convolutional layer learns spatial filters that detect progressively
       higher-level features: edges → shapes → face parts → full faces.

    2. The neck (PANet) merges features from multiple backbone scales so the
       model can detect both small and large faces.

    3. The head predicts, for each of 8400 anchor points across three scales,
       a class probability distribution and a bounding-box offset.

    4. During training, predicted boxes are matched to ground-truth boxes via
       the TaskAligned assigner; unmatched predictions are treated as background.

    5. Gradients flow back through box_loss + cls_loss + dfl_loss, and the
       Adam/SGD optimiser updates all weights.
    """
    # Load base checkpoint (downloads from Ultralytics hub on first run)
    model = YOLO(args.model)

    print(
        f"\n[train] Starting training:"
        f"\n  model  = {args.model}"
        f"\n  data   = {args.data}"
        f"\n  epochs = {args.epochs}"
        f"\n  batch  = {args.batch}"
        f"\n  imgsz  = {args.imgsz}"
        f"\n  device = {args.device or 'auto'}\n"
    )

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=PROJECT_DIR,
        name=RUN_NAME,
        device=args.device,   # None → auto-select GPU/CPU
        exist_ok=True,        # overwrite previous run if re-running
        verbose=True,
    )

    # Locate the saved best weights
    best_weights = (
        Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    )
    if best_weights.exists():
        print(f"\n[train] Best weights saved to: {best_weights}")
    else:
        print("\n[train] Warning: could not locate best.pt — check run directory.")

    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
