"""
export_onnx.py — Export a trained YOLOv8 model to ONNX format.

Usage:
    python src/export_onnx.py --weights runs/detect/face_mask/weights/best.pt

Then run ONNX inference:
    python src/export_onnx.py --infer --source path/to/image.jpg --onnx best.onnx

─── Why ONNX? ──────────────────────────────────────────────────────
ONNX (Open Neural Network Exchange) is a hardware-agnostic model
interchange format.  Benefits:
  • Deploy on servers/edge devices without a full PyTorch installation.
  • Use onnxruntime for CPU-optimised inference (often faster than
    plain PyTorch on CPU).
  • Interoperable with TensorRT, CoreML, OpenVINO, etc.

─── How the ONNX inference pipeline works ──────────────────────────
1. onnxruntime loads the .onnx graph.
2. Input image is pre-processed: resize → normalise to [0, 1] → NCHW.
3. The ONNX session runs a forward pass and returns raw output tensors
   with shape [1, 6, 8400] (cx, cy, w, h, with_mask_conf, without_mask_conf).
4. Boxes with max-class confidence ≥ threshold are kept.
5. NMS filters overlapping boxes.
6. Results are drawn and saved.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from utils import get_device


DEFAULT_WEIGHTS = str(
    Path(__file__).parent.parent / "runs" / "detect" / "face_mask" / "weights" / "best.pt"
)
CLASS_NAMES = ["with_mask", "without_mask"]


# ── Colour map (BGR) ─────────────────────────────────────────────────────────
COLORS = {0: (0, 200, 0), 1: (0, 0, 220)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 to ONNX and/or run ONNX inference."
    )
    parser.add_argument(
        "--weights", default=DEFAULT_WEIGHTS, help="PyTorch .pt weights to export."
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="Export/inference image size."
    )
    parser.add_argument(
        "--infer",
        action="store_true",
        help="If set, also run ONNX inference on --source.",
    )
    parser.add_argument("--source", default=None, help="Image path for ONNX inference.")
    parser.add_argument(
        "--onnx", default=None, help="Path to existing .onnx file (skips export)."
    )
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold."
    )
    return parser.parse_args()


# ── Export ───────────────────────────────────────────────────────────────────

def export_to_onnx(weights: str, imgsz: int = 640) -> str:
    """
    Convert a PyTorch YOLOv8 model to ONNX format.

    Returns the path to the exported .onnx file.
    """
    model = YOLO(weights)
    print(f"\n[export] Exporting {weights} → ONNX (imgsz={imgsz}) …")
    onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)
    print(f"[export] ONNX model saved → {onnx_path}\n")
    return str(onnx_path)


# ── ONNX Inference ───────────────────────────────────────────────────────────

def _preprocess(image_bgr: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, float]:
    """
    Resize and normalise an image for ONNX inference.

    Returns (blob, x_scale, y_scale) where scales map model coords back
    to original image coords.
    """
    orig_h, orig_w = image_bgr.shape[:2]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (imgsz, imgsz))
    blob = img_resized.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))           # HWC → CHW
    blob = np.expand_dims(blob, axis=0)             # add batch dim
    x_scale = orig_w / imgsz
    y_scale = orig_h / imgsz
    return blob, x_scale, y_scale


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45):
    """Simple greedy NMS. Returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]
    return keep


def run_onnx_inference(
    onnx_path: str,
    source: str,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
) -> str:
    """
    Run inference using an ONNX model file.

    Returns path to the saved annotated output image.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("Install onnxruntime: pip install onnxruntime")

    session = ort.InferenceSession(
        onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    img = cv2.imread(source)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {source}")

    blob, x_scale, y_scale = _preprocess(img, imgsz)

    # ── Forward pass ─────────────────────────────────────────────────────────
    outputs = session.run(None, {input_name: blob})
    # output shape: [1, 6, 8400]  (cx, cy, w, h, cls0_conf, cls1_conf)
    preds = outputs[0][0].T          # [8400, 6]

    boxes_list, scores_list, class_ids_list = [], [], []
    for pred in preds:
        cx, cy, w, h = pred[:4]
        class_scores = pred[4:]
        cls_id = int(np.argmax(class_scores))
        conf = float(class_scores[cls_id])
        if conf < conf_threshold:
            continue
        x1 = (cx - w / 2) * x_scale
        y1 = (cy - h / 2) * y_scale
        x2 = (cx + w / 2) * x_scale
        y2 = (cy + h / 2) * y_scale
        boxes_list.append([x1, y1, x2, y2])
        scores_list.append(conf)
        class_ids_list.append(cls_id)

    if boxes_list:
        boxes_arr = np.array(boxes_list)
        scores_arr = np.array(scores_list)
        kept = _nms(boxes_arr, scores_arr)
        print(f"\n[onnx_infer] {len(kept)} detection(s) found:")
        for i in kept:
            x1, y1, x2, y2 = map(int, boxes_arr[i])
            cls_id = class_ids_list[i]
            conf = scores_arr[i]
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
            color = COLORS.get(cls_id, (200, 200, 0))
            print(f"  {CLASS_NAMES[cls_id]:<14} conf={conf:.2f}  box=[{x1},{y1},{x2},{y2}]")
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    else:
        print("[onnx_infer] No detections above threshold.")

    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / (Path(source).stem + "_onnx_result.jpg"))
    cv2.imwrite(out_path, img)
    print(f"[onnx_infer] Output saved → {out_path}\n")
    return out_path


if __name__ == "__main__":
    args = parse_args()

    onnx_path = args.onnx
    if onnx_path is None:
        onnx_path = export_to_onnx(args.weights, args.imgsz)

    if args.infer:
        if args.source is None:
            raise ValueError("Provide --source <image_path> when using --infer.")
        run_onnx_inference(onnx_path, args.source, args.conf, args.imgsz)
