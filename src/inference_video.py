"""
inference_video.py — Run face-mask detection on a video file.

Usage:
    python src/inference_video.py --source path/to/video.mp4
    python src/inference_video.py --source path/to/video.mp4 --output outputs/result.mp4

The script:
1. Opens the input video with OpenCV VideoCapture.
2. Processes each frame with the YOLOv8 model.
3. Draws bounding boxes and labels on each frame.
4. Writes the annotated frames to an output video file.
5. Prints a progress update every 50 frames.

─── Output codec ───────────────────────────────────────────────────
The default output codec is mp4v (MPEG-4 Part 2), which is widely
supported.  For H.264 encoding use 'avc1' (requires FFmpeg build
of OpenCV) or post-process with ffmpeg.
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
        description="Run face-mask detection on a video file."
    )
    parser.add_argument("--source", required=True, help="Path to input video.")
    parser.add_argument(
        "--weights", default=DEFAULT_WEIGHTS, help="Path to model weights (.pt)."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save annotated output video (default: outputs/<source>_result.mp4).",
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


def process_video(
    source: str,
    weights: str = DEFAULT_WEIGHTS,
    output: str = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
) -> str:
    """
    Process every frame of *source* video and save annotated output.

    Returns the path to the saved output video.
    """
    device = get_device()
    model = YOLO(weights)
    class_names = list(model.names.values())

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source}")

    # Read video metadata
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine output path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output is None:
        stem = Path(source).stem
        output = str(OUTPUT_DIR / f"{stem}_result.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    print(f"\n[video] Source     : {source}")
    print(f"[video] Resolution : {width}×{height}  FPS={fps:.1f}  Frames={total_frames}")
    print(f"[video] Output     : {output}")
    print(f"[video] Device     : {device}\n")

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
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

            # ── Annotate & write ─────────────────────────────────────────
            annotated = draw_boxes(frame, results[0].boxes, class_names)
            writer.write(annotated)

            frame_idx += 1
            if frame_idx % 50 == 0:
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"[video] Processed {frame_idx}/{total_frames} frames ({pct:.1f}%)")

    finally:
        cap.release()
        writer.release()

    print(f"\n[video] Done! Annotated video saved → {output}\n")
    return output


if __name__ == "__main__":
    args = parse_args()
    process_video(
        source=args.source,
        weights=args.weights,
        output=args.output,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
    )
