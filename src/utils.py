"""
utils.py — Shared helper functions used across the project.

Includes:
    • get_device()     – select CUDA or CPU automatically
    • draw_boxes()     – draw bounding boxes and labels on a frame
    • get_color()      – deterministic per-class colour
"""

import cv2
import numpy as np
import torch


def get_device() -> str:
    """Return 'cuda:0' if a GPU is available, otherwise 'cpu'."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# Colour map: one BGR colour per class index.
# Add more entries here if you expand the number of classes.
_CLASS_COLORS = {
    0: (0, 200, 0),    # with_mask    → green
    1: (0, 0, 220),    # without_mask → red
}
_DEFAULT_COLOR = (200, 200, 0)  # fallback colour for unknown classes


def get_color(class_id: int) -> tuple:
    """Return the BGR colour tuple for *class_id*."""
    return _CLASS_COLORS.get(class_id, _DEFAULT_COLOR)


def draw_boxes(
    frame: np.ndarray,
    boxes,
    class_names: list[str],
) -> np.ndarray:
    """
    Draw bounding boxes and labels on *frame* in-place and return it.

    Parameters
    ----------
    frame       : BGR image as a NumPy array (H × W × 3).
    boxes       : Ultralytics result boxes object (result.boxes).
    class_names : List of class name strings, indexed by class id.

    Returns
    -------
    Annotated BGR frame.
    """
    if boxes is None or len(boxes) == 0:
        return frame

    for box in boxes:
        # Bounding-box pixel coordinates [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{class_names[cls_id]} {conf:.2f}"
        color = get_color(cls_id)

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

        # Background pill for label text
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            color,
            thickness=cv2.FILLED,
        )

        # Label text
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return frame
