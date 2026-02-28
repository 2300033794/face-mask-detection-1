# Dataset Directory — README
#
# This directory follows the standard YOLO dataset layout:
#
# data/
# ├── data.yaml              ← dataset config (paths, class names)
# ├── images/
# │   ├── train/             ← training images (.jpg / .png)
# │   └── val/               ← validation images (.jpg / .png)
# ├── labels/
# │   ├── train/             ← YOLO annotation .txt files for training images
# │   └── val/               ← YOLO annotation .txt files for validation images
# └── sample_labels/
#     └── example_image.txt  ← illustrative annotation file
#
# ── Annotation Format ────────────────────────────────────────
#
# Each image has a matching .txt file with the same base name.
# Each line in the .txt file describes one bounding box:
#
#   <class_id> <x_center> <y_center> <width> <height>
#
#   • class_id  : integer (0 = with_mask, 1 = without_mask)
#   • x_center  : horizontal centre of the box / image width   → [0, 1]
#   • y_center  : vertical centre of the box / image height    → [0, 1]
#   • width     : box width  / image width                     → [0, 1]
#   • height    : box height / image height                    → [0, 1]
#
# ── Train / Validation Split ──────────────────────────────────
#
# A typical 80 / 20 split is recommended:
#   • 80 % of annotated images → images/train  &  labels/train
#   • 20 % of annotated images → images/val    &  labels/val
#
# Tools like roboflow.com can automate the annotation and split.
#
# ── Public Datasets ───────────────────────────────────────────
#
# • Face Mask Detection dataset (Kaggle):
#   https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
# • MAFA dataset: http://www.escience.cn/people/geshiming/mafa.html
#
# After downloading, convert annotations to YOLO format and place
# images/labels in the directories above.
