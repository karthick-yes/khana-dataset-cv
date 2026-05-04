#!/usr/bin/env python3
"""
Task 2: Thali Food Detection with Correct Khana Labels
Pipeline:
  1. Canny edge detection to find thali compartment boundaries
  2. Filter contours to get food compartments
  3. Classify each compartment crop with Task 1 ConvNeXt model
  4. Fallback to YOLO if Canny finds < 3 compartments

This works well on clean bird's eye thali images (Task 2 dataset).
For natural angle images (Task 3), run BEV transformation first.

Evaluation: Precision-Recall on labels only (not bounding box accuracy)
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image, ImageFile
from torchvision import transforms
import timm
from ultralytics import YOLO

from config import get_paths

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# CONFIG
# ============================================================
paths = get_paths()
DATA_ROOT = paths["DATA_ROOT"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASK2_IMAGE_DIR = os.path.expanduser("~/data/task2_images")
OUTPUT_DIR = os.path.expanduser("~/data/task2_output")
CHECKPOINT_PATH = os.path.expanduser("~/data/checkpoints/best_model.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CLASS NAMES
# ============================================================
def get_class_names(data_root):
    root = Path(data_root)
    if not root.exists():
        print(f"[ERROR] Data root not found: {data_root}")
        sys.exit(1)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    print(f"[INFO] Loaded {len(classes)} class names from dataset")
    return classes


# ============================================================
# TRANSFORMS
# Square-pad preserves full food content vs center crop
# which cuts off food edges on small irregular crops
# ============================================================
def square_pad_and_normalize(crop_pil, target_size=224):
    w, h = crop_pil.size
    size = max(w, h)
    padded = Image.new("RGB", (size, size), (114, 114, 114))
    padded.paste(crop_pil, ((size - w) // 2, (size - h) // 2))
    resized = padded.resize((target_size, target_size), Image.LANCZOS)
    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(resized)
    return tensor


# ============================================================
# LOAD MODELS
# ============================================================
def load_task1_model(checkpoint_path, num_classes=80):
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    print(f"[INFO] Loading Task 1 classifier from {checkpoint_path}")
    model = timm.create_model(
        "convnext_base.fb_in22k_ft_in1k", pretrained=False, num_classes=num_classes
    )
    try:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded. Best val acc: {ckpt.get('best_acc', 'N/A')}%")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    model = model.to(DEVICE)
    model.eval()
    return model


def load_yolo():
    print("[INFO] Loading YOLOv8x (fallback detector)...")
    return YOLO("yolov8x.pt")


# ============================================================
# CLASSIFICATION
# ============================================================
@torch.no_grad()
def classify_crop(model, crop_pil, class_names):
    try:
        tensor = square_pad_and_normalize(crop_pil).unsqueeze(0).to(DEVICE)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_conf = probs.max(dim=1).values.item()
        top5_idx = probs[0].topk(5).indices.cpu().numpy()
        top5 = [(class_names[i], float(probs[0][i])) for i in top5_idx]
        return pred_idx, pred_conf, top5
    except Exception as e:
        print(f"[WARN] Classification failed: {e}")
        return None, 0.0, []


# ============================================================
# CANNY COMPARTMENT DETECTION
# Works well on clean bird's eye thali images.
# The metal tray creates strong edges, compartments are rectangular.
# ============================================================
def find_compartments_canny(
    img_pil, min_area_ratio=0.01, max_area_ratio=0.35, canny_low=30, canny_high=100
):
    """
    Find food compartment bounding boxes using Canny edge detection.

    Returns list of [x1, y1, x2, y2] boxes, one per compartment.
    """
    img_np = np.array(img_pil)
    img_w, img_h = img_pil.size
    img_area = img_w * img_h

    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Blur slightly to reduce noise while preserving strong tray edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Dilate to close gaps in compartment borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        ratio = area / img_area

        # Filter by area ratio — must be compartment-sized
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)

        # Skip very thin/wide boxes (not compartments)
        aspect = w / h if h > 0 else 0
        if aspect < 0.2 or aspect > 5.0:
            continue

        # Skip tiny boxes
        if w < 50 or h < 50:
            continue

        boxes.append([x, y, x + w, y + h])

    # Merge overlapping boxes (a single compartment can have multiple contours)
    boxes = merge_overlapping_boxes(boxes, iou_threshold=0.3)

    print(f"[CANNY] Found {len(boxes)} compartment candidates")
    return boxes


def merge_overlapping_boxes(boxes, iou_threshold=0.3):
    """Merge boxes that overlap significantly — same compartment detected twice"""
    if len(boxes) <= 1:
        return boxes

    merged = []
    used = [False] * len(boxes)

    for i, b1 in enumerate(boxes):
        if used[i]:
            continue
        group = [b1]
        for j, b2 in enumerate(boxes):
            if i == j or used[j]:
                continue
            if compute_iou(b1, b2) > iou_threshold:
                group.append(b2)
                used[j] = True
        # Merge group into one box (union)
        x1 = min(b[0] for b in group)
        y1 = min(b[1] for b in group)
        x2 = max(b[2] for b in group)
        y2 = max(b[3] for b in group)
        merged.append([x1, y1, x2, y2])
        used[i] = True

    return merged


def compute_iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter + 1e-6)


# ============================================================
# YOLO FALLBACK
# Used when Canny doesn't find enough compartments
# ============================================================
def find_compartments_yolo(
    image_path, yolo_model, img_pil, yolo_conf=0.15, min_box_size=50, max_box_ratio=0.40
):
    img_area = img_pil.width * img_pil.height
    results = yolo_model(image_path, conf=yolo_conf, iou=0.5, verbose=False)
    boxes_raw = results[0].boxes.xyxy.cpu().numpy()

    boxes = []
    for box in boxes_raw:
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        if w < min_box_size or h < min_box_size:
            continue
        if (w * h) / img_area > max_box_ratio:
            continue
        boxes.append([x1, y1, x2, y2])

    print(f"[YOLO] Fallback found {len(boxes)} boxes")
    return boxes


# ============================================================
# SAME-LABEL NMS
# Two detections with same label = duplicate, keep highest confidence
# ============================================================
def remove_duplicates(detections, iou_threshold=0.4):
    if len(detections) <= 1:
        return detections

    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    kept = []

    for det in detections:
        b1 = det["box"]
        is_duplicate = False
        for kept_det in kept:
            b2 = kept_det["box"]
            # Same label anywhere = duplicate
            if det["label"] == kept_det["label"]:
                is_duplicate = True
                break
            # High overlap = duplicate
            if compute_iou(b1, b2) > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(det)

    return kept


# ============================================================
# MAIN DETECTION PIPELINE
# ============================================================
def detect_thali(
    image_path,
    yolo_model,
    classifier_model,
    class_names,
    conf_threshold=0.55,
    canny_min_compartments=3,
    crop_padding=8,
):
    """
    1. Try Canny edge detection for compartment boundaries
    2. If < canny_min_compartments found, fallback to YOLO
    3. Classify each compartment crop
    4. Remove duplicates
    """
    try:
        img_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Cannot open {image_path}: {e}")
        return [], None

    img_w, img_h = img_pil.size

    # Stage 1: Try Canny first (works well on clean bird's eye images)
    canny_boxes = find_compartments_canny(img_pil)

    if len(canny_boxes) >= canny_min_compartments:
        print(f"[INFO] Using Canny detection ({len(canny_boxes)} compartments)")
        candidate_boxes = canny_boxes
        source = "canny"
    else:
        print(f"[INFO] Canny found only {len(canny_boxes)}, falling back to YOLO")
        candidate_boxes = find_compartments_yolo(image_path, yolo_model, img_pil)
        source = "yolo"

    if not candidate_boxes:
        print(f"[WARN] No compartments found for {image_path}")
        return [], img_pil

    # Stage 2: Classify each compartment
    raw_detections = []
    for box in candidate_boxes:
        x1, y1, x2, y2 = box

        # Pad crop slightly
        x1p = max(0, x1 - crop_padding)
        y1p = max(0, y1 - crop_padding)
        x2p = min(img_w, x2 + crop_padding)
        y2p = min(img_h, y2 + crop_padding)
        crop = img_pil.crop((x1p, y1p, x2p, y2p))

        pred_idx, pred_conf, top5 = classify_crop(classifier_model, crop, class_names)
        if pred_idx is None:
            continue

        if pred_conf < conf_threshold:
            continue

        raw_detections.append(
            {
                "box": [x1, y1, x2, y2],
                "label": class_names[pred_idx],
                "label_idx": pred_idx,
                "confidence": float(pred_conf),
                "source": source,
                "top5": top5,
            }
        )

    print(f"[INFO] After confidence filter ({conf_threshold}): {len(raw_detections)}")

    # Stage 3: Remove duplicates
    final_detections = remove_duplicates(raw_detections)
    print(f"[INFO] After NMS: {len(final_detections)}")

    for d in final_detections:
        print(f"  -> [{d['source']}] {d['label']:30s} conf={d['confidence']:.3f}")

    return final_detections, img_pil


# ============================================================
# VISUALIZATION
# Also saves the Canny edge map for debugging
# ============================================================
def visualize_detections(image_path, detections, output_dir, save_canny=True):
    img = cv2.imread(image_path)
    if img is None:
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # Save Canny debug image
    if save_canny:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        canny_path = os.path.join(output_dir, f"{Path(image_path).stem}_canny.jpg")
        cv2.imwrite(canny_path, edges)

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
        (255, 0, 128),
    ]

    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["box"]
        label = d["label"]
        conf = d["confidence"]
        source = d.get("source", "")
        color = colors[i % len(colors)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f} [{source}]"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    out_path = os.path.join(output_dir, f"{Path(image_path).stem}_detected.jpg")
    cv2.imwrite(out_path, img)
    print(f"[INFO] Visualization: {out_path}")
    return out_path


# ============================================================
# SAVE RESULTS
# ============================================================
def save_results(image_path, detections, output_dir):
    basename = Path(image_path).stem

    json_path = os.path.join(output_dir, f"{basename}_detections.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "image": basename,
                "num_detections": len(detections),
                "detections": detections,
            },
            f,
            indent=2,
        )

    txt_path = os.path.join(output_dir, f"{basename}.txt")
    with open(txt_path, "w") as f:
        for d in detections:
            b = d["box"]
            f.write(f"{d['label']} {d['confidence']:.4f} {b[0]} {b[1]} {b[2]} {b[3]}\n")

    print(f"[INFO] Results saved: {json_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--input_dir", type=str, default=TASK2_IMAGE_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.55,
        help="Classifier confidence threshold",
    )
    parser.add_argument(
        "--canny_low",
        type=int,
        default=30,
        help="Canny lower threshold. Lower=more edges detected",
    )
    parser.add_argument(
        "--canny_high", type=int, default=100, help="Canny upper threshold"
    )
    parser.add_argument("--single_image", type=str, default=None)
    parser.add_argument("--no_viz", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Conf threshold: {args.conf_threshold}")
    print(f"[INFO] Canny: low={args.canny_low} high={args.canny_high}")

    class_names = get_class_names(DATA_ROOT)
    classifier = load_task1_model(args.checkpoint, num_classes=len(class_names))
    yolo = load_yolo()  # kept as fallback only

    if args.single_image:
        image_paths = [args.single_image]
    else:
        image_paths = sorted(
            [
                str(p)
                for p in Path(args.input_dir).iterdir()
                if p.is_file() and not p.name.startswith(".")
            ]
        )

    if not image_paths:
        print(f"[ERROR] No images found in {args.input_dir}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Processing {len(image_paths)} images")
    print(f"{'=' * 60}\n")

    all_results = []

    for img_path in image_paths:
        print(f"\n{'=' * 60}")
        print(f"Image: {Path(img_path).name}")
        print(f"{'=' * 60}")

        detections, _ = detect_thali(
            img_path, yolo, classifier, class_names, conf_threshold=args.conf_threshold
        )

        save_results(img_path, detections, args.output_dir)

        if not args.no_viz:
            visualize_detections(img_path, detections, args.output_dir)

        all_results.append({"image": Path(img_path).name, "detections": detections})

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Task 2 complete. Results in: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
