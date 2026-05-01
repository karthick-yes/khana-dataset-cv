#!/usr/bin/env python3
"""
Task 2: Thali Food Detection with Correct Khana Labels

Detection strategies (chosen by --method):
  auto   : SAM → Hough → YOLO
  sam    : SAM only (fallback to Hough if SAM not installed)
  hough  : Hough Lines → YOLO fallback
  yolo   : YOLO only
  grid   : Auto-detect tray via silver mask, then 2×3 grid
  manual : Use manually calibrated compartment boxes scaled to image size

Each detected region is classified by the Task 1 ConvNeXt model.
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

# SAM checkpoint — download with:
# wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
SAM_CHECKPOINT = os.path.expanduser("~/data/sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"

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
# TRANSFORMS — square pad preserves food content
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
    print(f"[INFO] Loading Task 1 classifier...")
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
    print("[INFO] Loading YOLOv8x...")
    return YOLO("yolov8x.pt")


def load_sam():
    if not os.path.exists(SAM_CHECKPOINT):
        print(f"[WARN] SAM checkpoint not found at {SAM_CHECKPOINT}")
        print(f"[WARN] Download with:")
        print(
            f"  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O {SAM_CHECKPOINT}"
        )
        return None
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        print(f"[INFO] Loading SAM ({SAM_MODEL_TYPE})...")
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
        sam = sam.to(DEVICE)
        mask_generator = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            min_mask_region_area=500,
        )
        print(f"[INFO] SAM loaded.")
        return mask_generator
    except ImportError:
        print(
            f"[WARN] segment_anything not installed. Run: pip install segment-anything"
        )
        return None


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
# METHOD 1: SAM SEGMENTATION (unchanged)
# ============================================================
def find_compartments_sam(
    img_pil, sam_generator, min_area_ratio=0.01, max_area_ratio=0.35
):
    img_np = np.array(img_pil)
    img_area = img_np.shape[0] * img_np.shape[1]
    print(f"[SAM] Running automatic segmentation...")
    masks = sam_generator.generate(img_np)
    print(f"[SAM] Got {len(masks)} raw segments")
    boxes = []
    for mask_data in masks:
        area = mask_data["area"]
        ratio = area / img_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        x, y, w, h = mask_data["bbox"]
        x1, y1, x2, y2 = x, y, x + w, y + h
        aspect = w / h if h > 0 else 0
        if aspect < 0.15 or aspect > 6.0:
            continue
        if w < 40 or h < 40:
            continue
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    boxes = merge_overlapping_boxes(boxes, iou_threshold=0.4)
    print(f"[SAM] After filtering: {len(boxes)} compartments")
    return boxes


# ============================================================
# METHOD 2: HOUGH LINES (unchanged)
# ============================================================
def find_compartments_hough(img_pil, min_area_ratio=0.01, max_area_ratio=0.40):
    img_np = np.array(img_pil)
    img_w, img_h = img_pil.size
    img_area = img_w * img_h
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    min_line_length = int(min(img_w, img_h) * 0.15)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=20,
    )
    if lines is None:
        print(f"[HOUGH] No lines found")
        return []
    print(f"[HOUGH] Found {len(lines)} line segments")
    h_lines, v_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 15 or angle > 165:
            h_lines.append((y1 + y2) // 2)
        elif 75 < angle < 105:
            v_lines.append((x1 + x2) // 2)
    if len(h_lines) < 2 or len(v_lines) < 2:
        print(f"[HOUGH] Not enough grid lines (h={len(h_lines)}, v={len(v_lines)})")
        return []
    h_lines = cluster_lines(sorted(h_lines), gap=int(img_h * 0.05))
    v_lines = cluster_lines(sorted(v_lines), gap=int(img_w * 0.05))
    print(f"[HOUGH] Grid: {len(h_lines)} horizontal x {len(v_lines)} vertical lines")
    h_lines = sorted([0] + h_lines + [img_h])
    v_lines = sorted([0] + v_lines + [img_w])
    boxes = []
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            y1 = h_lines[i]
            y2 = h_lines[i + 1]
            x1 = v_lines[j]
            x2 = v_lines[j + 1]
            w, h = x2 - x1, y2 - y1
            area = w * h
            ratio = area / img_area
            if ratio < min_area_ratio or ratio > max_area_ratio:
                continue
            if w < 40 or h < 40:
                continue
            boxes.append([x1, y1, x2, y2])
    print(f"[HOUGH] Found {len(boxes)} compartment boxes")
    return boxes


def cluster_lines(lines, gap=30):
    if not lines:
        return []
    clusters = [[lines[0]]]
    for line in lines[1:]:
        if line - clusters[-1][-1] < gap:
            clusters[-1].append(line)
        else:
            clusters.append([line])
    return [int(np.mean(c)) for c in clusters]


# ============================================================
# METHOD 3: YOLO FALLBACK (unchanged)
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
    print(f"[YOLO] Found {len(boxes)} boxes")
    return boxes


# ============================================================
# METHOD 4: AUTOMATIC TRAY DETECTION + 2×3 GRID
# ============================================================
def find_tray_bbox(img_pil):
    """
    Find the bounding box of the thali tray using a colour mask for
    shiny stainless steel (low saturation, high value).
    Returns [x1, y1, x2, y2] or None.
    """
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 0, 180])
    upper = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area / (w * h) < 0.12:
        return None

    x, y, bw, bh = cv2.boundingRect(largest)
    return [x, y, x + bw, y + bh]


def find_compartments_tray_grid(img_pil, rows=2, cols=3):
    """
    Detect tray bounding box automatically and partition it into rows x cols grid.
    Falls back to central 80% of image if tray not detected.
    """
    tray_bbox = find_tray_bbox(img_pil)

    if tray_bbox is None:
        w, h = img_pil.size
        margin = 0.1
        tray_bbox = [
            int(w * margin),
            int(h * margin),
            int(w * (1 - margin)),
            int(h * (1 - margin)),
        ]
        print("[TRAY] Not found – using central 80% region")

    x1, y1, x2, y2 = tray_bbox
    cell_w = (x2 - x1) / cols
    cell_h = (y2 - y1) / rows

    boxes = []
    for r in range(rows):
        for c in range(cols):
            box = [
                int(x1 + c * cell_w),
                int(y1 + r * cell_h),
                int(x1 + (c + 1) * cell_w),
                int(y1 + (r + 1) * cell_h),
            ]
            boxes.append(box)
    return boxes


# ============================================================
# METHOD 5: MANUAL CALIBRATION GRID (your own boxes)
# ============================================================
# Manual boxes from your calibration (1280x720 image) expressed as relative coords
MANUAL_BOXES_REL = [
    [320 / 1280, 114 / 720, 426 / 1280, 247 / 720],
    [520 / 1280, 121 / 720, 716 / 1280, 247 / 720],
    [707 / 1280, 125 / 720, 889 / 1280, 222 / 720],
    [288 / 1280, 252 / 720, 359 / 1280, 453 / 720],
    [412 / 1280, 276 / 720, 778 / 1280, 455 / 720],
    [818 / 1280, 254 / 720, 948 / 1280, 455 / 720],
]


def find_compartments_manual_grid(img_pil):
    """Use manually calibrated compartment positions, scaled to image size."""
    w, h = img_pil.size
    boxes = []
    for rel in MANUAL_BOXES_REL:
        boxes.append(
            [
                int(rel[0] * w),
                int(rel[1] * h),
                int(rel[2] * w),
                int(rel[3] * h),
            ]
        )
    return boxes


# ============================================================
# SHARED UTILITIES
# ============================================================
def compute_iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def merge_overlapping_boxes(boxes, iou_threshold=0.4):
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
        merged.append(
            [
                min(b[0] for b in group),
                min(b[1] for b in group),
                max(b[2] for b in group),
                max(b[3] for b in group),
            ]
        )
        used[i] = True
    return merged


def remove_duplicates(detections, iou_threshold=0.4):
    if len(detections) <= 1:
        return detections
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    kept = []
    for det in detections:
        b1 = det["box"]
        is_dup = False
        for k in kept:
            b2 = k["box"]
            if det["label"] == k["label"]:
                is_dup = True
                break
            if compute_iou(b1, b2) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
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
    sam_generator=None,
    method="auto",
    conf_threshold=0.55,
    crop_padding=8,
):
    """
    method='auto', 'sam', 'hough', 'yolo', 'grid', 'manual'
    """
    try:
        img_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Cannot open {image_path}: {e}")
        return [], None

    img_w, img_h = img_pil.size
    candidate_boxes = []
    source = "unknown"

    # --- Method selection ---
    if method == "grid":
        candidate_boxes = find_compartments_tray_grid(img_pil)
        source = "grid"
    elif method == "manual":
        candidate_boxes = find_compartments_manual_grid(img_pil)
        source = "manual"
    elif method in ("auto", "sam") and sam_generator is not None:
        candidate_boxes = find_compartments_sam(img_pil, sam_generator)
        source = "sam"
        if len(candidate_boxes) < 3:
            print(f"[INFO] SAM found only {len(candidate_boxes)}, trying Hough...")
            candidate_boxes = find_compartments_hough(img_pil)
            source = "hough"
    elif method in ("auto", "hough"):
        candidate_boxes = find_compartments_hough(img_pil)
        source = "hough"
        if len(candidate_boxes) < 3 and method == "auto":
            print(
                f"[INFO] Hough found only {len(candidate_boxes)}, falling back to YOLO..."
            )
            candidate_boxes = find_compartments_yolo(image_path, yolo_model, img_pil)
            source = "yolo"
    elif method == "yolo":
        candidate_boxes = find_compartments_yolo(image_path, yolo_model, img_pil)
        source = "yolo"
    else:
        # Fallback if method not recognised
        candidate_boxes = find_compartments_yolo(image_path, yolo_model, img_pil)
        source = "yolo"

    if not candidate_boxes:
        print(f"[WARN] No compartments found for {image_path}")
        return [], img_pil

    print(f"[INFO] Using {len(candidate_boxes)} boxes from [{source}]")

    # --- Classify each region ---
    raw_detections = []
    for box in candidate_boxes:
        x1, y1, x2, y2 = box
        x1p = max(0, x1 - crop_padding)
        y1p = max(0, y1 - crop_padding)
        x2p = min(img_w, x2 + crop_padding)
        y2p = min(img_h, y2 + crop_padding)
        crop = img_pil.crop((x1p, y1p, x2p, y2p))

        pred_idx, pred_conf, top5 = classify_crop(classifier_model, crop, class_names)
        if pred_idx is None or pred_conf < conf_threshold:
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
    final_detections = remove_duplicates(raw_detections)
    print(f"[INFO] After NMS: {len(final_detections)}")

    for d in final_detections:
        print(f"  -> [{d['source']}] {d['label']:30s} conf={d['confidence']:.3f}")

    return final_detections, img_pil


# ============================================================
# VISUALIZATION
# ============================================================
def visualize_detections(image_path, detections, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

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
        color = colors[i % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{d['label']} {d['confidence']:.2f} [{d.get('source', '')}]"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    out_path = os.path.join(output_dir, f"{Path(image_path).stem}_detected.jpg")
    cv2.imwrite(out_path, img)
    print(f"[INFO] Saved: {out_path}")

    # Hough debug visualisation (only if hough is potentially used)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    h, w = edges.shape
    min_len = int(min(w, h) * 0.15)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 50, minLineLength=min_len, maxLineGap=20
    )
    hough_vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    hough_path = os.path.join(output_dir, f"{Path(image_path).stem}_hough.jpg")
    cv2.imwrite(hough_path, hough_vis)

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
    parser.add_argument("--conf_threshold", type=float, default=0.55)
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "sam", "hough", "yolo", "grid", "manual"],
        help="auto: SAM→Hough→YOLO | sam: SAM first | hough: Hough Lines | yolo: YOLO | grid: tray detection + 2x3 grid | manual: your calibrated boxes",
    )
    parser.add_argument("--sam_checkpoint", type=str, default=SAM_CHECKPOINT)
    parser.add_argument("--single_image", type=str, default=None)
    parser.add_argument("--no_viz", action="store_true")
    args = parser.parse_args()

    # Expand user paths
    args.single_image = os.path.expanduser(args.single_image) if args.single_image else None
    args.input_dir = os.path.expanduser(args.input_dir)
    args.output_dir = os.path.expanduser(args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Method: {args.method}")
    print(f"[INFO] Conf threshold: {args.conf_threshold}")

    class_names = get_class_names(DATA_ROOT)
    classifier = load_task1_model(args.checkpoint, num_classes=len(class_names))
    yolo = load_yolo()

    # Load SAM only if needed
    sam_generator = None
    if args.method in ("auto", "sam"):
        sam_generator = load_sam()
        if sam_generator is None and args.method == "sam":
            print("[WARN] SAM not available, falling back to hough")

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
            img_path,
            yolo,
            classifier,
            class_names,
            sam_generator=sam_generator,
            method=args.method,
            conf_threshold=args.conf_threshold,
        )

        save_results(img_path, detections, args.output_dir)
        if not args.no_viz:
            visualize_detections(img_path, detections, args.output_dir)

        all_results.append({"image": Path(img_path).name, "detections": detections})

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done. Results in: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
