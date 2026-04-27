#!/usr/bin/env python3
"""
Task 2: Thali Food Detection with Correct Khana Labels
Two-stage: YOLOv8 (region proposals) -> Task 1 ConvNeXt (re-classification)
Evaluation metric: Precision-Recall on labels only (not bounding box accuracy)

Fixes in this version:
1. Same-label NMS: if two boxes predict same label, keep only highest confidence
2. Tray-constrained grid: detect tray boundary first, grid only inside tray
3. Square-pad crops before classifying: preserves food content better than center crop
4. Large box filter: skip boxes covering >40% of image
5. Crop padding: expand YOLO boxes slightly before classifying
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
# CLASS NAMES — loaded from dataset folders, same order as training
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
# square-pad preserves the full food content unlike center crop
# which can cut off edges of irregularly shaped food
# ============================================================
def square_pad_transform(crop_pil, target_size=224):
    """
    Square-pad the crop maintaining aspect ratio, then resize.
    Better than center crop for small/irregular thali compartment crops.
    """
    w, h = crop_pil.size
    size = max(w, h)
    # Pad with neutral gray (not black/white which could confuse model)
    padded = Image.new('RGB', (size, size), (114, 114, 114))
    padded.paste(crop_pil, ((size - w) // 2, (size - h) // 2))
    return padded.resize((target_size, target_size), Image.LANCZOS)

NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# LOAD MODELS
# ============================================================
def load_task1_model(checkpoint_path, num_classes=80):
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"[INFO] Loading Task 1 classifier from {checkpoint_path}")
    model = timm.create_model(
        'convnext_base.fb_in22k_ft_in1k',
        pretrained=False,
        num_classes=num_classes
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Classifier loaded. Best val acc: {checkpoint.get('best_acc', 'N/A')}%")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        sys.exit(1)

    model = model.to(DEVICE)
    model.eval()
    return model

def load_yolo():
    print("[INFO] Loading YOLOv8x for region proposals...")
    return YOLO('yolov8x.pt')

# ============================================================
# TRAY DETECTION
# Find the thali tray boundary to constrain grid crops
# ============================================================
def get_tray_box(yolo_results, img_w, img_h):
    """
    Find the tray boundary from YOLO results.
    Strategy:
    1. Look for 'dining table' class first
    2. Fallback: largest box covering 20-70% of image area
    3. Final fallback: use full image (grid will be unconstrained)
    """
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    confs = yolo_results[0].boxes.conf.cpu().numpy()
    classes = yolo_results[0].boxes.cls.cpu().numpy()
    names = yolo_results[0].names
    img_area = img_w * img_h

    # Strategy 1: dining table class
    for box, cls, conf in zip(boxes, classes, confs):
        if names[int(cls)] in ('dining table', 'bowl') and conf > 0.3:
            x1, y1, x2, y2 = map(int, box)
            box_area = (x2 - x1) * (y2 - y1)
            ratio = box_area / img_area
            if 0.15 < ratio < 0.85:  # must be tray-sized, not background
                print(f"[INFO] Tray detected via '{names[int(cls)]}' class: [{x1},{y1},{x2},{y2}]")
                return [x1, y1, x2, y2]

    # Strategy 2: largest box in 20-70% range
    best_box = None
    best_area = 0
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        ratio = area / img_area
        if 0.20 < ratio < 0.70 and area > best_area:
            best_area = area
            best_box = [x1, y1, x2, y2]

    if best_box:
        print(f"[INFO] Tray detected via largest box: {best_box}")
        return best_box

    # Final fallback: slight inset of full image
    inset = int(min(img_w, img_h) * 0.05)
    fallback = [inset, inset, img_w - inset, img_h - inset]
    print(f"[WARN] No tray detected, using inset fallback: {fallback}")
    return fallback

# ============================================================
# TRAY-CONSTRAINED GRID CROPS
# ============================================================
def get_grid_crops(img_pil, tray_box, rows=3, cols=4):
    """
    Divide the TRAY (not full image) into a grid.
    Using cols=4 gives better compartment coverage for typical thalis.
    Only cells that are substantially inside the tray are kept.
    """
    tx1, ty1, tx2, ty2 = tray_box
    tray_w = tx2 - tx1
    tray_h = ty2 - ty1
    cell_w = tray_w // cols
    cell_h = tray_h // rows

    crops = []
    for r in range(rows):
        for c in range(cols):
            x1 = tx1 + c * cell_w
            y1 = ty1 + r * cell_h
            x2 = x1 + cell_w
            y2 = y1 + cell_h
            # Clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_pil.width, x2)
            y2 = min(img_pil.height, y2)
            crops.append((img_pil.crop((x1, y1, x2, y2)), [x1, y1, x2, y2]))

    return crops

# ============================================================
# CLASSIFICATION
# ============================================================
@torch.no_grad()
def classify_crop(model, crop_pil, class_names):
    try:
        # Square pad instead of center crop
        processed = square_pad_transform(crop_pil, target_size=224)
        tensor = NORMALIZE(processed).unsqueeze(0).to(DEVICE)
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
# NMS WITH SAME-LABEL DEDUPLICATION
# Two boxes predicting the same label = duplicate, keep highest conf
# ============================================================
def remove_duplicates(detections, iou_threshold=0.4):
    """
    Two-stage deduplication:
    1. Same label → keep highest confidence, discard others
    2. High IoU overlap → keep highest confidence
    """
    if len(detections) <= 1:
        return detections

    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    kept = []

    for det in detections:
        b1 = det['box']
        is_duplicate = False

        for kept_det in kept:
            b2 = kept_det['box']

            # FIX: Same label = duplicate regardless of box overlap
            # This fixes the 3x gulab jamun / 2x biryani problem
            if det['label'] == kept_det['label']:
                is_duplicate = True
                break

            # High IoU = duplicate even if different label
            ix1 = max(b1[0], b2[0])
            iy1 = max(b1[1], b2[1])
            ix2 = min(b1[2], b2[2])
            iy2 = min(b1[3], b2[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            iou = inter / (area1 + area2 - inter + 1e-6)
            if iou > iou_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(det)

    return kept

# ============================================================
# MAIN DETECTION PIPELINE
# ============================================================
def detect_thali(image_path, yolo_model, classifier_model, class_names,
                 conf_threshold=0.60,
                 grid_conf_threshold=0.70,
                 yolo_conf=0.15,
                 min_box_size=50,
                 max_box_ratio=0.40,
                 crop_padding=10):
    try:
        img_pil = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[ERROR] Cannot open {image_path}: {e}")
        return [], None

    img_w, img_h = img_pil.size
    img_area = img_w * img_h

    # YOLO proposals
    results = yolo_model(image_path, conf=yolo_conf, iou=0.5, verbose=False)
    yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
    yolo_confs_arr = results[0].boxes.conf.cpu().numpy()
    print(f"[INFO] YOLO raw proposals: {len(yolo_boxes)}")

    # Detect tray for constrained grid
    tray_box = get_tray_box(results, img_w, img_h)

    # Tray-constrained grid crops (3 rows x 4 cols inside tray)
    grid = get_grid_crops(img_pil, tray_box, rows=3, cols=4)
    print(f"[INFO] Grid crops (inside tray): {len(grid)}")

    # Combine: (box, yolo_conf, source, threshold)
    all_candidates = []
    for box, yc in zip(yolo_boxes, yolo_confs_arr):
        all_candidates.append((list(map(int, box)), float(yc), 'yolo', conf_threshold))
    for _, box in grid:
        all_candidates.append((box, 0.0, 'grid', grid_conf_threshold))

    raw_detections = []
    for box, yc, source, threshold in all_candidates:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1

        if w < min_box_size or h < min_box_size:
            continue

        # Skip boxes covering too much of the image (entire tray as one box)
        if (w * h) / img_area > max_box_ratio:
            print(f"  [SKIP] Box too large ({(w*h)/img_area:.0%}) source={source}")
            continue

        # Pad crop slightly to be closer to training distribution
        x1p = max(0, x1 - crop_padding)
        y1p = max(0, y1 - crop_padding)
        x2p = min(img_w, x2 + crop_padding)
        y2p = min(img_h, y2 + crop_padding)
        crop = img_pil.crop((x1p, y1p, x2p, y2p))

        pred_idx, pred_conf, top5 = classify_crop(classifier_model, crop, class_names)
        if pred_idx is None:
            continue

        if pred_conf < threshold:
            continue

        raw_detections.append({
            'box': [x1, y1, x2, y2],
            'label': class_names[pred_idx],
            'label_idx': pred_idx,
            'confidence': float(pred_conf),
            'yolo_confidence': yc,
            'source': source,
            'top5': top5
        })

    print(f"[INFO] After filters: {len(raw_detections)}")

    # NMS with same-label deduplication
    final_detections = remove_duplicates(raw_detections, iou_threshold=0.4)
    print(f"[INFO] After NMS: {len(final_detections)}")

    for d in final_detections:
        print(f"  -> [{d['source']}] {d['label']:30s} "
              f"conf={d['confidence']:.3f}  box={d['box']}")

    return final_detections, img_pil

# ============================================================
# VISUALIZATION
# ============================================================
def visualize_detections(image_path, detections, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        pil = Image.open(image_path).convert('RGB')
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0),
        (0, 128, 255), (255, 0, 128)
    ]

    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d['box']
        label = d['label']
        conf = d['confidence']
        source = d.get('source', '')
        color = colors[i % len(colors)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f} [{source}]"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

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
    with open(json_path, 'w') as f:
        json.dump({
            'image': basename,
            'num_detections': len(detections),
            'detections': detections
        }, f, indent=2)

    txt_path = os.path.join(output_dir, f"{basename}.txt")
    with open(txt_path, 'w') as f:
        for d in detections:
            b = d['box']
            f.write(f"{d['label']} {d['confidence']:.4f} "
                    f"{b[0]} {b[1]} {b[2]} {b[3]}\n")

    print(f"[INFO] Results saved: {json_path}")

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--input_dir", type=str, default=TASK2_IMAGE_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--conf_threshold", type=float, default=0.60,
                        help="YOLO crop threshold. Raise=precision, lower=recall")
    parser.add_argument("--grid_conf_threshold", type=float, default=0.70,
                        help="Grid crop threshold (higher = less noise from grid)")
    parser.add_argument("--single_image", type=str, default=None)
    parser.add_argument("--no_viz", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] YOLO conf threshold: {args.conf_threshold}")
    print(f"[INFO] Grid conf threshold: {args.grid_conf_threshold}")

    class_names = get_class_names(DATA_ROOT)
    classifier = load_task1_model(args.checkpoint, num_classes=len(class_names))
    yolo = load_yolo()

    if args.single_image:
        image_paths = [args.single_image]
    else:
        image_paths = sorted([
            str(p) for p in Path(args.input_dir).iterdir()
            if p.is_file() and not p.name.startswith('.')
        ])

    if not image_paths:
        print(f"[ERROR] No images found in {args.input_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Processing {len(image_paths)} images")
    print(f"{'='*60}\n")

    all_results = []

    for img_path in image_paths:
        print(f"\n{'='*60}")
        print(f"Image: {Path(img_path).name}")
        print(f"{'='*60}")

        detections, _ = detect_thali(
            img_path, yolo, classifier, class_names,
            conf_threshold=args.conf_threshold,
            grid_conf_threshold=args.grid_conf_threshold
        )

        save_results(img_path, detections, args.output_dir)

        if not args.no_viz:
            visualize_detections(img_path, detections, args.output_dir)

        all_results.append({
            'image': Path(img_path).name,
            'detections': detections
        })

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Task 2 complete. Results in: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()