#!/usr/bin/env python3
"""
Task 2: Thali Food Detection with Correct Khana Labels
Two-stage: YOLOv8 (region proposals) -> Task 1 ConvNeXt (re-classification)
Evaluation metric: Precision-Recall on labels only (not bounding box accuracy)
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
# CLASS NAMES — loaded from dataset folders (same order as training)
# Never hardcode this — must match exactly what training used
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
# TRANSFORMS — must match val_transform from training exactly
# ============================================================
CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
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
# CLASSIFICATION
# ============================================================
@torch.no_grad()
def classify_crop(model, crop_pil, class_names):
    """Classify a single cropped region using Task 1 model"""
    try:
        tensor = CLASSIFIER_TRANSFORM(crop_pil).unsqueeze(0).to(DEVICE)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_conf = probs.max(dim=1).values.item()
        top5_idx = probs[0].topk(5).indices.cpu().numpy()
        top5 = [(class_names[i], float(probs[0][i])) for i in top5_idx]
        return pred_idx, pred_conf, top5
    except Exception as e:
        print(f"[WARN] Classification failed on crop: {e}")
        return None, 0.0, []

# ============================================================
# NMS ON FINAL DETECTIONS
# Removes overlapping boxes after classifier relabeling
# ============================================================
def remove_duplicates(detections, iou_threshold=0.5):
    if len(detections) <= 1:
        return detections

    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    kept = []

    for det in detections:
        b1 = det['box']
        is_duplicate = False
        for kept_det in kept:
            b2 = kept_det['box']
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
                 conf_threshold=0.55, yolo_conf=0.15, min_box_size=50):
    """
    Stage 1: YOLO generates region proposals (YOLO labels are discarded)
    Stage 2: Task 1 classifier assigns correct Khana labels to each crop
    Stage 3: NMS to remove duplicate detections
    """
    try:
        img_pil = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[ERROR] Cannot open image {image_path}: {e}")
        return [], None

    # Stage 1: YOLO region proposals
    results = yolo_model(image_path, conf=yolo_conf, iou=0.5, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    yolo_confs = results[0].boxes.conf.cpu().numpy()
    print(f"[INFO] YOLO raw proposals: {len(boxes)}")

    # Stage 2: Classify each crop
    raw_detections = []
    for box, yc in zip(boxes, yolo_confs):
        x1, y1, x2, y2 = map(int, box)

        w, h = x2 - x1, y2 - y1
        if w < min_box_size or h < min_box_size:
            continue

        crop = img_pil.crop((x1, y1, x2, y2))
        pred_idx, pred_conf, top5 = classify_crop(classifier_model, crop, class_names)

        if pred_idx is None:
            continue

        if pred_conf >= conf_threshold:
            raw_detections.append({
                'box': [x1, y1, x2, y2],
                'label': class_names[pred_idx],
                'label_idx': pred_idx,
                'confidence': float(pred_conf),
                'yolo_confidence': float(yc),
                'top5': top5
            })

    print(f"[INFO] After confidence filter ({conf_threshold}): {len(raw_detections)}")

    # Stage 3: Remove duplicates
    final_detections = remove_duplicates(raw_detections, iou_threshold=0.5)
    print(f"[INFO] After NMS: {len(final_detections)}")

    for d in final_detections:
        print(f"  -> {d['label']:30s} conf={d['confidence']:.3f}  box={d['box']}")

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
        (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0)
    ]

    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d['box']
        label = d['label']
        conf = d['confidence']
        color = colors[i % len(colors)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    out_path = os.path.join(output_dir, f"{Path(image_path).stem}_detected.jpg")
    cv2.imwrite(out_path, img)
    print(f"[INFO] Visualization saved: {out_path}")
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
    parser.add_argument("--conf_threshold", type=float, default=0.55,
                        help="Raise for higher precision, lower for higher recall")
    parser.add_argument("--single_image", type=str, default=None,
                        help="Process a single image for quick testing")
    parser.add_argument("--no_viz", action="store_true",
                        help="Skip visualization output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Confidence threshold: {args.conf_threshold}")

    # Load class names from dataset — guaranteed same order as training
    class_names = get_class_names(DATA_ROOT)

    # Load models
    classifier = load_task1_model(args.checkpoint, num_classes=len(class_names))
    yolo = load_yolo()

    # Build image list — handles any format
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
            conf_threshold=args.conf_threshold
        )

        save_results(img_path, detections, args.output_dir)

        if not args.no_viz:
            visualize_detections(img_path, detections, args.output_dir)

        all_results.append({
            'image': Path(img_path).name,
            'detections': detections
        })

    # Summary of all images
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Task 2 complete. Results in: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()