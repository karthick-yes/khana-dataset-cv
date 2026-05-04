#!/usr/bin/env python3
"""
Task 2: Thali Food Detection using a calibrated tray grid.
Run once with --calibrate to generate tray_calibration.json, then process all images.
"""
import os, sys, json, argparse
from pathlib import Path
import torch, numpy as np, cv2
from PIL import Image, ImageFile
from torchvision import transforms
import timm

from config import get_paths

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Config ────────────────────────────────────────────────────
paths = get_paths()
DATA_ROOT = paths["DATA_ROOT"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASK2_IMAGE_DIR = os.path.expanduser("~/data/task2_images")
OUTPUT_DIR = os.path.expanduser("~/data/task2_output")
CHECKPOINT_PATH = os.path.expanduser("~/data/checkpoints/best_model.pt")
CALIB_FILE = os.path.join(os.path.dirname(__file__) or ".", "tray_calibration.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Class names ───────────────────────────────────────────────
def get_class_names(data_root):
    root = Path(data_root)
    if not root.exists():
        print(f"[ERROR] Data root not found: {data_root}")
        sys.exit(1)
    return sorted([d.name for d in root.iterdir() if d.is_dir()])

# ── Transforms ────────────────────────────────────────────────
def square_pad_and_normalize(crop_pil, target_size=224):
    w, h = crop_pil.size
    size = max(w, h)
    padded = Image.new("RGB", (size, size), (114, 114, 114))
    padded.paste(crop_pil, ((size - w) // 2, (size - h) // 2))
    resized = padded.resize((target_size, target_size), Image.LANCZOS)
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(resized)
    return tensor

# ── Load model ────────────────────────────────────────────────
def load_model(checkpoint_path, num_classes):
    model = timm.create_model(
        "convnext_base.fb_in22k_ft_in1k", pretrained=False, num_classes=num_classes
    )
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()
    return model

@torch.no_grad()
def classify_crop(model, crop_pil, class_names):
    tensor = square_pad_and_normalize(crop_pil).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(tensor), dim=1)
    idx = probs.argmax(dim=1).item()
    conf = probs.max(dim=1).values.item()
    top5_idx = probs[0].topk(5).indices.cpu().numpy()
    top5 = [(class_names[i], float(probs[0][i])) for i in top5_idx]
    return idx, conf, top5

# ── Generate grid boxes from calibration ──────────────────────
def get_grid_boxes(img_w, img_h):
    """
    Load the calibration file and return a list of [x1,y1,x2,y2] boxes
    for a 2x3 grid inside the tray.
    If calibration file not found, fallback to central 80% region.
    """
    if not os.path.exists(CALIB_FILE):
        print("[WARN] No calibration file found! Using central 80% fallback.")
        margin = 0.1
        x1, y1 = int(img_w * margin), int(img_h * margin)
        x2, y2 = int(img_w * (1 - margin)), int(img_h * (1 - margin))
    else:
        with open(CALIB_FILE, "r") as f:
            calib = json.load(f)
        # Scale relative coords to this image
        x1 = int(calib["rel_x1"] * img_w)
        y1 = int(calib["rel_y1"] * img_h)
        x2 = int(calib["rel_x2"] * img_w)
        y2 = int(calib["rel_y2"] * img_h)
        # Safety clip
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w-1, x2), min(img_h-1, y2)

    ROWS, COLS = 2, 3
    cell_w = (x2 - x1) / COLS
    cell_h = (y2 - y1) / ROWS
    boxes = []
    for r in range(ROWS):
        for c in range(COLS):
            boxes.append([
                int(x1 + c * cell_w),
                int(y1 + r * cell_h),
                int(x1 + (c+1) * cell_w),
                int(y1 + (r+1) * cell_h),
            ])
    return boxes

# ── Duplicate removal (keep same label only if boxes are close) ─
def remove_duplicates(detections, iou_thr=0.1, same_label_thr=0.05):
    """
    Keep only one detection per food label unless boxes are far apart.
    For the same label, keep the one with highest confidence.
    """
    if not detections:
        return []
    kept = []
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    for d in detections:
        duplicate = False
        for k in kept:
            # Same label – check if box is very close (same compartment)
            if d["label"] == k["label"]:
                # If IoU is very small, they are likely different compartments with same food
                iou = compute_iou(d["box"], k["box"])
                if iou > same_label_thr:
                    duplicate = True
                    break
            else:
                # Different label but highly overlapping – keep higher confidence
                if compute_iou(d["box"], k["box"]) > iou_thr:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(d)
    return kept

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (area1 + area2 - inter + 1e-6)

# ── Main detection per image ──────────────────────────────────
def detect_thali(image_path, model, class_names, conf_threshold=0.55):
    try:
        img_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Cannot open {image_path}: {e}")
        return []

    img_w, img_h = img_pil.size
    boxes = get_grid_boxes(img_w, img_h)

    raw_detections = []
    for box in boxes:
        x1, y1, x2, y2 = box
        # Add a small crop padding
        pad = 8
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(img_w, x2 + pad)
        y2p = min(img_h, y2 + pad)
        crop = img_pil.crop((x1p, y1p, x2p, y2p))
        pred_idx, conf, top5 = classify_crop(model, crop, class_names)
        if conf < conf_threshold:
            continue
        raw_detections.append({
            "box": [x1, y1, x2, y2],
            "label": class_names[pred_idx],
            "confidence": float(conf),
            "top5": top5,
        })

    # Remove duplicates while allowing same label in different compartments
    final_detections = remove_duplicates(raw_detections)
    print(f"[INFO] {len(final_detections)} detections after duplicate removal")
    for d in final_detections:
        print(f"  -> {d['label']:30s} conf={d['confidence']:.3f}")
    return final_detections

# ── Visualization & saving ────────────────────────────────────
def visualize_and_save(image_path, detections, output_dir, grid_boxes=None):
    img = cv2.imread(image_path)
    if img is None:
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # Draw grid boxes (green)
    if grid_boxes:
        for box in grid_boxes:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0), 1)

    # Draw detection boxes
    colors = [(0,0,255), (255,0,0), (0,255,255), (255,0,255), (255,255,0)]
    for i, d in enumerate(detections):
        x1,y1,x2,y2 = d["box"]
        col = colors[i % len(colors)]
        cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
        txt = f"{d['label']} {d['confidence']:.2f}"
        (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1-th-8), (x1+tw+4, y1), col, -1)
        cv2.putText(img, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    stem = Path(image_path).stem
    out_path = os.path.join(output_dir, f"{stem}_detected.jpg")
    cv2.imwrite(out_path, img)

    # Save JSON
    json_out = os.path.join(output_dir, f"{stem}_detections.json")
    with open(json_out, "w") as f:
        json.dump({
            "image": stem,
            "num_detections": len(detections),
            "detections": detections,
        }, f, indent=2)

    print(f"[INFO] Saved {out_path} and {json_out}")

# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_threshold", type=float, default=0.55)
    parser.add_argument("--single_image", type=str, default=None)
    parser.add_argument("--calibrate", action="store_true",
                        help="Run interactive calibration on the single image first (requires display)")
    args = parser.parse_args()
    args.single_image = os.path.expanduser(args.single_image) if args.single_image else None

    if args.calibrate:
        # Launch the calibration script (must have display)
        import subprocess
        print("Starting calibration...")
        cmd = [sys.executable, "grid_calibrator.py"]
        if args.single_image:
            cmd += ["--image", args.single_image]
        subprocess.run(cmd, check=True)
        print("Calibration done. You can now run processing.")
        sys.exit(0)

    # Load model
    class_names = get_class_names(DATA_ROOT)
    model = load_model(CHECKPOINT_PATH, len(class_names))

    if args.single_image:
        images = [args.single_image]
    else:
        images = sorted([
            str(p) for p in Path(TASK2_IMAGE_DIR).iterdir()
            if p.is_file() and not p.name.startswith('.')
        ])

    print(f"Processing {len(images)} images...")
    for img_path in images:
        print(f"\n{'='*40}\n{Path(img_path).name}")
        grid_boxes = get_grid_boxes(*Image.open(img_path).size)  # for visualization
        detections = detect_thali(img_path, model, class_names, args.conf_threshold)
        visualize_and_save(img_path, detections, OUTPUT_DIR, grid_boxes)

if __name__ == "__main__":
    main()
