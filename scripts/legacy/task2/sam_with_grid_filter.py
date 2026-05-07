#!/usr/bin/env python3
"""
SAM Inspector with Grid Filtering.

Modes:
  default : area + aspect filter only (original behaviour)
  grid    : automatically detect tray → 2x3 grid → only masks that fall inside grid cells
  manual  : use hardcoded relative manual boxes to filter masks

Usage:
  python sam_with_grid_filter.py --image ~/data/task2_images/Plate146.jpg --mode grid
  python sam_with_grid_filter.py --image ~/data/task2_images/Plate146.jpg --mode manual
  python sam_with_grid_filter.py --image ~/data/task2_images/Plate146.jpg --mode default
"""

import os, sys, argparse, cv2, numpy as np
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import timm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ── Config ──────────────────────────────────────────────────
SAM_CHECKPOINT = os.path.expanduser("~/data/sam_vit_b_01ec64.pth")
SAM_TYPE = "vit_b"
CHECKPOINT_PATH = os.path.expanduser("~/data/checkpoints/best_model.pt")
DATA_ROOT = os.path.expanduser("~/data/khana")   # adjust if different
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = os.path.expanduser("~/data/sam_grid_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def resolve_sam_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested for SAM but unavailable, using CPU")
        return "cpu"
    return requested


def resize_for_sam(img_np, max_side):
    if not max_side:
        return img_np, 1.0
    h, w = img_np.shape[:2]
    if max(h, w) <= max_side:
        return img_np, 1.0
    scale = max_side / float(max(h, w))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

# ── Manual calibration boxes (relative, from your previous manual.py) ──
MANUAL_BOXES_REL = [
    [320/1280, 114/720, 426/1280, 247/720],
    [520/1280, 121/720, 716/1280, 247/720],
    [707/1280, 125/720, 889/1280, 222/720],
    [288/1280, 252/720, 359/1280, 453/720],
    [412/1280, 276/720, 778/1280, 455/720],
    [818/1280, 254/720, 948/1280, 455/720],
]

# ── Classifier ──────────────────────────────────────────────
def load_classifier():
    class_names = sorted([d.name for d in Path(DATA_ROOT).iterdir() if d.is_dir()])
    model = timm.create_model("convnext_base.fb_in22k_ft_in1k", pretrained=False,
                              num_classes=len(class_names))
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()
    return model, class_names

def square_pad_and_normalize(crop_pil, size=224):
    w, h = crop_pil.size
    s = max(w, h)
    padded = Image.new("RGB", (s, s), (114, 114, 114))
    padded.paste(crop_pil, ((s - w)//2, (s - h)//2))
    resized = padded.resize((size, size), Image.LANCZOS)
    tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(resized)
    return tensor

@torch.no_grad()
def classify_crop(model, crop_pil, class_names):
    tensor = square_pad_and_normalize(crop_pil).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(tensor), dim=1)
    idx = probs.argmax(dim=1).item()
    conf = probs.max(dim=1).values.item()
    top5 = [(class_names[i], float(probs[0][i])) for i in probs[0].topk(5).indices]
    return class_names[idx], conf, top5

# ── Tray detection (same as task2_detect.py) ─────────────────
def find_tray_bbox(img_np):
    """Auto-detect tray bounding box using silver mask + Canny fallback."""
    h, w = img_np.shape[:2]
    # Method 1: silver mask
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 0, 140])
    upper = np.array([180, 90, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) / (w * h) >= 0.12:
            x, y, bw, bh = cv2.boundingRect(largest)
            return [x, y, x + bw, y + bh]
    # Method 2: Canny edge largest blob
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 120)
    edges = cv2.dilate(edges, None, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) / (w * h) >= 0.12:
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            x, y, bw, bh = cv2.boundingRect(box.astype(int))
            return [x, y, x + bw, y + bh]
    return None

def get_grid_boxes(img_w, img_h, mode="grid", tray_bbox=None):
    """
    Returns a list of grid cell boxes [x1,y1,x2,y2] for filtering.
    mode='grid' -> auto tray detection + 2x3
    mode='manual' -> scaled manual boxes
    """
    if mode == "manual":
        boxes = []
        for rel in MANUAL_BOXES_REL:
            boxes.append([
                int(rel[0] * img_w), int(rel[1] * img_h),
                int(rel[2] * img_w), int(rel[3] * img_h)
            ])
        return boxes
    else:  # grid mode
        if tray_bbox is None:
            # fallback central 80%
            x1 = int(img_w * 0.1)
            y1 = int(img_h * 0.1)
            x2 = int(img_w * 0.9)
            y2 = int(img_h * 0.9)
        else:
            x1, y1, x2, y2 = tray_bbox
        cell_w = (x2 - x1) / 3
        cell_h = (y2 - y1) / 2
        boxes = []
        for r in range(2):
            for c in range(3):
                boxes.append([
                    int(x1 + c * cell_w), int(y1 + r * cell_h),
                    int(x1 + (c+1) * cell_w), int(y1 + (r+1) * cell_h)
                ])
        return boxes

def mask_in_grid(mask_bbox, grid_boxes, method="center"):
    """
    Check if a mask's bbox is inside any grid cell.
    method='center': mask centre must be inside cell.
    method='iou': IoU with cell > 0.3.
    """
    x, y, w, h = mask_bbox
    cx, cy = x + w/2, y + h/2
    for gbox in grid_boxes:
        if method == "center":
            if gbox[0] <= cx <= gbox[2] and gbox[1] <= cy <= gbox[3]:
                return True
        elif method == "iou":
            ix1 = max(x, gbox[0]); iy1 = max(y, gbox[1])
            ix2 = min(x+w, gbox[2]); iy2 = min(y+h, gbox[3])
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                area = w*h
                if inter / area > 0.3:
                    return True
    return False

# ── Main inspector ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--mode", choices=["default", "grid", "manual"], default="grid")
    parser.add_argument("--outdir", default=OUTPUT_DIR)
    parser.add_argument("--pts_per_side", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--sam_device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--detector_max_side", type=int, default=None)
    parser.add_argument(
        "--no_low_vram",
        action="store_true",
        help="Disable automatic low-VRAM defaults",
    )
    args = parser.parse_args()
    
    # Expand user paths
    args.image = os.path.expanduser(args.image)
    args.outdir = os.path.expanduser(args.outdir)

    low_vram = not args.no_low_vram
    if low_vram and args.detector_max_side is None:
        args.detector_max_side = 960
    if low_vram and args.pts_per_side == 16:
        args.pts_per_side = 12

    sam_device = resolve_sam_device(args.sam_device)

    img_pil = Image.open(args.image).convert("RGB")
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]
    sam_input, sam_scale = resize_for_sam(img_np, args.detector_max_side)

    # Load SAM
    print(f"[INFO] Loading SAM on {sam_device}...")
    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT).to(sam_device)
    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.pts_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=500,
    )
    print(
        f"[INFO] pts_per_side={args.pts_per_side} detector_max_side={args.detector_max_side} low_vram={low_vram}"
    )

    print("[INFO] Running SAM...")
    masks = mask_gen.generate(sam_input)
    print(f"[INFO] Total raw segments: {len(masks)}")

    # ── 1) Save raw masks overlay ────────────────────────────
    overlay_raw = sam_input.copy()
    for m in masks:
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        overlay_raw[m["segmentation"]] = (overlay_raw[m["segmentation"]] * 0.5 + color * 0.5).astype(np.uint8)
    if sam_scale != 1.0:
        overlay_raw = cv2.resize(overlay_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(args.outdir, f"{Path(args.image).stem}_raw.jpg"),
                cv2.cvtColor(overlay_raw, cv2.COLOR_RGB2BGR))

    # ── 2) Determine grid cells ──────────────────────────────
    if args.mode == "default":
        grid_boxes = None
    else:
        if args.mode == "grid":
            tray_bbox = find_tray_bbox(img_np)
            if tray_bbox:
                print(f"[INFO] Tray detected: {tray_bbox}")
            else:
                print("[INFO] Tray not found, using central 80%")
        else:
            tray_bbox = None
        grid_boxes = get_grid_boxes(w, h, mode=args.mode, tray_bbox=tray_bbox if args.mode == 'grid' else None)

        # Draw grid on image for visualisation
        grid_vis = img_np.copy()
        for box in grid_boxes:
            cv2.rectangle(grid_vis, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
        cv2.imwrite(os.path.join(args.outdir, f"{Path(args.image).stem}_grid.jpg"),
                    cv2.cvtColor(grid_vis, cv2.COLOR_RGB2BGR))

    # ── 3) Filter masks ─────────────────────────────────────
    img_area = sam_input.shape[0] * sam_input.shape[1]
    kept_boxes = []
    rejected = []

    for m in masks:
        area = m["area"]
        ratio = area / img_area
        x, y, bw, bh = m["bbox"]
        x = int(round(x / sam_scale))
        y = int(round(y / sam_scale))
        bw = int(round(bw / sam_scale))
        bh = int(round(bh / sam_scale))
        scaled_bbox = (x, y, bw, bh)
        aspect = bw / bh if bh > 0 else 0

        # Basic area/aspect filter
        if ratio < 0.01 or ratio > 0.35:
            rejected.append((scaled_bbox, "area"))
            continue
        if aspect < 0.15 or aspect > 6.0:
            rejected.append((scaled_bbox, "aspect"))
            continue
        if bw < 40 or bh < 40:
            rejected.append((scaled_bbox, "tiny"))
            continue

        # Grid filter (if not default)
        if grid_boxes is not None:
            if not mask_in_grid(scaled_bbox, grid_boxes, method="center"):
                rejected.append((scaled_bbox, "grid"))
                continue

        kept_boxes.append(scaled_bbox)

    # Save filter visualisation
    filter_vis = img_np.copy()
    for (rx, ry, rbw, rbh), reason in rejected:
        cv2.rectangle(filter_vis, (rx, ry), (rx+rbw, ry+rbh), (0,0,255), 2)
        cv2.putText(filter_vis, reason[:4], (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,0,255), 1)
    for (kx, ky, kbw, kbh) in kept_boxes:
        cv2.rectangle(filter_vis, (kx, ky), (kx+kbw, ky+kbh), (0,255,0), 2)
    cv2.imwrite(os.path.join(args.outdir, f"{Path(args.image).stem}_filter.jpg"),
                cv2.cvtColor(filter_vis, cv2.COLOR_RGB2BGR))
    print(f"[INFO] After filtering: {len(kept_boxes)} kept, {len(rejected)} rejected")

    # ── 4) Classify & final NMS ──────────────────────────────
    model, class_names = load_classifier()
    detections = []
    for (x, y, bw, bh) in kept_boxes:
        crop = img_pil.crop((x, y, x+bw, y+bh))
        label, conf, top5 = classify_crop(model, crop, class_names)
        print(f"  [{x},{y},{x+bw},{y+bh}] -> {label} ({conf:.2%})  top3: {', '.join([f'{t[0]} {t[1]:.2%}' for t in top5[:3]])}")
        if conf >= args.conf:
            detections.append({"box": [x, y, x+bw, y+bh], "label": label, "conf": conf, "top5": top5})

    # NMS: allow same label if boxes far apart
    def iou(b1, b2):
        ix1 = max(b1[0],b2[0]); iy1 = max(b1[1],b2[1])
        ix2 = min(b1[2],b2[2]); iy2 = min(b1[3],b2[3])
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter/(area1+area2-inter+1e-6)

    detections.sort(key=lambda d: d["conf"], reverse=True)
    kept_det = []
    for d in detections:
        dup = False
        for k in kept_det:
            if d["label"] == k["label"] and iou(d["box"], k["box"]) > 0.1:
                dup = True
                break
        if not dup:
            kept_det.append(d)

    # ── 5) Final detection image ─────────────────────────────
    final_img = img_np.copy()
    colors = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]
    for i, d in enumerate(kept_det):
        x1,y1,x2,y2 = d["box"]
        col = colors[i%len(colors)]
        cv2.rectangle(final_img, (x1,y1), (x2,y2), col, 2)
        txt = f"{d['label']} {d['conf']:.2f}"
        cv2.putText(final_img, txt, (x1, max(y1-5,15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.imwrite(os.path.join(args.outdir, f"{Path(args.image).stem}_final.jpg"),
                cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    print(f"\n[DONE] {len(kept_det)} unique items after NMS.")
    print(f"Outputs saved in {args.outdir}")
    if low_vram and DEVICE == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
