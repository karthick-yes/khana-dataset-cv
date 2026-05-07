#!/usr/bin/env python3
"""
SAM Inspector – visualises every step of SAM-based compartment detection.
Usage: python sam_inspector.py --image ~/data/task2_images/Plate146.jpg
"""
from pathlib import Path
import os, sys, argparse, cv2, numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import timm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ── Config ──────────────────────────────────────────────────
SAM_CHECKPOINT = os.path.expanduser("~/data/sam_vit_b_01ec64.pth")
SAM_TYPE = "vit_b"
CHECKPOINT_PATH = os.path.expanduser("~/data/checkpoints/best_model.pt")
DATA_ROOT = os.path.expanduser("~/data/khana")   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = os.path.expanduser("~/data/sam_inspector_outputs")
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

# ── Load classifier ─────────────────────────────────────────
def load_classifier():
    class_names = sorted([d.name for d in Path(DATA_ROOT).iterdir() if d.is_dir()])
    model = timm.create_model("convnext_base.fb_in22k_ft_in1k", pretrained=False,
                              num_classes=len(class_names))
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()
    return model, class_names

# ── Transforms ──────────────────────────────────────────────
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

# ── Main inspector ──────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--outdir", default=OUTPUT_DIR)
    parser.add_argument("--pts_per_side", type=int, default=16)
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
    os.makedirs(args.outdir, exist_ok=True)

    low_vram = not args.no_low_vram
    if low_vram and args.detector_max_side is None:
        args.detector_max_side = 960
    if low_vram and args.pts_per_side == 16:
        args.pts_per_side = 12
    sam_device = resolve_sam_device(args.sam_device)

    # Load image
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

    # Generate all masks
    print("[INFO] Running SAM...")
    masks = mask_gen.generate(sam_input)
    print(f"[INFO] Total raw segments: {len(masks)}")

    # ── 1) Save raw masks overlay ────────────────────────────
    overlay_raw = sam_input.copy()
    for mask in masks:
        # semi‑transparent overlay
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        overlay_raw[mask["segmentation"]] = (overlay_raw[mask["segmentation"]] * 0.5 + color * 0.5).astype(np.uint8)
    if sam_scale != 1.0:
        overlay_raw = cv2.resize(overlay_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    raw_path = os.path.join(args.outdir, f"{Path(args.image).stem}_raw_masks.jpg")
    cv2.imwrite(raw_path, cv2.cvtColor(overlay_raw, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Raw masks saved: {raw_path}")

    # ── 2) Filter masks ──────────────────────────────────────
    img_area = sam_input.shape[0] * sam_input.shape[1]
    kept_boxes = []
    rejected_boxes = []
    for mask in masks:
        area = mask["area"]
        ratio = area / img_area
        if ratio < 0.01 or ratio > 0.35:
            x, y, bw, bh = mask["bbox"]
            rejected_boxes.append(((int(round(x / sam_scale)), int(round(y / sam_scale)), int(round(bw / sam_scale)), int(round(bh / sam_scale))), "area"))
            continue
        x, y, bw, bh = mask["bbox"]
        x = int(round(x / sam_scale))
        y = int(round(y / sam_scale))
        bw = int(round(bw / sam_scale))
        bh = int(round(bh / sam_scale))
        aspect = bw / bh if bh > 0 else 0
        if aspect < 0.15 or aspect > 6.0:
            rejected_boxes.append(((x, y, bw, bh), "aspect"))
            continue
        if bw < 40 or bh < 40:
            rejected_boxes.append(((x, y, bw, bh), "tiny"))
            continue
        kept_boxes.append(mask["bbox"])

    # Save filter visualisation
    filter_vis = img_np.copy()
    for (x, y, bw, bh), reason in rejected_boxes:
        cv2.rectangle(filter_vis, (x, y), (x+bw, y+bh), (0, 0, 255), 2)
        cv2.putText(filter_vis, reason[:4], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)
    for (x, y, bw, bh) in kept_boxes:
        cv2.rectangle(filter_vis, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
    filter_path = os.path.join(args.outdir, f"{Path(args.image).stem}_filter_vis.jpg")
    cv2.imwrite(filter_path, cv2.cvtColor(filter_vis, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Filter visualisation: {filter_path}  (kept={len(kept_boxes)}, rejected={len(rejected_boxes)})")

    # ── 3) Classify each kept box ────────────────────────────
    model, class_names = load_classifier()
    detections = []
    for (x, y, bw, bh) in kept_boxes:
        x1, y1, x2, y2 = x, y, x+bw, y+bh
        crop = img_pil.crop((x1, y1, x2, y2))
        label, conf, top5 = classify_crop(model, crop, class_names)
        print(f"  Box [{x1},{y1},{x2},{y2}] -> {label} ({conf:.2%})  top3: {', '.join([f'{t[0]} {t[1]:.2%}' for t in top5[:3]])}")
        detections.append({"box": [x1, y1, x2, y2], "label": label, "confidence": conf, "top5": top5})

    # Apply simple NMS (remove same label duplicates only if overlap)
    # Here we use a lenient version: keep all unless IoU > 0.1 and same label
    def iou(b1, b2):
        ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    kept_det = []
    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    for d in detections:
        dup = False
        for k in kept_det:
            if d["label"] == k["label"] and iou(d["box"], k["box"]) > 0.1:
                dup = True
                break
        if not dup:
            kept_det.append(d)

    # ── 4) Save final detection image ────────────────────────
    final_img = img_np.copy()
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
    for i, d in enumerate(kept_det):
        x1,y1,x2,y2 = d["box"]
        col = colors[i % len(colors)]
        cv2.rectangle(final_img, (x1,y1), (x2,y2), col, 2)
        txt = f"{d['label']} {d['confidence']:.2f}"
        (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(final_img, (x1, y1-th-8), (x1+tw+4, y1), col, -1)
        cv2.putText(final_img, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    final_path = os.path.join(args.outdir, f"{Path(args.image).stem}_final_detect.jpg")
    cv2.imwrite(final_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Final detections saved: {final_path}")

    print(f"\n[DONE] {len(kept_det)} unique items found.")
    if low_vram and DEVICE == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
