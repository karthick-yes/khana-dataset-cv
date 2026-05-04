#!/usr/bin/env python3
"""
Task 2 – SAM segmentation + automatic tray grid (no calibration needed).

How it works:
  SAM finds all segments → filter by area/aspect → bounding box of
  all remaining masks = the thali tray → 2×3 grid inside that box →
  keep only masks whose centre lies in a grid cell → classify → NMS.

Usage:
  python task2_sam_grid.py --image ~/data/task2_images/Plate146.jpg
  python task2_sam_grid.py --input_dir ~/data/task2_images --conf 0.5
"""

import os, sys, argparse, json, cv2, numpy as np
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
import timm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ── Config ──────────────────────────────────────────────────
SAM_CKPT = os.path.expanduser("~/data/sam_vit_b_01ec64.pth")
CLF_CKPT = os.path.expanduser("~/data/checkpoints/best_model.pt")
DATA_ROOT = os.path.expanduser("~/data/khana")        # adjust if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = os.path.expanduser("~/data/task2_sam_grid_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Classifier ──────────────────────────────────────────────
def load_classifier():
    classes = sorted(d.name for d in Path(DATA_ROOT).iterdir() if d.is_dir())
    model = timm.create_model("convnext_base.fb_in22k_ft_in1k", pretrained=False,
                              num_classes=len(classes))
    ckpt = torch.load(CLF_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()
    return model, classes

def square_pad(crop_pil, size=224):
    w, h = crop_pil.size
    s = max(w, h)
    padded = Image.new("RGB", (s, s), (114, 114, 114))
    padded.paste(crop_pil, ((s-w)//2, (s-h)//2))
    padded = padded.resize((size, size), Image.LANCZOS)
    t = transforms.ToTensor()(padded)
    t = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(t)
    return t

@torch.no_grad()
def classify(model, crop_pil, classes):
    t = square_pad(crop_pil).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(t), dim=1)
    idx = probs.argmax().item()
    conf = probs.max().item()
    top5 = [(classes[i], float(probs[0, i])) for i in probs[0].topk(5).indices]
    return classes[idx], conf, top5

# ── SAM → tray → grid → filter → classify ─────────────────
def iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)

def process_image(image_path, mask_gen, model, classes, conf_thresh=0.5):
    img_pil = Image.open(image_path).convert("RGB")
    img_np = np.array(img_pil)
    h, w = img_np.shape[:2]
    img_area = h * w

    # 1. SAM
    masks = mask_gen.generate(img_np)
    print(f"  SAM raw segments: {len(masks)}")

    # 2. Basic food-like filter (area + aspect)
    food_boxes = []
    for m in masks:
        area = m["area"]
        ratio = area / img_area
        if ratio < 0.005 or ratio > 0.40:
            continue
        x, y, bw, bh = m["bbox"]
        if bw < 30 or bh < 30:
            continue
        aspect = bw / bh if bh > 0 else 0
        if aspect < 0.15 or aspect > 6.0:
            continue
        food_boxes.append([x, y, x+bw, y+bh])

    if len(food_boxes) < 3:
        # fallback: use all masks that aren't tiny (very permissive)
        food_boxes = []
        for m in masks:
            x, y, bw, bh = m["bbox"]
            if bw < 20 or bh < 20:
                continue
            food_boxes.append([x, y, x+bw, y+bh])

    print(f"  Food-like boxes: {len(food_boxes)}")

    # 3. Tray = expanded bounding box of all food boxes
    if not food_boxes:
        print("  No food found, skipping")
        return []
    xs = [b[0] for b in food_boxes] + [b[2] for b in food_boxes]
    ys = [b[1] for b in food_boxes] + [b[3] for b in food_boxes]
    tx1, ty1 = min(xs), min(ys)
    tx2, ty2 = max(xs), max(ys)
    # Expand by 5%
    pad_x = int((tx2 - tx1) * 0.05)
    pad_y = int((ty2 - ty1) * 0.05)
    tx1 = max(0, tx1 - pad_x)
    ty1 = max(0, ty1 - pad_y)
    tx2 = min(w-1, tx2 + pad_x)
    ty2 = min(h-1, ty2 + pad_y)
    print(f"  Tray box: [{tx1},{ty1},{tx2},{ty2}]")

    # 4. 2x3 grid inside tray
    cell_w = (tx2 - tx1) / 3
    cell_h = (ty2 - ty1) / 2
    grid_boxes = []
    for r in range(2):
        for c in range(3):
            grid_boxes.append([
                int(tx1 + c*cell_w), int(ty1 + r*cell_h),
                int(tx1 + (c+1)*cell_w), int(ty1 + (r+1)*cell_h)
            ])

    # 5. Keep masks whose centre is inside a grid cell
    kept = []
    for box in food_boxes:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        for g in grid_boxes:
            if g[0] <= cx <= g[2] and g[1] <= cy <= g[3]:
                kept.append(box)
                break
    print(f"  Masks inside grid: {len(kept)}")

    # 6. Classify kept crops
    detections = []
    for box in kept:
        crop = img_pil.crop(box)
        label, conf, top5 = classify(model, crop, classes)
        if conf >= conf_thresh:
            detections.append({"box": box, "label": label, "conf": conf, "top5": top5})
            print(f"    {box} -> {label} ({conf:.2f})")

    # 7. NMS – allow same label if boxes far apart (IoU < 0.1)
    detections.sort(key=lambda d: d["conf"], reverse=True)
    final = []
    for d in detections:
        dup = False
        for k in final:
            if d["label"] == k["label"] and iou(d["box"], k["box"]) > 0.1:
                dup = True
                break
        if not dup:
            final.append(d)
    return final

# ── Visualisation ───────────────────────────────────────────
def draw_output(img_path, dets, out_dir):
    img = cv2.imread(img_path)
    if img is None:
        pil = Image.open(img_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    cols = [(0,255,0),(255,0,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255)]
    for i,d in enumerate(dets):
        x1,y1,x2,y2 = d["box"]
        col = cols[i%len(cols)]
        cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
        txt = f"{d['label']} {d['conf']:.2f}"
        (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1-th-8), (x1+tw+4, y1), col, -1)
        cv2.putText(img, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    stem = Path(img_path).stem
    out_path = os.path.join(out_dir, f"{stem}_detected.jpg")
    cv2.imwrite(out_path, img)
    # JSON
    with open(os.path.join(out_dir, f"{stem}_detections.json"), "w") as f:
        json.dump({"image": stem, "num": len(dets), "detections": dets}, f, indent=2)

# ── Main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Single image")
    group.add_argument("--input_dir", help="Folder of images")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--points_per_side", type=int, default=16)
    args = parser.parse_args()
    
    # Expand user paths
    if args.image:
        args.image = os.path.expanduser(args.image)
    if args.input_dir:
        args.input_dir = os.path.expanduser(args.input_dir)
    args.out_dir = os.path.expanduser(args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    # Load SAM and classifier once
    print("[INFO] Loading SAM...")
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CKPT).to(DEVICE)
    mask_gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=300,
    )
    model, classes = load_classifier()

    if args.image:
        images = [args.image]
    else:
        exts = {'.jpg','.jpeg','.png','.bmp'}
        images = sorted(str(p) for p in Path(args.input_dir).iterdir()
                       if p.suffix.lower() in exts and not p.name.startswith('.'))

    print(f"Processing {len(images)} images...")
    for impath in images:
        print(f"\n--- {Path(impath).name} ---")
        dets = process_image(impath, mask_gen, model, classes, args.conf)
        print(f"  Final detections: {len(dets)}")
        for d in dets:
            print(f"    {d['label']} ({d['conf']:.3f})")
        draw_output(impath, dets, args.out_dir)
        # Save txt
        txt_path = os.path.join(args.out_dir, f"{Path(impath).stem}.txt")
        with open(txt_path, "w") as f:
            for d in dets:
                f.write(f"{d['label']} {d['conf']:.4f} {d['box'][0]} {d['box'][1]} {d['box'][2]} {d['box'][3]}\n")

    print(f"\n[DONE] Results in {args.out_dir}")

if __name__ == "__main__":
    main()