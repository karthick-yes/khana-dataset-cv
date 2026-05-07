#!/usr/bin/env python3
"""
Task 3: Bird's Eye View correction using SAM for EVERYTHING.

Pipeline:
  1. Run SAM on the natural-angle image
  2. Find the tray mask from SAM outputs (largest roughly-rectangular mask)
  3. Extract 4 corners from tray mask → compute homography → warpPerspective → BEV
  4. Run SAM again on the BEV image
  5. Classify each SAM crop with Task 1 classifier (same as Task 2)
  6. Save annotated results + debug images at every stage

Why SAM for tray detection:
  The silver tray on a silver table has nearly zero gradient at the boundary,
  so Canny/adaptive-threshold approaches fail. SAM is class-agnostic and
  segments by visual structure — the tray IS one of the 37 masks SAM
  already produces. We just need to pick the right one.

Usage:
  python task3_bev.py --image ~/data/task3_samples/thali_natural.jpg
  python task3_bev.py --input_dir ~/data/task3_samples/ --output_dir ~/data/task3_out
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image, ImageFile
from torchvision import transforms
from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    from config import get_paths

    paths = get_paths()
    DATA_ROOT = paths["DATA_ROOT"]
    CHECKPOINT_PATH = os.path.expanduser("~/data/checkpoints/best_model.pt")
except Exception:
    DATA_ROOT = os.path.expanduser("~/data/khana")
    CHECKPOINT_PATH = os.path.expanduser("~/data/checkpoints/best_model.pt")

SAM_CHECKPOINT = os.path.expanduser("~/data/sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"
OUTPUT_DIR = os.path.expanduser("~/data/task3_output")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# SHARED UTILS (same as task2)
# ============================================================
def get_class_names(data_root):
    root = Path(data_root)
    if not root.exists():
        print(f"[ERROR] Data root not found: {data_root}")
        sys.exit(1)
    return sorted(d.name for d in root.iterdir() if d.is_dir())


def square_pad_and_normalize(crop_pil, target_size=224):
    w, h = crop_pil.size
    size = max(w, h)
    padded = Image.new("RGB", (size, size), (114, 114, 114))
    padded.paste(crop_pil, ((size - w) // 2, (size - h) // 2))
    resized = padded.resize((target_size, target_size), Image.LANCZOS)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )(resized)


def load_classifier(checkpoint_path, num_classes):
    print("[INFO] Loading classifier...")
    model = timm.create_model(
        "convnext_base.fb_in22k_ft_in1k", pretrained=False, num_classes=num_classes
    )
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE).eval()
    print(f"[INFO] Classifier loaded. Best val acc: {ckpt.get('best_acc', 'N/A')}%")
    return model


def load_sam(checkpoint, points_per_side):
    if not os.path.exists(checkpoint):
        print(f"[ERROR] SAM checkpoint not found: {checkpoint}")
        sys.exit(1)
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError:
        print("[ERROR] segment_anything not installed.")
        sys.exit(1)
    print(f"[INFO] Loading SAM ({SAM_MODEL_TYPE})...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=checkpoint).to(DEVICE)
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        min_mask_region_area=300,
    )
    print("[INFO] SAM loaded.")
    return generator


@torch.no_grad()
def classify_crop(model, crop_pil, class_names):
    tensor = square_pad_and_normalize(crop_pil).unsqueeze(0).to(DEVICE)
    probs = torch.softmax(model(tensor), dim=1)
    idx = probs.argmax(dim=1).item()
    conf = probs.max(dim=1).values.item()
    top5_idx = probs[0].topk(5).indices.cpu().numpy()
    top5 = [(class_names[i], float(probs[0][i])) for i in top5_idx]
    return idx, conf, top5


def compute_iou(b1, b2):
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def remove_duplicates(detections, iou_threshold=0.4, same_label_iou=0.05):
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    kept = []
    for det in detections:
        dup = False
        for k in kept:
            iou = compute_iou(det["box"], k["box"])
            if det["label"] == k["label"] and iou > same_label_iou:
                dup = True
                break
            if det["label"] != k["label"] and iou > iou_threshold:
                dup = True
                break
        if not dup:
            kept.append(det)
    return kept


def save_debug(output_dir, stem, suffix, img_bgr):
    path = os.path.join(output_dir, f"{stem}_{suffix}.jpg")
    cv2.imwrite(path, img_bgr)
    return path


def make_sam_overlay(img_rgb, masks):
    overlay = img_rgb.copy()
    rng = np.random.default_rng(42)
    for m in masks:
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        overlay[m["segmentation"]] = (
            0.5 * overlay[m["segmentation"]] + 0.5 * color
        ).astype(np.uint8)
    return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)


# ============================================================
# STEP 1: FIND TRAY FROM SAM MASKS
# ============================================================
def order_points(pts):
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def find_tray_from_sam_masks(masks, img_shape, debug_bgr=None):
    """
    Among SAM masks, find the one most likely to be the tray.

    Criteria (in order of importance):
      1. Area: 25-80% of image (tray is large but not the whole image)
      2. Aspect ratio: 0.5 to 2.5 (thali trays are roughly rectangular/landscape)
      3. Rectangularity: convex hull area / bounding box area (close to 1 = rectangular)
      4. Solidity: mask area / convex hull area (tray is a solid region)

    Returns (corners_4x2, best_mask) or (None, None)
    """
    h, w = img_shape[:2]
    img_area = h * w

    candidates = []
    for m in masks:
        area = m["area"]
        ratio = area / img_area
        if ratio < 0.20 or ratio > 0.82:
            continue

        seg = m["segmentation"].astype(np.uint8) * 255
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw == 0 or bh == 0:
            continue
        aspect = bw / bh
        if aspect < 0.5 or aspect > 2.8:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        bbox_area = float(bw * bh)
        cnt_area = cv2.contourArea(cnt)

        if hull_area < 1:
            continue

        # rectangularity: how close is the convex hull to filling the bounding box
        rectangularity = hull_area / bbox_area
        # solidity: how filled is the mask (avoids hollow/frame masks)
        solidity = cnt_area / hull_area

        # score: we want high rectangularity, high solidity, reasonable size
        score = rectangularity * solidity * ratio
        candidates.append((score, rectangularity, solidity, ratio, cnt, m))

    if not candidates:
        print("[TRAY] No tray candidate found in SAM masks.")
        return None, None

    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    print(f"[TRAY] Top 3 candidates:")
    for i, (score, rect, sol, rat, _, _) in enumerate(candidates[:3]):
        print(
            f"  #{i + 1}: score={score:.3f} rect={rect:.3f} solid={sol:.3f} area={rat:.1%}"
        )

    best_cnt = candidates[0][4]
    best_mask = candidates[0][5]

    # Get 4 corners from the contour
    peri = cv2.arcLength(best_cnt, True)
    approx = cv2.approxPolyDP(best_cnt, 0.02 * peri, True)

    if len(approx) == 4:
        print(f"[TRAY] Clean quadrilateral found ({len(approx)} vertices)")
        corners = order_points(approx.reshape(4, 2))
    elif len(approx) > 4:
        print(
            f"[TRAY] Rounded shape ({len(approx)} vertices) → using min bounding rect"
        )
        rect = cv2.minAreaRect(best_cnt)
        box = cv2.boxPoints(rect)
        corners = order_points(box)
    else:
        print(f"[TRAY] Too few vertices ({len(approx)}) → using bounding rect")
        bx, by, bw, bh = cv2.boundingRect(best_cnt)
        corners = order_points(
            [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]]
        )

    return corners, best_mask


# ============================================================
# STEP 2: WARP TO BEV
# ============================================================
def warp_to_bev(img_bgr, corners, output_size=900):
    tl, tr, br, bl = corners
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    dst_w = int(max(w_top, w_bot))
    dst_h = int(max(h_left, h_right))

    scale = min(output_size / dst_w, output_size / dst_h)
    dst_w = int(dst_w * scale)
    dst_h = int(dst_h * scale)

    dst_pts = np.array(
        [
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1],
        ],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(corners.astype("float32"), dst_pts)
    warped = cv2.warpPerspective(img_bgr, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
    return warped, M


# ============================================================
# STEP 3: DETECT FOOD ON BEV (same logic as task2 SAM pipeline)
# ============================================================
def detect_food_on_bev(
    bev_bgr,
    sam_generator,
    classifier,
    class_names,
    conf_threshold=0.45,
    crop_padding=8,
    min_area_ratio=0.01,
    max_area_ratio=0.40,
    min_box_size=40,
):
    bev_rgb = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2RGB)
    bev_pil = Image.fromarray(bev_rgb)
    img_w, img_h = bev_pil.size
    img_area = float(img_w * img_h)

    print("[DETECT] Running SAM on BEV...")
    masks = sam_generator.generate(bev_rgb)
    print(f"[DETECT] SAM produced {len(masks)} masks on BEV")

    # filter masks to food-sized regions (same as task2)
    candidate_boxes = []
    for m in masks:
        x, y, w, h = m["bbox"]
        ratio = m["area"] / img_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        if w < min_box_size or h < min_box_size:
            continue
        aspect = w / h if h > 0 else 0
        if aspect < 0.15 or aspect > 6.0:
            continue
        candidate_boxes.append([int(x), int(y), int(x + w), int(y + h)])

    print(f"[DETECT] {len(candidate_boxes)} boxes after area/size filter")

    # merge overlapping boxes
    from itertools import combinations

    def _iou(a, b):
        return compute_iou(a, b)

    pre_nms = []
    for box in candidate_boxes:
        x1, y1, x2, y2 = box
        x1p = max(0, x1 - crop_padding)
        y1p = max(0, y1 - crop_padding)
        x2p = min(img_w, x2 + crop_padding)
        y2p = min(img_h, y2 + crop_padding)
        crop = bev_pil.crop((x1p, y1p, x2p, y2p))

        idx, conf, top5 = classify_crop(classifier, crop, class_names)
        if conf < conf_threshold:
            continue
        pre_nms.append(
            {
                "box": [x1, y1, x2, y2],
                "label": class_names[idx],
                "label_idx": idx,
                "confidence": float(conf),
                "top5": top5,
            }
        )

    print(f"[DETECT] {len(pre_nms)} after confidence filter ({conf_threshold})")
    final = remove_duplicates(pre_nms)
    print(f"[DETECT] {len(final)} after NMS")
    for d in final:
        print(f"  → {d['label']:30s} conf={d['confidence']:.3f}")

    return final, masks


# ============================================================
# MAIN PIPELINE PER IMAGE
# ============================================================
def process_image(
    image_path,
    sam_generator,
    classifier,
    class_names,
    output_dir,
    conf_threshold=0.45,
    bev_size=900,
    points_per_side=16,
    debug=True,
):
    stem = Path(image_path).stem
    print(f"\n{'=' * 60}")
    print(f"Image: {Path(image_path).name}")
    print(f"{'=' * 60}")

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        pil = Image.open(image_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # ── 1. Run SAM on original image ─────────────────────────────────────────
    print("[STEP 1] Running SAM on original image...")
    masks = sam_generator.generate(img_rgb)
    print(f"[STEP 1] {len(masks)} masks generated")

    if debug:
        overlay = make_sam_overlay(img_rgb, masks)
        save_debug(output_dir, stem, "step1_sam_overlay", overlay)

    # ── 2. Find tray from masks ───────────────────────────────────────────────
    print("[STEP 2] Finding tray mask...")
    corners, tray_mask = find_tray_from_sam_masks(masks, img_bgr.shape)

    if corners is None:
        print("[ERROR] Could not find tray. Skipping.")
        return None

    # debug: draw tray corners on original
    if debug:
        dbg = img_bgr.copy()
        pts = corners.astype(int)
        cv2.polylines(dbg, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
        for i, (label, pt) in enumerate(zip(["TL", "TR", "BR", "BL"], pts)):
            cv2.circle(dbg, tuple(pt), 12, (0, 255, 0), -1)
            cv2.putText(
                dbg,
                label,
                (pt[0] + 14, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
        # also highlight the tray mask
        if tray_mask is not None:
            tray_color = np.zeros_like(img_bgr)
            tray_color[tray_mask["segmentation"]] = (0, 180, 0)
            dbg = cv2.addWeighted(dbg, 0.7, tray_color, 0.3, 0)
        save_debug(output_dir, stem, "step2_tray_corners", dbg)

    # ── 3. Warp to BEV ───────────────────────────────────────────────────────
    print("[STEP 3] Warping to BEV...")
    bev_bgr, homography = warp_to_bev(img_bgr, corners, output_size=bev_size)
    bev_path = os.path.join(output_dir, f"{stem}_bev.jpg")
    cv2.imwrite(bev_path, bev_bgr)
    print(f"[STEP 3] BEV saved: {bev_path}  size={bev_bgr.shape[1]}x{bev_bgr.shape[0]}")

    # side-by-side comparison
    if debug:
        oh, ow = img_bgr.shape[:2]
        bh, bw = bev_bgr.shape[:2]
        th = max(oh, bh)
        orig_r = cv2.resize(img_bgr, (int(ow * th / oh), th))
        bev_r = cv2.resize(bev_bgr, (int(bw * th / bh), th))
        sbs = np.hstack([orig_r, bev_r])
        save_debug(output_dir, stem, "step3_comparison", sbs)

    # ── 4. Detect food on BEV ─────────────────────────────────────────────────
    print("[STEP 4] Detecting food on BEV...")
    detections, bev_masks = detect_food_on_bev(
        bev_bgr,
        sam_generator,
        classifier,
        class_names,
        conf_threshold=conf_threshold,
    )

    if debug:
        bev_rgb = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2RGB)
        bev_overlay = make_sam_overlay(bev_rgb, bev_masks)
        save_debug(output_dir, stem, "step4_bev_sam_overlay", bev_overlay)

    # ── 5. Visualise final detections on BEV ─────────────────────────────────
    vis = bev_bgr.copy()
    colors = [
        (0, 255, 0),
        (255, 80, 0),
        (0, 80, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 255, 0),
        (0, 128, 255),
    ]
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["box"]
        c = colors[i % len(colors)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        txt = f"{d['label']} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), c, -1)
        cv2.putText(
            vis, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2
        )

    detected_path = os.path.join(output_dir, f"{stem}_detected.jpg")
    cv2.imwrite(detected_path, vis)
    print(f"[STEP 5] Final detection saved: {detected_path}")

    # ── 6. Save JSON ──────────────────────────────────────────────────────────
    payload = {
        "image": stem,
        "num_detections": len(detections),
        "tray_corners": corners.tolist(),
        "bev_path": bev_path,
        "detected_path": detected_path,
        "detections": detections,
    }
    json_path = os.path.join(output_dir, f"{stem}_detections.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    return payload


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--sam_checkpoint", type=str, default=SAM_CHECKPOINT)
    parser.add_argument("--sam_points_per_side", type=int, default=16)
    parser.add_argument("--conf_threshold", type=float, default=0.45)
    parser.add_argument("--bev_size", type=int, default=900)
    parser.add_argument("--no_debug", action="store_true")
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    args.checkpoint = os.path.expanduser(args.checkpoint)
    args.sam_checkpoint = os.path.expanduser(args.sam_checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        image_paths = [os.path.expanduser(args.image)]
    elif args.input_dir:
        d = os.path.expanduser(args.input_dir)
        image_paths = sorted(
            str(p)
            for p in Path(d).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and not p.name.startswith(".")
        )
    else:
        parser.print_help()
        sys.exit(0)

    if not image_paths:
        print("[ERROR] No images found.")
        sys.exit(1)

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Processing {len(image_paths)} image(s)")

    class_names = get_class_names(DATA_ROOT)
    classifier = load_classifier(args.checkpoint, num_classes=len(class_names))
    sam_generator = load_sam(args.sam_checkpoint, args.sam_points_per_side)

    all_results = []
    for img_path in image_paths:
        result = process_image(
            img_path,
            sam_generator=sam_generator,
            classifier=classifier,
            class_names=class_names,
            output_dir=args.output_dir,
            conf_threshold=args.conf_threshold,
            bev_size=args.bev_size,
            debug=(not args.no_debug),
        )
        if result:
            all_results.append(result)

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done. {len(all_results)} images processed.")
    print(f"Results in: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
