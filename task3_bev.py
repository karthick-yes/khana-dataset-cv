#!/usr/bin/env python3
"""
Task 3: Bird's Eye View (BEV) correction for natural-angle thali images.

Pipeline:
  1. Auto-detect the 4 corners of the thali tray
       → large quadrilateral contour in the image
  2. If auto fails (or --manual flag), let the user click 4 corners
  3. Compute homography → warpPerspective → BEV image
  4. Run Task 2 detect_thali() on the BEV image
  5. Save annotated result

Usage:
  # Fully automatic, then detect food:
  python task3_bev.py --image ~/data/task3_samples/thali_natural.jpg

  # Force manual corner selection:
  python task3_bev.py --image ~/data/task3_samples/thali_natural.jpg --manual

  # BEV only, skip food detection:
  python task3_bev.py --image ~/data/task3_samples/thali_natural.jpg --no_detect

  # Batch a whole folder:
  python task3_bev.py --input_dir ~/data/task3_samples/
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── optional: reuse Task 2 detection ──────────────────────────────────────────
try:
    import torch
    import timm
    from ultralytics import YOLO
    from torchvision import transforms

    HAS_DETECT = True
except ImportError:
    HAS_DETECT = False

try:
    from config import get_paths

    _paths = get_paths()
    DATA_ROOT = _paths["DATA_ROOT"]
    CHECKPOINT_DIR = _paths["CHECKPOINT_DIR"]
except Exception:
    DATA_ROOT = os.path.expanduser("~/data/khana")
    CHECKPOINT_DIR = os.path.expanduser("~/data/checkpoints")

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
OUTPUT_DIR = os.path.expanduser("~/data/task3_output")
DEVICE = "cpu"  # safe default; change to "cuda" on lab machine


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — ORDER CORNERS (top-left, top-right, bottom-right, bottom-left)
# ══════════════════════════════════════════════════════════════════════════════
def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sort 4 points into [TL, TR, BR, BL] order.
    Works regardless of how the user clicked or how the contour returned them.
    """
    pts = pts.reshape(4, 2).astype("float32")
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL — smallest x+y
    rect[2] = pts[np.argmax(s)]  # BR — largest  x+y

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR — smallest y-x
    rect[3] = pts[np.argmax(diff)]  # BL — largest  y-x

    return rect


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2A — AUTO corner detection
# ══════════════════════════════════════════════════════════════════════════════
def auto_detect_corners(img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Finds the thali tray as the largest quadrilateral-ish contour.

    Strategy
    --------
    - Convert to grayscale → blur → Canny
    - Find all contours, keep the largest by area
    - Approximate to a polygon; if it has 4 sides → done
    - If it has more sides (rounded tray), fit a min-area rectangle
    - Returns None if nothing plausible is found

    Why this works
    --------------
    The tray is the dominant rectangular/elliptical shape in a thali photo.
    It typically covers >30% of the image area, so it wins the area sort.
    Rounded trays don't give a clean 4-point contour, so we fall back to
    the minimum bounding rectangle of the largest contour — that gives us
    4 tight corners even for circular or slightly curved trays.
    """
    h, w = img_bgr.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive thresholding handles shadows well; Canny catches the strong tray edge
    edges = cv2.Canny(blurred, 30, 120)

    # Dilate to connect broken edges (common where food sits against the tray wall)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Sort by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours[:5]:  # only inspect the 5 largest
        area = cv2.contourArea(cnt)
        if area < img_area * 0.15:  # tray must cover at least 15% of image
            continue

        # Try to get a tight polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            print(f"[AUTO] Found quadrilateral contour (area={area / img_area:.0%})")
            return order_points(approx.reshape(4, 2))

        # Rounded tray — fall back to minimum bounding rectangle
        if len(approx) > 4:
            print(
                f"[AUTO] Rounded tray ({len(approx)} vertices) → using min bounding rect"
            )
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)  # 4 floating-point corners
            return order_points(box)

    print("[AUTO] No suitable quadrilateral found")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2B — MANUAL corner selection (interactive OpenCV window)
# ══════════════════════════════════════════════════════════════════════════════
_clicked_points: list[tuple[int, int]] = []


def _mouse_callback(event, x, y, flags, param):
    global _clicked_points
    if event == cv2.EVENT_LBUTTONDOWN and len(_clicked_points) < 4:
        _clicked_points.append((x, y))
        print(f"  Point {len(_clicked_points)}: ({x}, {y})")


def manual_select_corners(img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Opens a window; user clicks 4 corners of the thali tray in any order.
    Order: TL → TR → BR → BL is suggested but order_points() will fix it.
    Press 'q' to cancel.
    """
    global _clicked_points
    _clicked_points = []

    display = img_bgr.copy()
    h, w = display.shape[:2]
    scale = min(1.0, 1200 / max(h, w))  # fit in 1200px window
    disp_w = int(w * scale)
    disp_h = int(h * scale)
    display = cv2.resize(display, (disp_w, disp_h))

    cv2.namedWindow("Select 4 corners — TL TR BR BL — then press ENTER")
    cv2.setMouseCallback(
        "Select 4 corners — TL TR BR BL — then press ENTER", _mouse_callback
    )

    print("\n[MANUAL] Click the 4 corners of the thali tray.")
    print("         Suggested order: Top-Left → Top-Right → Bottom-Right → Bottom-Left")
    print("         Press ENTER when done, 'r' to reset, 'q' to cancel.\n")

    while True:
        vis = display.copy()
        for i, (px, py) in enumerate(_clicked_points):
            cv2.circle(vis, (px, py), 8, (0, 255, 0), -1)
            cv2.putText(
                vis,
                str(i + 1),
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
        if len(_clicked_points) == 4:
            pts = np.array(_clicked_points, dtype="float32")
            cv2.polylines(
                vis, [pts.astype(int).reshape(-1, 1, 2)], True, (0, 255, 0), 2
            )
        cv2.imshow("Select 4 corners — TL TR BR BL — then press ENTER", vis)
        key = cv2.waitKey(20) & 0xFF
        if key == 13 and len(_clicked_points) == 4:  # ENTER
            break
        if key == ord("r"):
            _clicked_points = []
        if key == ord("q"):
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    # Scale points back to original image coords
    pts = np.array(_clicked_points, dtype="float32") / scale
    return order_points(pts)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — PERSPECTIVE WARP → BEV
# ══════════════════════════════════════════════════════════════════════════════
def warp_to_bev(
    img_bgr: np.ndarray, corners: np.ndarray, output_size: int = 800
) -> np.ndarray:
    """
    Given the 4 ordered corners of the tray, warp to a square top-down view.

    We compute the actual tray width/height from the corners so the aspect
    ratio of the destination rectangle is as close to reality as possible,
    then scale it to fit within output_size × output_size.

    corners: [TL, TR, BR, BL] in image pixel coordinates
    """
    tl, tr, br, bl = corners

    # Width = average of top edge and bottom edge lengths
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    dst_w = int(max(w_top, w_bot))

    # Height = average of left edge and right edge lengths
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    dst_h = int(max(h_left, h_right))

    # Scale to fit inside output_size while preserving aspect ratio
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

    M = cv2.getPerspectiveTransform(corners, dst_pts)
    warped = cv2.warpPerspective(img_bgr, M, (dst_w, dst_h), flags=cv2.INTER_LANCZOS4)
    return warped


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — OPTIONAL TASK 2 DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def run_detection_on_bev(
    bev_bgr: np.ndarray, checkpoint: str, output_dir: str, stem: str
) -> list[dict]:
    """
    Saves the BEV as a temp JPEG and pipes it through detect_thali().
    Re-uses exactly the same detection code as Task 2.
    Returns the list of detection dicts.
    """
    if not HAS_DETECT:
        print("[DETECT] torch/timm/ultralytics not available — skipping detection")
        return []

    import json
    import timm as _timm
    import torch as _torch

    # ── lazy import of shared task2 helpers ──────────────────────────────────
    # We replicate the minimal parts here so task3 works as a standalone file.
    # If you prefer, replace this block with:
    #   from task2_detect import detect_thali, visualize_detections
    # ─────────────────────────────────────────────────────────────────────────
    from torchvision import transforms as T

    def _get_class_names(root):
        return sorted([d.name for d in Path(root).iterdir() if d.is_dir()])

    def _square_pad(crop_pil, size=224):
        w, h = crop_pil.size
        s = max(w, h)
        padded = Image.new("RGB", (s, s), (114, 114, 114))
        padded.paste(crop_pil, ((s - w) // 2, (s - h) // 2))
        resized = padded.resize((size, size), Image.LANCZOS)
        return T.Compose(
            [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )(resized)

    @_torch.no_grad()
    def _classify(model, crop_pil, names):
        t = _square_pad(crop_pil).unsqueeze(0).to(DEVICE)
        probs = _torch.softmax(model(t), dim=1)
        idx = probs.argmax(dim=1).item()
        conf = probs.max(dim=1).values.item()
        top5_idx = probs[0].topk(5).indices.cpu().numpy()
        top5 = [(names[i], float(probs[0][i])) for i in top5_idx]
        return idx, conf, top5

    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aa + ab - inter + 1e-6)

    # ── load models ───────────────────────────────────────────────────────────
    if not os.path.exists(DATA_ROOT):
        print(f"[DETECT] Data root not found: {DATA_ROOT} — skipping detection")
        return []
    class_names = _get_class_names(DATA_ROOT)

    if not os.path.exists(checkpoint):
        print(f"[DETECT] Checkpoint not found: {checkpoint} — skipping detection")
        return []

    print("[DETECT] Loading classifier…")
    clf = _timm.create_model(
        "convnext_base.fb_in22k_ft_in1k", pretrained=False, num_classes=len(class_names)
    )
    ckpt = _torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    clf.load_state_dict(ckpt["model_state_dict"])
    clf = clf.to(DEVICE).eval()

    print("[DETECT] Loading YOLOv8x…")
    yolo = YOLO("yolov8x.pt")

    # ── save BEV to disk so YOLO can read it ─────────────────────────────────
    tmp_path = os.path.join(output_dir, f"{stem}_bev_tmp.jpg")
    cv2.imwrite(tmp_path, bev_bgr)

    bev_pil = Image.fromarray(cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2RGB))
    img_w, img_h = bev_pil.size

    results = yolo(tmp_path, conf=0.10, iou=0.4, verbose=False)
    boxes_raw = results[0].boxes.xyxy.cpu().numpy()

    CONF_THRESH = 0.50
    raw = []
    for box in boxes_raw:
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        if w < 50 or h < 50:
            continue
        if (w * h) / (img_w * img_h) > 0.45:
            continue
        crop = bev_pil.crop(
            (max(0, x1 - 8), max(0, y1 - 8), min(img_w, x2 + 8), min(img_h, y2 + 8))
        )
        idx, conf, top5 = _classify(clf, crop, class_names)
        if conf < CONF_THRESH:
            continue
        raw.append(
            {
                "box": [x1, y1, x2, y2],
                "label": class_names[idx],
                "confidence": float(conf),
                "top5": top5,
            }
        )

    # NMS — keep highest-conf prediction per class (same logic as task2)
    raw.sort(key=lambda d: d["confidence"], reverse=True)
    kept = []
    for det in raw:
        dup = False
        for k in kept:
            if det["label"] == k["label"] or _iou(det["box"], k["box"]) > 0.4:
                dup = True
                break
        if not dup:
            kept.append(det)

    print(f"[DETECT] {len(kept)} items found in BEV:")
    for d in kept:
        print(f"  {d['label']:30s}  conf={d['confidence']:.3f}")

    # ── annotate BEV image ───────────────────────────────────────────────────
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
    for i, d in enumerate(kept):
        x1, y1, x2, y2 = d["box"]
        c = colors[i % len(colors)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        txt = f"{d['label']} {d['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), c, -1)
        cv2.putText(
            vis, txt, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2
        )

    det_path = os.path.join(output_dir, f"{stem}_bev_detected.jpg")
    cv2.imwrite(det_path, vis)
    print(f"[DETECT] Annotated BEV saved: {det_path}")

    # JSON
    json_path = os.path.join(output_dir, f"{stem}_detections.json")
    import json

    with open(json_path, "w") as f:
        json.dump({"image": stem, "num": len(kept), "detections": kept}, f, indent=2)

    os.remove(tmp_path)  # clean up temp file
    return kept


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def process_image(
    image_path: str,
    output_dir: str,
    manual: bool = False,
    no_detect: bool = False,
    checkpoint: str = CHECKPOINT_PATH,
    bev_size: int = 800,
) -> dict:

    stem = Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Image : {Path(image_path).name}")
    print(f"{'=' * 60}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        pil = Image.open(image_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # ── 1. Find tray corners ─────────────────────────────────────────────────
    corners = None

    if not manual:
        corners = auto_detect_corners(img_bgr)

    if corners is None:
        print("[INFO] Falling back to manual corner selection…")
        corners = manual_select_corners(img_bgr)

    if corners is None:
        print("[ERROR] No corners selected — skipping this image")
        return {}

    print(f"[INFO] Corners (TL TR BR BL):")
    for label, pt in zip(["TL", "TR", "BR", "BL"], corners):
        print(f"         {label}: ({pt[0]:.0f}, {pt[1]:.0f})")

    # ── 2. Save debug image with corner overlay ───────────────────────────────
    dbg = img_bgr.copy()
    pts_int = corners.astype(int)
    cv2.polylines(dbg, [pts_int.reshape(-1, 1, 2)], True, (0, 255, 0), 3)
    for i, (label, pt) in enumerate(zip(["TL", "TR", "BR", "BL"], pts_int)):
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
    corners_path = os.path.join(output_dir, f"{stem}_corners.jpg")
    cv2.imwrite(corners_path, dbg)
    print(f"[INFO] Corner overlay saved: {corners_path}")

    # ── 3. Warp ───────────────────────────────────────────────────────────────
    bev_bgr = warp_to_bev(img_bgr, corners, output_size=bev_size)
    bev_path = os.path.join(output_dir, f"{stem}_bev.jpg")
    cv2.imwrite(bev_path, bev_bgr)
    print(f"[INFO] BEV saved: {bev_path}")

    # ── 4. Side-by-side comparison ────────────────────────────────────────────
    orig_h, orig_w = img_bgr.shape[:2]
    bev_h, bev_w = bev_bgr.shape[:2]
    target_h = max(orig_h, bev_h)
    orig_resized = cv2.resize(img_bgr, (int(orig_w * target_h / orig_h), target_h))
    bev_resized = cv2.resize(bev_bgr, (int(bev_w * target_h / bev_h), target_h))
    side_by_side = np.hstack([orig_resized, bev_resized])
    sbs_path = os.path.join(output_dir, f"{stem}_comparison.jpg")
    cv2.imwrite(sbs_path, side_by_side)
    print(f"[INFO] Comparison saved: {sbs_path}")

    # ── 5. Detection on BEV ───────────────────────────────────────────────────
    detections = []
    if not no_detect:
        detections = run_detection_on_bev(bev_bgr, checkpoint, output_dir, stem)

    return {
        "image": Path(image_path).name,
        "corners": corners.tolist(),
        "bev_path": bev_path,
        "detections": detections,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Task 3: Perspective correction → BEV → food detection"
    )

    parser.add_argument("--image", type=str, default=None, help="Single image path")
    parser.add_argument(
        "--input_dir", type=str, default=None, help="Folder of images (batch mode)"
    )
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument(
        "--bev_size",
        type=int,
        default=800,
        help="Max dimension of output BEV image (default 800)",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Skip auto-detection; click corners interactively",
    )
    parser.add_argument(
        "--no_detect", action="store_true", help="Only do BEV warp, skip food detection"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        image_paths = [args.image]
    elif args.input_dir:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(
            [
                str(p)
                for p in Path(args.input_dir).iterdir()
                if p.suffix.lower() in exts and not p.name.startswith(".")
            ]
        )
        if not image_paths:
            print(f"[ERROR] No images found in {args.input_dir}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)

    print(f"[INFO] Processing {len(image_paths)} image(s)")
    print(f"[INFO] Output: {args.output_dir}")

    import json

    all_results = []
    for path in image_paths:
        result = process_image(
            path,
            args.output_dir,
            manual=args.manual,
            no_detect=args.no_detect,
            checkpoint=args.checkpoint,
            bev_size=args.bev_size,
        )
        all_results.append(result)

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] Summary: {summary_path}")


if __name__ == "__main__":
    main()
