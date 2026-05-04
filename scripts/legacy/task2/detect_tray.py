#!/usr/bin/env python3
"""
Tray boundary detector — validation script.

Tries multiple strategies and saves a debug image per input so you can
quickly see which approach works on your dataset.

Usage:
    python detect_tray.py --input_dir ~/data/task2_images --output_dir ~/data/tray_debug
    python detect_tray.py --single_image ~/data/task2_images/Plate_177.jpg --output_dir ~/data/tray_debug
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# STRATEGY 1: Largest contour by area (your original idea)
# Likely to fail on silver-on-silver but included as baseline
# ============================================================
def strategy_largest_contour(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 20, 80)
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.15 * img_area or area > 0.80 * img_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h
        if aspect < 0.7 or aspect > 2.8:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, x + w, y + h)
    return best


# ============================================================
# STRATEGY 2: Adaptive threshold — better for low-contrast scenes
# The compartment dividers create strong local contrast even if
# overall brightness is similar to the background
# ============================================================
def strategy_adaptive_thresh(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # adaptive threshold picks up divider lines even on silver bg
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 4
    )
    kernel = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.15 * img_area or area > 0.85 * img_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h
        if aspect < 0.7 or aspect > 2.8:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, x + w, y + h)
    return best


# ============================================================
# STRATEGY 3: Saturation channel — food has colour, metal doesn't.
# The tray bounding box roughly = bounding box of all coloured regions.
# Works well when there's visible food. Raita/white items may not help
# but dal/curry/fruits will.
# ============================================================
def strategy_saturation(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]  # saturation channel

    # threshold: keep pixels with meaningful colour
    _, sat_mask = cv2.threshold(sat, 40, 255, cv2.THRESH_BINARY)

    kernel = np.ones((15, 15), np.uint8)
    closed = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    best, best_area = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.10 * img_area or area > 0.85 * img_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h
        if aspect < 0.6 or aspect > 3.0:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, x + w, y + h)
    return best


# ============================================================
# STRATEGY 4: Convex hull of all coloured regions combined
# More robust than single contour — handles fragmented food regions
# ============================================================
def strategy_convex_hull_of_color(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, sat_mask = cv2.threshold(sat, 35, 255, cv2.THRESH_BINARY)

    kernel = np.ones((11, 11), np.uint8)
    cleaned = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    # gather all contour points from reasonably sized blobs
    all_points = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 0.005 * img_area:
            all_points.append(cnt)

    if not all_points:
        return None

    combined = np.vstack(all_points)
    hull = cv2.convexHull(combined)
    x, y, w, h = cv2.boundingRect(hull)

    # add a small margin
    pad = 20
    img_h, img_w = img_bgr.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_w, x + w + pad)
    y2 = min(img_h, y + h + pad)

    aspect = (x2 - x1) / (y2 - y1)
    if aspect < 0.5 or aspect > 3.5:
        return None

    return (x1, y1, x2, y2)


# ============================================================
# DEBUG VISUALISER
# Draws all 4 strategy results side-by-side so you can compare
# ============================================================
def draw_result(img_bgr, box, label, color=(0, 255, 0)):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    # scale down for display
    scale = 640 / max(h, w)
    out = cv2.resize(out, (int(w * scale), int(h * scale)))
    if box is not None:
        sx = scale
        sy = scale
        x1, y1, x2, y2 = (
            int(box[0] * sx),
            int(box[1] * sy),
            int(box[2] * sx),
            int(box[3] * sy),
        )
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        success = "OK"
    else:
        success = "FAILED"
        color = (0, 0, 255)

    cv2.putText(
        out, f"{label}: {success}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
    )
    return out


def process_image(image_path, output_dir):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        pil = Image.open(image_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    stem = Path(image_path).stem

    strategies = [
        ("1_largest_contour", strategy_largest_contour),
        ("2_adaptive_thresh", strategy_adaptive_thresh),
        ("3_saturation", strategy_saturation),
        ("4_convex_hull_of_color", strategy_convex_hull_of_color),
    ]

    panels = []
    results = {}
    for name, fn in strategies:
        box = fn(img_bgr)
        results[name] = box
        panel = draw_result(img_bgr, box, name)
        panels.append(panel)
        status = f"box={box}" if box else "NO BOX"
        print(f"  [{name}] {status}")

    # stack 2x2 grid
    row1 = np.hstack(panels[:2])
    row2 = np.hstack(panels[2:])

    # make sure rows are same width (they should be, but just in case)
    if row1.shape[1] != row2.shape[1]:
        w = min(row1.shape[1], row2.shape[1])
        row1 = row1[:, :w]
        row2 = row2[:, :w]

    grid = np.vstack([row1, row2])

    out_path = os.path.join(output_dir, f"{stem}_tray_strategies.jpg")
    cv2.imwrite(out_path, grid)
    print(f"  Saved: {out_path}")

    # also save individual crops for the best looking strategy
    # (saturation and convex hull are most likely to work here)
    for name, fn in strategies:
        box = results[name]
        if box is not None:
            x1, y1, x2, y2 = box
            crop = img_bgr[y1:y2, x1:x2]
            crop_path = os.path.join(output_dir, f"{stem}_crop_{name}.jpg")
            cv2.imwrite(crop_path, crop)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--single_image", type=str, default=None)
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.single_image:
        image_paths = [os.path.expanduser(args.single_image)]
    elif args.input_dir:
        input_dir = os.path.expanduser(args.input_dir)
        image_paths = sorted(
            str(p)
            for p in Path(input_dir).iterdir()
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and not p.name.startswith(".")
        )
    else:
        print("[ERROR] Provide --input_dir or --single_image")
        sys.exit(1)

    if not image_paths:
        print("[ERROR] No images found.")
        sys.exit(1)

    print(f"Processing {len(image_paths)} images → {args.output_dir}\n")

    for img_path in image_paths:
        print(f"{'=' * 50}")
        print(f"Image: {Path(img_path).name}")
        process_image(img_path, args.output_dir)

    print(f"\nDone. Check {args.output_dir} for *_tray_strategies.jpg files.")
    print("Each image shows 4 strategies in a 2x2 grid — green=found, red=failed.")


if __name__ == "__main__":
    main()
