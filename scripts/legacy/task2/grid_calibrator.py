#!/usr/bin/env python3
"""
Robust 2x3 tray calibration – click 4 tray corners (any order),
the script uses the axis‑aligned bounding box and divides it into a 2x3 grid.
Saves the relative tray area as a JSON file for later use in Task 2.
"""
import argparse
import os
import sys, json, cv2, numpy as np, matplotlib.pyplot as plt
from PIL import Image

DEFAULT_IMAGE_PATH = "/home/karthiksunil/data/task2_images/Plate 146.jpg"
ROWS, COLS = 2, 3
DEFAULT_OUT_JSON = "tray_calibration.json"

def main(image_path, out_json):
    image_path = os.path.expanduser(image_path)
    out_json = os.path.expanduser(out_json)

    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    print(f"Image size: {w}x{h}")

    # --- Click 4 corners of the tray (outer boundary) ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_np)
    ax.set_title("Click the 4 outer corners of the tray (any order), press Enter")
    pts = plt.ginput(n=4, timeout=0)
    plt.close(fig)

    if len(pts) < 4:
        print("Need 4 points. Exiting.")
        sys.exit(1)

    pts = np.array(pts).astype(np.int32)

    # --- Bounding rect of the clicked points ---
    x1 = pts[:, 0].min()
    y1 = pts[:, 1].min()
    x2 = pts[:, 0].max()
    y2 = pts[:, 1].max()

    # Clip to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)

    # --- Compute relative coordinates (0..1 fractions) ---
    rel_x1 = x1 / w
    rel_y1 = y1 / h
    rel_x2 = x2 / w
    rel_y2 = y2 / h

    print(f"Tray bounding box (pixels): [{x1}, {y1}, {x2}, {y2}]")
    print(f"Relative tray: [x1={rel_x1:.4f}, y1={rel_y1:.4f}, x2={rel_x2:.4f}, y2={rel_y2:.4f}]")

    # --- Generate 2x3 grid inside that box (for visual check) ---
    tray_w = x2 - x1
    tray_h = y2 - y1
    cell_w = tray_w / COLS
    cell_h = tray_h / ROWS

    grid_boxes = []
    for row in range(ROWS):
        for col in range(COLS):
            bx1 = int(x1 + col * cell_w)
            by1 = int(y1 + row * cell_h)
            bx2 = int(x1 + (col+1) * cell_w)
            by2 = int(y1 + (row+1) * cell_h)
            grid_boxes.append([bx1, by1, bx2, by2])

    # --- Draw on image and save ---
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    for box in grid_boxes:
        cv2.rectangle(img_bgr, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
    for pt in pts:
        cv2.circle(img_bgr, tuple(pt), 10, (0,0,255), -1)
    cv2.imwrite("grid_calib_debug.jpg", img_bgr)
    print("Debug image saved: grid_calib_debug.jpg")

    # --- Save relative tray coords to JSON ---
    calib = {
        "image_width": w,
        "image_height": h,
        "rel_x1": rel_x1,
        "rel_y1": rel_y1,
        "rel_x2": rel_x2,
        "rel_y2": rel_y2,
        "description": "Relative coordinates of the tray bounding box (top‑left, bottom‑right)"
    }
    with open(out_json, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Calibration saved to {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH, help="Image used for tray corner clicks")
    parser.add_argument("--out_json", default=DEFAULT_OUT_JSON, help="Output calibration JSON path")
    args = parser.parse_args()
    main(args.image, args.out_json)
