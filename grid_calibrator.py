#!/usr/bin/env python3
"""
2x3 grid calibration – click the 4 tray corners (any order),
the script computes the axis‑aligned bounding box and overlays a 2x3 grid.
"""

import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

IMAGE_PATH = (
    "/home/karthiksunil/data/task2_images/Plate 113.jpg"  # change to your image
)
ROWS, COLS = 2, 3


def main():
    img = Image.open(IMAGE_PATH).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    print(f"Image size: {w}x{h}")

    # Click 4 corners of the tray
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_np)
    ax.set_title("Click the 4 tray corners (any order), then press Enter")
    pts = plt.ginput(n=4, timeout=0)
    plt.close(fig)
    if len(pts) < 4:
        print("Need 4 points. Exiting.")
        sys.exit(1)

    pts = np.array(pts).astype(np.int32)

    # Bounding rect of the clicked points
    x1 = pts[:, 0].min()
    y1 = pts[:, 1].min()
    x2 = pts[:, 0].max()
    y2 = pts[:, 1].max()

    # Clip to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    tray_w = x2 - x1
    tray_h = y2 - y1
    cell_w = tray_w / COLS
    cell_h = tray_h / ROWS

    grid_boxes = []
    for row in range(ROWS):
        for col in range(COLS):
            bx1 = int(x1 + col * cell_w)
            by1 = int(y1 + row * cell_h)
            bx2 = int(x1 + (col + 1) * cell_w)
            by2 = int(y1 + (row + 1) * cell_h)
            grid_boxes.append([bx1, by1, bx2, by2])

    print("\nGrid boxes (x1,y1,x2,y2):")
    for i, box in enumerate(grid_boxes):
        print(f"Cell {i + 1}: {box}")

    # Draw on image
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    for box in grid_boxes:
        cv2.rectangle(img_bgr, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(img_bgr, tuple(pt), 10, (0, 0, 255), -1)
    cv2.imwrite("grid_debug.jpg", img_bgr)
    print("Debug image saved as 'grid_debug.jpg'")


if __name__ == "__main__":
    main()
