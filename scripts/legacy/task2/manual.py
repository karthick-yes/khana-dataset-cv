#!/usr/bin/env python3
"""
Manually draw bounding box for each compartment (2 clicks per box, 6 compartments).
"""

import sys, numpy as np, cv2, matplotlib.pyplot as plt
from PIL import Image

IMAGE_PATH = "/home/karthiksunil/data/task2_images/Plate 113.jpg"
N = 6  # number of compartments (2x3)


def main():
    img = Image.open(IMAGE_PATH).convert("RGB")
    img_np = np.array(img)
    h, w = img_np.shape[:2]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img_np)
    ax.set_title(
        f"Click top‑left then bottom‑right for each of {N} compartments. Press Enter after last pair."
    )
    pts = plt.ginput(n=N * 2, timeout=0)
    plt.close(fig)
    if len(pts) < N * 2:
        print(f"Need exactly {N * 2} points. Exiting.")
        sys.exit(1)

    grid_boxes = []
    for i in range(N):
        tl = pts[2 * i]
        br = pts[2 * i + 1]
        x1, y1 = int(round(tl[0])), int(round(tl[1]))
        x2, y2 = int(round(br[0])), int(round(br[1]))
        # Ensure correct order
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        grid_boxes.append([x1, y1, x2, y2])

    print("\nManual compartment boxes:")
    for i, box in enumerate(grid_boxes):
        print(f"Compartment {i + 1}: {box}")

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    for box in grid_boxes:
        cv2.rectangle(img_bgr, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imwrite("manual_grid.jpg", img_bgr)
    print("Debug image saved as 'manual_grid.jpg'")


if __name__ == "__main__":
    main()
