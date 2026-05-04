#!/usr/bin/env python3
"""
Two tools in one file:

1. sample_dataset.py behavior  -- run with: python diagnose_task2.py --mode sample
   Randomly samples images from the Khana dataset and saves them as viewable PNGs

2. Task 2 diagnostic           -- run with: python diagnose_task2.py --mode diagnose --image path/to/thali.jpg
   Shows exactly what YOLO proposes and what the classifier thinks for each crop
   Saves a grid of all crops with their top5 predictions so you can see what's failing
"""
import os
import sys
import argparse
import random
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image, ImageFile, ImageDraw, ImageFont
from torchvision import transforms
import timm
from ultralytics import YOLO

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from config import get_paths

paths = get_paths()
DATA_ROOT = paths["DATA_ROOT"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = os.path.expanduser("~/data/checkpoints/best_model.pt")

CLASSIFIER_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_class_names(data_root):
    return sorted([d.name for d in Path(data_root).iterdir() if d.is_dir()])

# ============================================================
# TOOL 1: DATASET SAMPLER
# ============================================================
def sample_dataset(data_root, output_dir, n_per_class=3, classes=None):
    """
    Sample random images from the dataset and save as viewable PNGs.
    Since Khana images have no extensions, PIL loads them and we save as PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    root = Path(data_root)
    all_classes = sorted([d.name for d in root.iterdir() if d.is_dir()])

    # If specific classes requested, filter
    if classes:
        all_classes = [c for c in all_classes if c in classes]

    print(f"[INFO] Sampling {n_per_class} images from each of {len(all_classes)} classes")
    print(f"[INFO] Output: {output_dir}")

    saved = 0
    failed = 0

    for cls in all_classes:
        cls_dir = root / cls
        files = [f for f in cls_dir.iterdir() if f.is_file() and not f.name.startswith('.')]

        if not files:
            continue

        # Sample randomly
        sampled = random.sample(files, min(n_per_class, len(files)))

        cls_out = output_dir / cls.replace(' ', '_')
        cls_out.mkdir(exist_ok=True)

        for f in sampled:
            try:
                img = Image.open(f).convert('RGB')
                out_path = cls_out / f"{f.name}.png"
                img.save(out_path)
                saved += 1
            except Exception as e:
                print(f"[WARN] Failed {f}: {e}")
                failed += 1

    print(f"\n[DONE] Saved {saved} images, {failed} failed")
    print(f"[INFO] Open {output_dir} in your file manager to view")

# ============================================================
# TOOL 2: TASK 2 DIAGNOSTIC
# ============================================================
def load_task1_model(checkpoint_path, num_classes):
    model = timm.create_model(
        'convnext_base.fb_in22k_ft_in1k',
        pretrained=False,
        num_classes=num_classes
    )
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    return model

@torch.no_grad()
def classify_crop_full(model, crop_pil, class_names, topk=5):
    tensor = CLASSIFIER_TRANSFORM(crop_pil).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    topk_vals, topk_idx = probs[0].topk(topk)
    results = [(class_names[i], float(v)) for i, v in zip(topk_idx.cpu().numpy(), topk_vals.cpu().numpy())]
    return results

def make_crop_grid(crops_and_preds, output_path, cols=4):
    """
    Save a visual grid showing each YOLO crop and its top5 predictions.
    This is your main diagnostic tool — you can see exactly what the model thinks.
    """
    if not crops_and_preds:
        print("[WARN] No crops to visualize")
        return

    cell_w, cell_h = 300, 380
    rows = (len(crops_and_preds) + cols - 1) // cols
    grid_w = cols * cell_w
    grid_h = rows * cell_h

    grid = Image.new('RGB', (grid_w, grid_h), (240, 240, 240))

    for idx, (crop_pil, box, preds, yolo_conf) in enumerate(crops_and_preds):
        row = idx // cols
        col = idx % cols
        x_off = col * cell_w
        y_off = row * cell_h

        # Resize crop to fit cell
        crop_resized = crop_pil.resize((cell_w, 200), Image.LANCZOS)
        grid.paste(crop_resized, (x_off, y_off))

        # Draw predictions as text below crop
        draw = ImageDraw.Draw(grid)

        # Box coords
        draw.text((x_off + 4, y_off + 202),
                  f"Box: {box} | YOLO:{yolo_conf:.2f}",
                  fill=(80, 80, 80))

        # Top5
        for rank, (label, prob) in enumerate(preds):
            color = (0, 150, 0) if rank == 0 else (100, 100, 100)
            bar_w = int(prob * (cell_w - 10))
            draw.rectangle([x_off + 4, y_off + 220 + rank*30,
                           x_off + 4 + bar_w, y_off + 244 + rank*30],
                          fill=color)
            draw.text((x_off + 4, y_off + 222 + rank*30),
                     f"{rank+1}. {label[:22]} {prob:.2%}",
                     fill=(255, 255, 255))

    grid.save(output_path)
    print(f"[INFO] Diagnostic grid saved: {output_path}")

def diagnose_image(image_path, classifier_model, class_names, output_dir,
                   yolo_conf=0.10, conf_threshold=0.0, min_box_size=30):
    """
    Full diagnostic on one thali image.
    Shows ALL YOLO proposals regardless of confidence so you can see what's being missed.
    conf_threshold=0.0 means show everything, not just confident ones.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[DIAG] Image: {image_path}")

    img_pil = Image.open(image_path).convert('RGB')
    img_np = np.array(img_pil)
    w, h = img_pil.size
    print(f"[DIAG] Image size: {w}x{h}")

    # YOLO with very low conf to catch everything
    yolo = YOLO('yolov8x.pt')
    results = yolo(image_path, conf=yolo_conf, iou=0.3, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    yolo_confs = results[0].boxes.conf.cpu().numpy()
    yolo_classes = results[0].boxes.cls.cpu().numpy()
    yolo_names = results[0].names

    print(f"[DIAG] YOLO found {len(boxes)} proposals")
    print(f"[DIAG] YOLO class breakdown:")
    from collections import Counter
    yolo_class_counts = Counter([yolo_names[int(c)] for c in yolo_classes])
    for cls_name, count in yolo_class_counts.most_common():
        print(f"         {cls_name}: {count}")

    # Annotate original image with ALL YOLO boxes
    img_annotated = img_np.copy()
    for i, (box, yc, yc_idx) in enumerate(zip(boxes, yolo_confs, yolo_classes)):
        x1, y1, x2, y2 = map(int, box)
        yolo_label = yolo_names[int(yc_idx)]
        cv2.rectangle(img_annotated, (x1,y1), (x2,y2), (0,165,255), 2)
        cv2.putText(img_annotated, f"{yolo_label} {yc:.2f}",
                   (x1, max(y1-5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

    yolo_viz_path = str(output_dir / f"{Path(image_path).stem}_yolo_raw.jpg")
    cv2.imwrite(yolo_viz_path, cv2.cvtColor(img_annotated, cv2.COLOR_RGB2BGR))
    print(f"[DIAG] YOLO raw proposals: {yolo_viz_path}")

    # Classify ALL crops (no threshold) and make diagnostic grid
    crops_and_preds = []
    for box, yc in zip(boxes, yolo_confs):
        x1, y1, x2, y2 = map(int, box)
        if (x2-x1) < min_box_size or (y2-y1) < min_box_size:
            continue
        crop = img_pil.crop((x1, y1, x2, y2))
        preds = classify_crop_full(classifier_model, crop, class_names, topk=5)
        crops_and_preds.append((crop, [x1,y1,x2,y2], preds, yc))
        top_label, top_conf = preds[0]
        print(f"  Box [{x1},{y1},{x2},{y2}] YOLO_conf={yc:.2f} "
              f"-> #{1} {top_label} ({top_conf:.2%})")

    # Save diagnostic grid
    grid_path = str(output_dir / f"{Path(image_path).stem}_crop_grid.png")
    make_crop_grid(crops_and_preds, grid_path)

    print(f"\n[DIAG] Summary:")
    print(f"  Total YOLO proposals: {len(boxes)}")
    print(f"  Crops classified: {len(crops_and_preds)}")
    above_55 = sum(1 for _, _, preds, _ in crops_and_preds if preds[0][1] >= 0.55)
    above_30 = sum(1 for _, _, preds, _ in crops_and_preds if preds[0][1] >= 0.30)
    print(f"  Crops with conf >= 0.55: {above_55}")
    print(f"  Crops with conf >= 0.30: {above_30}")
    print(f"\n  -> If 'Crops with conf >= 0.55' is low, the domain gap is the issue")
    print(f"  -> If YOLO is finding wrong regions, check the _yolo_raw.jpg")
    print(f"  -> Check the _crop_grid.png to see what the classifier actually thinks")

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sample", "diagnose"], required=True,
                        help="sample: view dataset images | diagnose: debug task2 pipeline")

    # Sample mode args
    parser.add_argument("--n_per_class", type=int, default=3,
                        help="[sample] Number of images per class")
    parser.add_argument("--classes", nargs='+', default=None,
                        help="[sample] Specific classes to sample e.g. --classes idli biryani samosa")
    parser.add_argument("--sample_output", type=str,
                        default=os.path.expanduser("~/data/dataset_samples"))

    # Diagnose mode args
    parser.add_argument("--image", type=str, default=None,
                        help="[diagnose] Path to thali image")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--diag_output", type=str,
                        default=os.path.expanduser("~/data/task2_diag"))

    args = parser.parse_args()

    if args.mode == "sample":
        sample_dataset(
            DATA_ROOT,
            args.sample_output,
            n_per_class=args.n_per_class,
            classes=args.classes
        )

    elif args.mode == "diagnose":
        if not args.image:
            print("[ERROR] --image required for diagnose mode")
            sys.exit(1)

        class_names = get_class_names(DATA_ROOT)
        print(f"[INFO] Loading classifier...")
        classifier = load_task1_model(args.checkpoint, len(class_names))

        diagnose_image(
            args.image,
            classifier,
            class_names,
            args.diag_output
        )

if __name__ == "__main__":
    main()