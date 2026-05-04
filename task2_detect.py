#!/usr/bin/env python3
"""
Task 2 detector (YOLO + SAM only).

Methods:
  - yolo : YOLO proposals -> Task 1 classifier relabeling
  - sam  : SAM proposals -> Task 1 classifier relabeling
           (optional YOLO fallback when SAM yields no usable boxes)

Debug mode:
  Saves per-image stage artifacts so you can inspect:
    YOLO: raw boxes, survived boxes, classified-pre-nms, final
    SAM : raw segment overlay, raw mask boxes, survived boxes, classified-pre-nms, final
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

from config import get_paths

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# CONFIG
# ============================================================
paths = get_paths()
DATA_ROOT = paths["DATA_ROOT"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TASK2_IMAGE_DIR = os.path.expanduser("~/data/task2_images")
OUTPUT_DIR = os.path.expanduser("~/data/task2_output")
CHECKPOINT_PATH = os.path.expanduser("~/data/checkpoints/best_model.pt")
SAM_CHECKPOINT = os.path.expanduser("~/data/sam_vit_b_01ec64.pth")
SAM_MODEL_TYPE = "vit_b"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATA + MODELS
# ============================================================
def get_class_names(data_root):
    root = Path(data_root)
    if not root.exists():
        print(f"[ERROR] Data root not found: {data_root}")
        sys.exit(1)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    print(f"[INFO] Loaded {len(classes)} class names from dataset")
    return classes


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


def load_task1_model(checkpoint_path, num_classes):
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    print("[INFO] Loading Task 1 classifier...")
    model = timm.create_model(
        "convnext_base.fb_in22k_ft_in1k", pretrained=False, num_classes=num_classes
    )
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Loaded. Best val acc: {ckpt.get('best_acc', 'N/A')}%")
    model = model.to(DEVICE).eval()
    return model


def load_yolo(model_path):
    print(f"[INFO] Loading YOLO model: {model_path}")
    return YOLO(model_path)


def load_sam(sam_checkpoint, points_per_side):
    if not os.path.exists(sam_checkpoint):
        print(f"[WARN] SAM checkpoint not found: {sam_checkpoint}")
        return None
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError:
        print("[WARN] segment_anything is not installed. SAM unavailable.")
        return None

    print(f"[INFO] Loading SAM ({SAM_MODEL_TYPE})...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=sam_checkpoint).to(DEVICE)
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
    pred_idx = probs.argmax(dim=1).item()
    pred_conf = probs.max(dim=1).values.item()
    top5_idx = probs[0].topk(5).indices.cpu().numpy()
    top5 = [(class_names[i], float(probs[0][i])) for i in top5_idx]
    return pred_idx, pred_conf, top5


# ============================================================
# BOX + IMAGE UTILS
# ============================================================
def compute_iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (area1 + area2 - inter + 1e-6)


def merge_overlapping_boxes(boxes, iou_threshold=0.4):
    if len(boxes) <= 1:
        return boxes
    merged = []
    used = [False] * len(boxes)
    for i, b1 in enumerate(boxes):
        if used[i]:
            continue
        group = [b1]
        for j, b2 in enumerate(boxes):
            if i == j or used[j]:
                continue
            if compute_iou(b1, b2) > iou_threshold:
                group.append(b2)
                used[j] = True
        merged.append(
            [
                min(b[0] for b in group),
                min(b[1] for b in group),
                max(b[2] for b in group),
                max(b[3] for b in group),
            ]
        )
        used[i] = True
    return merged


def remove_duplicates(detections, iou_threshold=0.4, same_label_iou=0.05):
    if len(detections) <= 1:
        return detections
    detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    kept = []
    for det in detections:
        duplicate = False
        for k in kept:
            iou = compute_iou(det["box"], k["box"])
            if det["label"] == k["label"] and iou > same_label_iou:
                duplicate = True
                break
            if det["label"] != k["label"] and iou > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


def resize_for_detection(img_np, detector_max_side):
    h, w = img_np.shape[:2]
    if not detector_max_side or max(h, w) <= detector_max_side:
        return img_np, 1.0
    scale = detector_max_side / float(max(h, w))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def clip_box(box, img_w, img_h):
    x1 = max(0, min(img_w - 1, int(box[0])))
    y1 = max(0, min(img_h - 1, int(box[1])))
    x2 = max(0, min(img_w - 1, int(box[2])))
    y2 = max(0, min(img_h - 1, int(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def draw_boxes(img_bgr, boxes, color, label_prefix=None):
    out = img_bgr.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if label_prefix is not None:
            cv2.putText(
                out,
                f"{label_prefix}{i+1}",
                (x1, max(15, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
    return out


def save_debug_image(output_dir, stem, suffix, img_bgr):
    out_path = os.path.join(output_dir, f"{stem}_{suffix}.jpg")
    cv2.imwrite(out_path, img_bgr)
    return out_path


def make_sam_overlay(img_rgb, masks):
    overlay = img_rgb.copy()
    rng = np.random.default_rng(7)
    for m in masks:
        seg = m["segmentation"]
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        overlay[seg] = (0.5 * overlay[seg] + 0.5 * color).astype(np.uint8)
    return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)


# ============================================================
# PROPOSAL GENERATORS
# ============================================================
def find_compartments_yolo(
    yolo_model,
    img_pil,
    yolo_conf=0.15,
    yolo_iou=0.5,
    min_box_size=50,
    max_box_ratio=0.40,
    detector_max_side=None,
    detector_device="auto",
):
    img_np = np.array(img_pil)
    img_h, img_w = img_np.shape[:2]
    img_area = float(img_w * img_h)
    work_np, scale = resize_for_detection(img_np, detector_max_side)

    predict_kwargs = {"conf": yolo_conf, "iou": yolo_iou, "verbose": False}
    if detector_device == "cuda" and torch.cuda.is_available():
        predict_kwargs["device"] = 0
    elif detector_device == "cpu":
        predict_kwargs["device"] = "cpu"

    results = yolo_model(work_np, **predict_kwargs)
    raw_boxes = results[0].boxes.xyxy.cpu().numpy()

    raw_boxes_scaled = []
    filtered_boxes = []
    for box in raw_boxes:
        x1, y1, x2, y2 = box.tolist()
        x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
        clipped = clip_box([x1, y1, x2, y2], img_w, img_h)
        if clipped is None:
            continue
        raw_boxes_scaled.append(clipped)

        w = clipped[2] - clipped[0]
        h = clipped[3] - clipped[1]
        if w < min_box_size or h < min_box_size:
            continue
        if (w * h) / img_area > max_box_ratio:
            continue
        filtered_boxes.append(clipped)

    print(f"[YOLO] raw={len(raw_boxes_scaled)} filtered={len(filtered_boxes)}")
    return filtered_boxes, {
        "raw_boxes": raw_boxes_scaled,
        "filtered_boxes": filtered_boxes,
        "raw_count": len(raw_boxes_scaled),
        "filtered_count": len(filtered_boxes),
    }


def find_compartments_sam(
    img_pil,
    sam_generator,
    min_area_ratio=0.01,
    max_area_ratio=0.35,
    min_box_size=40,
):
    img_np = np.array(img_pil)
    img_area = float(img_np.shape[0] * img_np.shape[1])
    print("[SAM] Running automatic segmentation...")
    masks = sam_generator.generate(img_np)

    raw_boxes = []
    filtered_boxes = []
    for mask_data in masks:
        x, y, w, h = mask_data["bbox"]
        raw_boxes.append([int(x), int(y), int(x + w), int(y + h)])

        area = mask_data["area"]
        ratio = area / img_area
        if ratio < min_area_ratio or ratio > max_area_ratio:
            continue
        if w < min_box_size or h < min_box_size:
            continue
        aspect = w / h if h > 0 else 0
        if aspect < 0.15 or aspect > 6.0:
            continue
        filtered_boxes.append([int(x), int(y), int(x + w), int(y + h)])

    filtered_boxes = merge_overlapping_boxes(filtered_boxes, iou_threshold=0.4)
    print(
        f"[SAM] raw_masks={len(masks)} raw_boxes={len(raw_boxes)} filtered={len(filtered_boxes)}"
    )
    return filtered_boxes, {
        "masks": masks,
        "raw_boxes": raw_boxes,
        "filtered_boxes": filtered_boxes,
        "raw_mask_count": len(masks),
        "raw_box_count": len(raw_boxes),
        "filtered_count": len(filtered_boxes),
    }


# ============================================================
# PIPELINE
# ============================================================
def detect_thali(
    image_path,
    classifier_model,
    class_names,
    method,
    get_yolo_model,
    get_sam_generator,
    conf_threshold=0.55,
    crop_padding=8,
    yolo_conf=0.15,
    yolo_iou=0.5,
    detector_max_side=None,
    detector_device="auto",
    sam_fallback_yolo=True,
    debug=False,
    output_dir=None,
):
    try:
        img_pil = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Cannot open {image_path}: {e}")
        return [], None, "error", {}

    stem = Path(image_path).stem
    img_rgb = np.array(img_pil)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    proposal_debug = {}
    if method == "yolo":
        yolo_model = get_yolo_model()
        candidate_boxes, proposal_debug = find_compartments_yolo(
            yolo_model,
            img_pil,
            yolo_conf=yolo_conf,
            yolo_iou=yolo_iou,
            detector_max_side=detector_max_side,
            detector_device=detector_device,
        )
        source = "yolo"
    else:
        sam_generator = get_sam_generator()
        if sam_generator is None:
            if not sam_fallback_yolo:
                print("[ERROR] SAM unavailable and fallback disabled.")
                return [], img_pil, "sam", {}
            print("[WARN] SAM unavailable. Falling back to YOLO.")
            yolo_model = get_yolo_model()
            candidate_boxes, proposal_debug = find_compartments_yolo(
                yolo_model,
                img_pil,
                yolo_conf=yolo_conf,
                yolo_iou=yolo_iou,
                detector_max_side=detector_max_side,
                detector_device=detector_device,
            )
            source = "yolo_fallback"
        else:
            candidate_boxes, proposal_debug = find_compartments_sam(img_pil, sam_generator)
            source = "sam"
            if not candidate_boxes and sam_fallback_yolo:
                print("[INFO] SAM returned no usable boxes. Falling back to YOLO.")
                yolo_model = get_yolo_model()
                candidate_boxes, proposal_debug = find_compartments_yolo(
                    yolo_model,
                    img_pil,
                    yolo_conf=yolo_conf,
                    yolo_iou=yolo_iou,
                    detector_max_side=detector_max_side,
                    detector_device=detector_device,
                )
                source = "yolo_fallback"

    debug_summary = {
        "proposal_source": source,
        "proposal_raw_count": proposal_debug.get("raw_count", proposal_debug.get("raw_box_count", 0)),
        "proposal_survived_count": proposal_debug.get("filtered_count", 0),
        "classified_pre_nms_count": 0,
        "final_count": 0,
        "artifacts": {},
    }
    if "raw_mask_count" in proposal_debug:
        debug_summary["raw_mask_count"] = proposal_debug["raw_mask_count"]

    if debug and output_dir:
        if source.startswith("yolo"):
            raw_overlay = draw_boxes(
                img_bgr, proposal_debug.get("raw_boxes", []), (0, 165, 255), "r"
            )
            filtered_overlay = draw_boxes(
                img_bgr, proposal_debug.get("filtered_boxes", []), (0, 255, 0), "f"
            )
            debug_summary["artifacts"]["yolo_raw_boxes"] = save_debug_image(
                output_dir, stem, "yolo_raw_boxes", raw_overlay
            )
            debug_summary["artifacts"]["yolo_filtered_boxes"] = save_debug_image(
                output_dir, stem, "yolo_filtered_boxes", filtered_overlay
            )
        elif source == "sam":
            sam_overlay = make_sam_overlay(img_rgb, proposal_debug.get("masks", []))
            raw_overlay = draw_boxes(
                img_bgr, proposal_debug.get("raw_boxes", []), (0, 165, 255), "r"
            )
            filtered_overlay = draw_boxes(
                img_bgr, proposal_debug.get("filtered_boxes", []), (0, 255, 0), "f"
            )
            debug_summary["artifacts"]["sam_raw_segments"] = save_debug_image(
                output_dir, stem, "sam_raw_segments", sam_overlay
            )
            debug_summary["artifacts"]["sam_raw_boxes"] = save_debug_image(
                output_dir, stem, "sam_raw_boxes", raw_overlay
            )
            debug_summary["artifacts"]["sam_filtered_boxes"] = save_debug_image(
                output_dir, stem, "sam_filtered_boxes", filtered_overlay
            )

    if not candidate_boxes:
        print(f"[WARN] No compartments found for {image_path}")
        return [], img_pil, source, debug_summary

    img_w, img_h = img_pil.size
    pre_nms_detections = []
    for box in candidate_boxes:
        x1, y1, x2, y2 = box
        x1p = max(0, x1 - crop_padding)
        y1p = max(0, y1 - crop_padding)
        x2p = min(img_w, x2 + crop_padding)
        y2p = min(img_h, y2 + crop_padding)
        crop = img_pil.crop((x1p, y1p, x2p, y2p))

        pred_idx, pred_conf, top5 = classify_crop(classifier_model, crop, class_names)
        if pred_conf < conf_threshold:
            continue
        pre_nms_detections.append(
            {
                "box": [x1, y1, x2, y2],
                "label": class_names[pred_idx],
                "label_idx": pred_idx,
                "confidence": float(pred_conf),
                "source": source,
                "top5": top5,
            }
        )

    debug_summary["classified_pre_nms_count"] = len(pre_nms_detections)
    print(f"[INFO] After confidence filter ({conf_threshold}): {len(pre_nms_detections)}")

    if debug and output_dir:
        pre_nms_boxes = [d["box"] for d in pre_nms_detections]
        pre_nms_overlay = draw_boxes(img_bgr, pre_nms_boxes, (255, 0, 255), "c")
        debug_summary["artifacts"]["classified_pre_nms"] = save_debug_image(
            output_dir, stem, "classified_pre_nms", pre_nms_overlay
        )

    final_detections = remove_duplicates(pre_nms_detections)
    debug_summary["final_count"] = len(final_detections)
    print(f"[INFO] After NMS: {len(final_detections)}")
    for d in final_detections:
        print(f"  -> [{d['source']}] {d['label']:30s} conf={d['confidence']:.3f}")

    if debug:
        if "raw_mask_count" in debug_summary:
            print(
                "[DEBUG][SAM] "
                f"raw_masks={debug_summary['raw_mask_count']} "
                f"survived={debug_summary['proposal_survived_count']} "
                f"classified={debug_summary['classified_pre_nms_count']} "
                f"final={debug_summary['final_count']}"
            )
        else:
            print(
                "[DEBUG][YOLO] "
                f"raw_boxes={debug_summary['proposal_raw_count']} "
                f"survived={debug_summary['proposal_survived_count']} "
                f"classified={debug_summary['classified_pre_nms_count']} "
                f"final={debug_summary['final_count']}"
            )

    return final_detections, img_pil, source, debug_summary


# ============================================================
# OUTPUT
# ============================================================
def visualize_detections(image_path, detections, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        pil = Image.open(image_path).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
        (255, 0, 128),
    ]
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["box"]
        color = colors[i % len(colors)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{d['label']} {d['confidence']:.2f} [{d.get('source', '')}]"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
        )

    out_path = os.path.join(output_dir, f"{Path(image_path).stem}_detected.jpg")
    cv2.imwrite(out_path, img)
    return out_path


def save_results(image_path, detections, output_dir, method_used, debug_summary):
    basename = Path(image_path).stem
    payload = {
        "image": basename,
        "method_used": method_used,
        "num_detections": len(detections),
        "detections": detections,
        "debug_summary": debug_summary,
    }

    json_path = os.path.join(output_dir, f"{basename}_detections.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    txt_path = os.path.join(output_dir, f"{basename}.txt")
    with open(txt_path, "w") as f:
        for d in detections:
            b = d["box"]
            f.write(f"{d['label']} {d['confidence']:.4f} {b[0]} {b[1]} {b[2]} {b[3]}\n")
    print(f"[INFO] Results saved: {json_path}")
    return json_path


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--input_dir", type=str, default=TASK2_IMAGE_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--single_image", type=str, default=None)
    parser.add_argument("--no_viz", action="store_true")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument(
        "--method",
        type=str,
        default="yolo",
        choices=["yolo", "sam"],
        help="Task 2 proposal generator",
    )
    parser.add_argument("--conf_threshold", type=float, default=0.55)
    parser.add_argument("--crop_padding", type=int, default=8)

    parser.add_argument("--yolo_model", type=str, default="yolov8x.pt")
    parser.add_argument("--yolo_conf", type=float, default=0.15)
    parser.add_argument("--yolo_iou", type=float, default=0.50)
    parser.add_argument(
        "--detector_device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for YOLO proposal generation.",
    )
    parser.add_argument(
        "--detector_max_side",
        type=int,
        default=None,
        help="Downscale long side for YOLO proposal generation.",
    )
    parser.add_argument("--low_vram", action="store_true")

    parser.add_argument("--sam_checkpoint", type=str, default=SAM_CHECKPOINT)
    parser.add_argument("--sam_points_per_side", type=int, default=16)
    parser.add_argument(
        "--no_sam_fallback_yolo",
        action="store_true",
        help="Disable YOLO fallback when SAM is unavailable or empty.",
    )

    args = parser.parse_args()

    args.single_image = os.path.expanduser(args.single_image) if args.single_image else None
    args.input_dir = os.path.expanduser(args.input_dir)
    args.output_dir = os.path.expanduser(args.output_dir)
    args.checkpoint = os.path.expanduser(args.checkpoint)
    args.sam_checkpoint = os.path.expanduser(args.sam_checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.low_vram and args.detector_max_side is None:
        args.detector_max_side = 960

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Method: {args.method}")
    print(f"[INFO] Confidence threshold: {args.conf_threshold}")
    print(f"[INFO] Detector max side: {args.detector_max_side}")
    print(f"[INFO] Debug mode: {args.debug}")

    class_names = get_class_names(DATA_ROOT)
    classifier = load_task1_model(args.checkpoint, num_classes=len(class_names))

    model_cache = {"yolo": None, "sam": None}

    def get_yolo_model():
        if model_cache["yolo"] is None:
            model_cache["yolo"] = load_yolo(args.yolo_model)
        return model_cache["yolo"]

    def get_sam_generator():
        if model_cache["sam"] is None:
            model_cache["sam"] = load_sam(args.sam_checkpoint, args.sam_points_per_side)
        return model_cache["sam"]

    if args.single_image:
        image_paths = [args.single_image]
    else:
        image_paths = sorted(
            str(p) for p in Path(args.input_dir).iterdir() if p.is_file() and not p.name.startswith(".")
        )

    if not image_paths:
        print(f"[ERROR] No images found in {args.input_dir}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Processing {len(image_paths)} images")
    print(f"{'=' * 60}\n")

    all_results = []
    for img_path in image_paths:
        print(f"\n{'=' * 60}")
        print(f"Image: {Path(img_path).name}")
        print(f"{'=' * 60}")

        detections, _, method_used, debug_summary = detect_thali(
            img_path,
            classifier_model=classifier,
            class_names=class_names,
            method=args.method,
            get_yolo_model=get_yolo_model,
            get_sam_generator=get_sam_generator,
            conf_threshold=args.conf_threshold,
            crop_padding=args.crop_padding,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            detector_max_side=args.detector_max_side,
            detector_device=args.detector_device,
            sam_fallback_yolo=(not args.no_sam_fallback_yolo),
            debug=args.debug,
            output_dir=args.output_dir,
        )

        json_path = save_results(
            img_path,
            detections,
            args.output_dir,
            method_used=method_used,
            debug_summary=debug_summary,
        )
        if not args.no_viz:
            final_viz_path = visualize_detections(img_path, detections, args.output_dir)
            print(f"[INFO] Final visualization: {final_viz_path}")
        else:
            final_viz_path = None

        all_results.append(
            {
                "image": Path(img_path).name,
                "method_used": method_used,
                "json_path": json_path,
                "final_viz_path": final_viz_path,
                "debug_summary": debug_summary,
                "detections": detections,
            }
        )

        if args.low_vram and DEVICE == "cuda":
            torch.cuda.empty_cache()

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done. Results in: {args.output_dir}")
    print(f"Summary: {summary_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
