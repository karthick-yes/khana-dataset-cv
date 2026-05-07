#!/usr/bin/env python3
"""Task 2 wrapper that outputs {image_name: [dish labels]}."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

import task2_detect as t2


def collect_images(single_image: str | None, input_dir: str) -> list[str]:
    if single_image:
        image_path = os.path.expanduser(single_image)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return [image_path]
    image_dir = Path(os.path.expanduser(input_dir))
    if not image_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {image_dir}")
    return sorted(
        str(p)
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=t2.CHECKPOINT_PATH)
    parser.add_argument("--input_dir", type=str, default=t2.TASK2_IMAGE_DIR)
    parser.add_argument("--single_image", type=str, default=None)
    parser.add_argument("--method", type=str, default="yolo", choices=["yolo", "sam"])
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
    )
    parser.add_argument("--detector_max_side", type=int, default=None)
    parser.add_argument("--low_vram", action="store_true")
    parser.add_argument("--sam_checkpoint", type=str, default=t2.SAM_CHECKPOINT)
    parser.add_argument("--sam_points_per_side", type=int, default=16)
    parser.add_argument("--no_sam_fallback_yolo", action="store_true")
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.path.expanduser("~/data/task2_dict_predictions.json"),
    )
    args = parser.parse_args()

    args.checkpoint = os.path.expanduser(args.checkpoint)
    args.sam_checkpoint = os.path.expanduser(args.sam_checkpoint)
    args.output_json = os.path.expanduser(args.output_json)
    if args.low_vram and args.detector_max_side is None:
        args.detector_max_side = 960

    image_paths = collect_images(args.single_image, args.input_dir)
    if not image_paths:
        print("[ERROR] No images found.")
        sys.exit(1)

    class_names = t2.get_class_names(t2.DATA_ROOT)
    classifier = t2.load_task1_model(args.checkpoint, num_classes=len(class_names))
    model_cache = {"yolo": None, "sam": None}

    def get_yolo_model():
        if model_cache["yolo"] is None:
            model_cache["yolo"] = t2.load_yolo(args.yolo_model)
        return model_cache["yolo"]

    def get_sam_generator():
        if model_cache["sam"] is None:
            model_cache["sam"] = t2.load_sam(args.sam_checkpoint, args.sam_points_per_side)
        return model_cache["sam"]

    payload = {}
    for image_path in image_paths:
        detections, _, _, _ = t2.detect_thali(
            image_path,
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
            debug=False,
            output_dir=None,
        )
        payload[Path(image_path).stem] = [d["label"] for d in detections]
        if args.low_vram and t2.DEVICE == "cuda":
            torch.cuda.empty_cache()

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved dictionary predictions: {args.output_json}")


if __name__ == "__main__":
    main()

