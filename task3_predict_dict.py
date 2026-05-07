#!/usr/bin/env python3
"""Task 3 wrapper that outputs {image_name: [dish labels]}."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

import task3_bev as t3


def collect_images(single_image: str | None, input_dir: str | None) -> list[str]:
    if single_image:
        image_path = os.path.expanduser(single_image)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        return [image_path]
    if not input_dir:
        raise ValueError("Provide --single_image or --input_dir")
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
    parser.add_argument("--single_image", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=t3.OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=str, default=t3.CHECKPOINT_PATH)
    parser.add_argument("--sam_checkpoint", type=str, default=t3.SAM_CHECKPOINT)
    parser.add_argument("--sam_points_per_side", type=int, default=16)
    parser.add_argument("--conf_threshold", type=float, default=0.45)
    parser.add_argument("--bev_size", type=int, default=900)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--output_json",
        type=str,
        default=os.path.expanduser("~/data/task3_dict_predictions.json"),
    )
    args = parser.parse_args()

    args.output_dir = os.path.expanduser(args.output_dir)
    args.checkpoint = os.path.expanduser(args.checkpoint)
    args.sam_checkpoint = os.path.expanduser(args.sam_checkpoint)
    args.output_json = os.path.expanduser(args.output_json)
    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = collect_images(args.single_image, args.input_dir)
    if not image_paths:
        print("[ERROR] No images found.")
        sys.exit(1)

    class_names = t3.get_class_names(t3.DATA_ROOT)
    classifier = t3.load_classifier(args.checkpoint, num_classes=len(class_names))
    sam_generator = t3.load_sam(args.sam_checkpoint, args.sam_points_per_side)

    payload = {}
    for image_path in image_paths:
        result = t3.process_image(
            image_path,
            sam_generator=sam_generator,
            classifier=classifier,
            class_names=class_names,
            output_dir=args.output_dir,
            conf_threshold=args.conf_threshold,
            bev_size=args.bev_size,
            debug=args.debug,
        )
        labels = []
        if result:
            labels = [d["label"] for d in result.get("detections", [])]
        payload[Path(image_path).stem] = labels
        if t3.DEVICE == "cuda":
            torch.cuda.empty_cache()

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved dictionary predictions: {args.output_json}")


if __name__ == "__main__":
    main()

