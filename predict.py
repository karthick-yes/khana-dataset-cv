#!/usr/bin/env python3
"""Task 1 evaluation wrapper exposing predict(image)."""

from __future__ import annotations

import os
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms

from config import get_paths

_MODEL = None
_CLASS_NAMES = None
_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(int(224 * 1.14)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_class_names() -> list[str]:
    paths = get_paths()
    labels_path = os.path.expanduser(paths.get("LABELS_PATH", ""))
    if labels_path and os.path.exists(labels_path):
        labels = [line.strip() for line in Path(labels_path).read_text().splitlines() if line.strip()]
        if labels:
            return labels

    data_root = Path(os.path.expanduser(paths["DATA_ROOT"]))
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT not found: {data_root}")
    return sorted(d.name for d in data_root.iterdir() if d.is_dir())


def _resolve_checkpoint_path() -> str:
    env_path = os.getenv("TASK1_CHECKPOINT")
    if env_path:
        return os.path.expanduser(env_path)
    paths = get_paths()
    return os.path.join(os.path.expanduser(paths["CHECKPOINT_DIR"]), "best_model.pt")


def _ensure_model_loaded():
    global _MODEL, _CLASS_NAMES
    if _MODEL is not None and _CLASS_NAMES is not None:
        return

    _CLASS_NAMES = _load_class_names()
    checkpoint_path = _resolve_checkpoint_path()
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = timm.create_model(
        "convnext_base.fb_in22k_ft_in1k",
        pretrained=False,
        num_classes=len(_CLASS_NAMES),
    )
    checkpoint = torch.load(checkpoint_path, map_location=_DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _MODEL = model.to(_DEVICE).eval()


@torch.no_grad()
def predict(image):
    """
    Predict a Task 1 class label from a PIL image.

    Args:
        image: PIL RGB image (or PIL image convertible to RGB)

    Returns:
        str: class name
    """
    if not isinstance(image, Image.Image):
        raise TypeError("predict(image) expects a PIL.Image input")

    _ensure_model_loaded()
    rgb = image.convert("RGB")
    tensor = _TRANSFORM(rgb).unsqueeze(0).to(_DEVICE)
    logits = _MODEL(tensor)
    idx = int(torch.argmax(logits, dim=1).item())
    return _CLASS_NAMES[idx]

