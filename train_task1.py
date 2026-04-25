#!/usr/bin/env python3
"""
Task 1: Khana Image Classification — BULLETPROOF VERSION
- Saves checkpoint EVERY epoch
- Saves best model separately
- Auto-resumes from latest checkpoint
- Survives SSH disconnect (run inside tmux)
- Handles truncated/corrupted images
- TQDM progress bars
- Memory fragmentation fix
"""
import os
import sys
import argparse
import glob
from pathlib import Path
from collections import Counter

# Fix CUDA memory fragmentation BEFORE importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from timm.data.mixup import Mixup
from PIL import Image, ImageFile
import numpy as np
from tqdm import tqdm

# Allow truncated images to load instead of crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

from config import get_paths

# ============================================================
# CONFIG
# ============================================================
paths = get_paths()
DATA_ROOT = paths["DATA_ROOT"]
CHECKPOINT_DIR = paths["CHECKPOINT_DIR"]
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 25
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ============================================================
# DATASET
# ============================================================
class KhanaDataset(Dataset):
    def __init__(self, root, transform=None, split="train", val_ratio=0.2, seed=SEED):
        self.root = Path(root)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

        all_samples = []
        for cls in self.classes:
            folder = self.root / cls
            imgs = sorted([
                f for f in folder.iterdir()
                if f.is_file() and not f.name.startswith('.')
            ])
            all_samples.extend([(str(f), self.class_to_idx[cls]) for f in imgs])

        rng = np.random.RandomState(seed)
        train_samples, val_samples = [], []
        for cls_idx in range(len(self.classes)):
            cls_samples = [s for s in all_samples if s[1] == cls_idx]
            rng.shuffle(cls_samples)
            n_val = max(1, int(len(cls_samples) * val_ratio))
            val_samples.extend(cls_samples[:n_val])
            train_samples.extend(cls_samples[n_val:])

        self.samples = train_samples if split == "train" else val_samples
        print(f"[{split}] {len(self.samples)} samples across {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Edge case 1: corrupted/truncated image files
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            # Return a different random sample if this one is unreadable
            return self.__getitem__(np.random.randint(0, len(self.samples)))

        # Edge case 2: image loaded but has wrong mode or bad data after conversion
        try:
            if self.transform:
                img = self.transform(img)
        except Exception:
            return self.__getitem__(np.random.randint(0, len(self.samples)))

        return img, label

# ============================================================
# TRANSFORMS
# ============================================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# CHECKPOINT UTILS — BULLETPROOF
# ============================================================
def get_latest_checkpoint(checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return checkpoints[-1]

def save_checkpoint(epoch, model, optimizer, scheduler, best_acc, path, is_best=False):
    os.makedirs(path, exist_ok=True)
    filename = "best_model.pt" if is_best else f"checkpoint_epoch_{epoch}.pt"
    filepath = os.path.join(path, filename)

    # Edge case 3: disk write failure — don't crash training over a failed save
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'classes': model.num_classes if hasattr(model, 'num_classes') else 80,
        }, filepath)
        print(f"[INFO] Saved checkpoint: {filepath}")
    except Exception as e:
        print(f"[WARN] Failed to save checkpoint: {e}")

    return filepath

def load_checkpoint(path, model, optimizer, scheduler):
    print(f"[INFO] Loading checkpoint: {path}")
    # Edge case 4: checkpoint file exists but is corrupted (partial write from crash)
    try:
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['best_acc']
    except Exception as e:
        print(f"[WARN] Checkpoint corrupted or unreadable: {e}")
        print(f"[WARN] Starting fresh instead of crashing")
        return 0, 0.0

def cleanup_old_checkpoints(path, keep=3):
    pattern = os.path.join(path, "checkpoint_epoch_*.pt")
    checkpoints = sorted(
        glob.glob(pattern),
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )
    for old in checkpoints[:-keep]:
        try:
            os.remove(old)
            print(f"[INFO] Cleaned old checkpoint: {os.path.basename(old)}")
        except OSError:
            pass

# ============================================================
# TRAINING
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, mixup_fn, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    running_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [train]", 
                dynamic_ncols=True, leave=True)

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        images, labels = mixup_fn(images, labels)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        # Edge case 5: exploding gradients — clip to prevent NaN loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        running_loss = total_loss / (batch_idx + 1)

        # Update tqdm bar with current loss
        pbar.set_postfix({'loss': f'{running_loss:.4f}'})

    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, device, epoch, total_epochs):
    model.eval()
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [val]  ",
                dynamic_ncols=True, leave=True)

    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})

    return 100.0 * correct / total

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default="auto",
                       help="Path to checkpoint, or 'auto' to find latest, or 'none' to start fresh")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print(f"[INFO] Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"[INFO] Data: {DATA_ROOT}")
    print(f"[INFO] Checkpoints: {CHECKPOINT_DIR}")
    print(f"[INFO] PID: {os.getpid()}")

    # Edge case 6: data root doesn't exist
    if not os.path.exists(DATA_ROOT):
        print(f"[ERROR] Data root not found: {DATA_ROOT}")
        sys.exit(1)

    # Datasets
    print("[INFO] Loading datasets...")
    train_dataset = KhanaDataset(DATA_ROOT, transform=train_transform, split="train")
    val_dataset = KhanaDataset(DATA_ROOT, transform=val_transform, split="val")

    # Weighted sampler for class imbalance (63x ratio in Khana)
    print("[INFO] Computing class weights...")
    class_counts = Counter([label for _, label in train_dataset.samples])
    weights = [1.0 / class_counts[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        # Edge case 7: prevent DataLoader deadlock on display sleep
        multiprocessing_context='spawn'
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True,
        multiprocessing_context='spawn'
    )

    # Model — IN-22k pretrained, beats IN-1k baseline by ~1.5-2%
    print("[INFO] Loading ConvNeXt-Base (IN-22k)...")
    model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=80)
    model = model.to(DEVICE)

    # MixUp + CutMix — critical for visually similar classes (dosa/pav/paneer families)
    mixup_fn = Mixup(
        mixup_alpha=0.2,
        cutmix_alpha=1.0,
        prob=0.5,
        switch_prob=0.5,
        mode='batch',
        num_classes=80
    )

    # Layer-wise LR: low for backbone (don't destroy pretrained features), high for head
    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'head' not in n], 'lr': 1e-5},
        {'params': model.head.parameters(), 'lr': 1e-3}
    ], weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss()

    # Resume logic
    start_epoch = 0
    best_acc = 0.0
    resume_path = None

    if args.resume == "auto":
        resume_path = get_latest_checkpoint(CHECKPOINT_DIR)
        if resume_path:
            print(f"[INFO] Auto-resume found: {resume_path}")
        else:
            print("[INFO] No checkpoint found, starting fresh")
    elif args.resume != "none":
        resume_path = args.resume

    if resume_path and os.path.exists(resume_path):
        start_epoch, best_acc = load_checkpoint(resume_path, model, optimizer, scheduler)
        start_epoch += 1
        print(f"[INFO] Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")

    # Training loop
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING: epochs {start_epoch}-{args.epochs}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        # Edge case 8: GPU OOM mid-training — clear cache before each epoch
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            mixup_fn, DEVICE, epoch, args.epochs
        )
        val_acc = validate(model, val_loader, DEVICE, epoch, args.epochs)

        scheduler.step()

        print(f"\n[RESULT] Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss={train_loss:.4f} | Val Acc={val_acc:.2f}% | Best={best_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(epoch, model, optimizer, scheduler, best_acc, CHECKPOINT_DIR, is_best=True)
            print(f"[INFO] *** NEW BEST: {best_acc:.2f}% ***")

        save_checkpoint(epoch, model, optimizer, scheduler, best_acc, CHECKPOINT_DIR)
        cleanup_old_checkpoints(CHECKPOINT_DIR, keep=3)

        # Flush so logs are written even if process is killed
        sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — Best Val Accuracy: {best_acc:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()