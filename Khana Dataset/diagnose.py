#!/usr/bin/env python3
"""
Diagnostic script to explore the structure and contents of the KHANA dataset.
This script will:"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import get_paths

paths = get_paths()
KHANA = paths["DATA_ROOT"]

print("=" * 70)
print("DIAGNOSTIC")
print("=" * 70)

# 1. Does the path even exist?
print(f"\n1. PATH CHECK")
print(f"   KHANA path: {KHANA}")
print(f"   Exists: {os.path.exists(KHANA)}")
if os.path.exists(KHANA):
    print(f"   Is dir: {os.path.isdir(KHANA)}")

# 2. List EVERYTHING in khana folder
print(f"\n2. CONTENTS OF khana/")
items = os.listdir(KHANA)
print(f"   Total items: {len(items)}")
print(f"   First 10: {items[:10]}")

# 3. Pick first folder and drill down
first_folder = os.path.join(KHANA, items[0]) if items else None
if first_folder and os.path.isdir(first_folder):
    print(f"\n3. DRILLING INTO: {items[0]}")
    files = os.listdir(first_folder)
    print(f"   Files found: {len(files)}")
    print(f"   First 10 filenames: {files[:10]}")

    # Show extensions
    exts = {}
    for f in files:
        ext = os.path.splitext(f)[1]
        exts[ext] = exts.get(ext, 0) + 1
    print(f"   Extension breakdown: {exts}")

    # Try to open one with PIL
    print(f"\n4. PIL IMAGE TEST")
    try:
        from PIL import Image

        test_file = os.path.join(first_folder, files[0])
        print(f"   Trying to open: {test_file}")
        img = Image.open(test_file)
        print(f"   SUCCESS! Size: {img.size}, Mode: {img.mode}")
    except Exception as e:
        print(f"   FAILED: {e}")

# 5. Count ALL files recursively with every possible extension
print(f"\n5. RECURSIVE FILE COUNT")
all_files = []
for root, dirs, files in os.walk(KHANA):
    for f in files:
        all_files.append(os.path.join(root, f))

print(f"   Total files (any type): {len(all_files)}")

ext_counts = {}
for f in all_files:
    ext = os.path.splitext(f)[1]
    ext_counts[ext] = ext_counts.get(ext, 0) + 1

print(
    f"   Extension counts: {dict(sorted(ext_counts.items(), key=lambda x: -x[1])[:20])}"
)

# 6. Try our exact filter logic
print(f"\n6. FILTER LOGIC TEST")
exts_lower = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
matched = [f for f in all_files if os.path.splitext(f)[1].lower() in exts_lower]
print(f"   Images found with .lower() filter: {len(matched)}")
print(f"   Sample paths:")
for p in matched[:3]:
    print(f"      {p}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
