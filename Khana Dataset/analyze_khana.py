import os
import sys
import json
import csv
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================
BASE_DIR = "/home/karthiksunil/work/code/cv/CuisineProject/Khana Dataset"
DATA_ROOT = os.path.join(BASE_DIR, "extracted", "khana")
LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")
TAXONOMY_PATH = os.path.join(BASE_DIR, "taxonomy.csv")
OUTDIR = os.path.join(BASE_DIR, "analysis_output")


def get_class_folders(root):
    """List all class folders under extracted/khana/"""
    if not os.path.exists(root):
        print(f"[ERROR] Data root not found: {root}")
        sys.exit(1)
    folders = []
    for item in sorted(os.listdir(root)):
        full = os.path.join(root, item)
        if os.path.isdir(full):
            folders.append((item, full))
    return folders


def collect_images(class_folders):
    """
    Collect ALL non-hidden files from each class folder.
    """
    class_images = {}
    all_images = []

    for class_name, folder in class_folders:
        imgs = []
        for f in os.listdir(folder):
            # Skip hidden files (like .DS_Store) and directories
            if f.startswith("."):
                continue
            fp = os.path.join(folder, f)
            if os.path.isfile(fp):
                imgs.append(fp)
                all_images.append(fp)
        class_images[class_name] = sorted(imgs)

    return class_images, all_images


def parse_labels(labels_path):
    if not os.path.exists(labels_path):
        return []
    with open(labels_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    print(f"[INFO] labels.txt: {len(lines)} class names")
    return lines


def parse_taxonomy(taxonomy_path):
    if not os.path.exists(taxonomy_path):
        return []
    data = []
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        f.seek(0)
        delimiter = "," if sample.count(",") > sample.count("\t") else "\t"
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            data.append(dict(row))
    print(
        f"[INFO] taxonomy.csv: {len(data)} rows, cols: {list(data[0].keys()) if data else 'N/A'}"
    )
    return data


def compute_stats(class_images):
    class_counts = {cls: len(imgs) for cls, imgs in class_images.items()}

    # Sample dimensions (up to 1000 images, spread across classes)
    dims = []
    sampled = 0
    max_sample = 1000

    print("[INFO] Sampling image dimensions...")
    for cls, imgs in class_images.items():
        to_sample = min(len(imgs), max(1, max_sample // len(class_images)))
        for fp in imgs[:to_sample]:
            if sampled >= max_sample:
                break
            try:
                with Image.open(fp) as img:
                    w, h = img.size
                    mode = img.mode
                    dims.append((w, h, mode, cls))
                    sampled += 1
            except Exception:
                continue

    dim_stats = None
    if dims:
        widths, heights, modes, _ = zip(*dims)
        mode_counts = Counter(modes)
        dim_stats = {
            "count": len(dims),
            "width_mean": float(np.mean(widths)),
            "width_std": float(np.std(widths)),
            "height_mean": float(np.mean(heights)),
            "height_std": float(np.std(heights)),
            "min_w": int(min(widths)),
            "max_w": int(max(widths)),
            "min_h": int(min(heights)),
            "max_h": int(max(heights)),
            "aspect_ratios": [w / h for w, h in zip(widths, heights)],
            "mode_distribution": dict(mode_counts),
        }

    counts = list(class_counts.values())
    return {
        "num_classes": len(class_counts),
        "total_images": sum(counts),
        "class_counts": dict(sorted(class_counts.items(), key=lambda x: -x[1])),
        "dim_stats": dim_stats,
        "min_class_size": min(counts),
        "max_class_size": max(counts),
        "median_class_size": int(np.median(counts)),
        "mean_class_size": int(np.mean(counts)),
    }


def find_confusing_pairs(class_counts, taxonomy_data):
    classes = list(class_counts.keys())

    # Name-based: shared words
    word_map = defaultdict(set)
    for c in classes:
        words = set(c.lower().replace("_", " ").replace("-", " ").split())
        for w in words:
            if len(w) > 2:
                word_map[w].add(c)

    confusing = []
    seen = set()
    for word, cls_set in word_map.items():
        if 1 < len(cls_set) <= 6:
            pair = tuple(sorted(cls_set))
            if pair not in seen:
                seen.add(pair)
                counts = [class_counts.get(c, 0) for c in cls_set]
                confusing.append(
                    {
                        "reason": f"share word '{word}'",
                        "classes": list(cls_set),
                        "counts": counts,
                        "total": sum(counts),
                    }
                )

    confusing.sort(key=lambda x: x["total"], reverse=True)

    # Taxonomy-based
    tax_conf = []
    if taxonomy_data:
        dish_map = {}
        for row in taxonomy_data:
            dish = row.get("dish", "").strip().lower()
            if dish:
                dish_map[dish] = row

        cat_groups = defaultdict(list)
        variety_groups = defaultdict(list)

        for cls in classes:
            cls_key = cls.strip().lower()
            row = dish_map.get(cls_key, {})
            cat = row.get("category", "").strip()
            var = row.get("variety", "").strip()
            if cat:
                cat_groups[cat].append(cls)
            if var:
                variety_groups[var].append(cls)

        for cat, group in cat_groups.items():
            if len(group) > 1:
                tax_conf.append(
                    {
                        "group_col": "category",
                        "group_val": cat,
                        "classes": sorted(group),
                        "counts": [class_counts.get(c, 0) for c in group],
                    }
                )

        for var, group in variety_groups.items():
            if len(group) > 1:
                tax_conf.append(
                    {
                        "group_col": "variety",
                        "group_val": var,
                        "classes": sorted(group),
                        "counts": [class_counts.get(c, 0) for c in group],
                    }
                )

    return confusing[:25], tax_conf[:20]


def generate_plots(stats, confusing_pairs, taxonomy_conf, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Top 40 class distribution
    plt.figure(figsize=(14, 12))
    items = list(stats["class_counts"].items())[:40]
    classes, counts = zip(*items)
    plt.barh(range(len(classes)), counts, color="steelblue")
    plt.yticks(range(len(classes)), classes, fontsize=8)
    plt.xlabel("Number of Images")
    plt.title(
        f"Top 40 Classes (Total: {stats['num_classes']} classes, {stats['total_images']} images)"
    )
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_class_distribution.png", dpi=150)
    plt.close()
    print(f"[INFO] Saved: 01_class_distribution.png")

    # 2. Imbalance histogram
    plt.figure(figsize=(10, 6))
    all_counts = list(stats["class_counts"].values())
    plt.hist(all_counts, bins=30, color="coral", edgecolor="black", alpha=0.8)
    plt.axvline(
        stats["median_class_size"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median: {stats['median_class_size']}",
    )
    plt.axvline(
        stats["mean_class_size"],
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {stats['mean_class_size']}",
    )
    plt.xlabel("Images per Class")
    plt.ylabel("Number of Classes")
    plt.title("Class Imbalance Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_imbalance_histogram.png", dpi=150)
    plt.close()
    print(f"[INFO] Saved: 02_imbalance_histogram.png")

    # 3. Dimensions
    if stats["dim_stats"]:
        ds = stats["dim_stats"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        np.random.seed(42)
        n = min(500, ds["count"])
        w_samples = np.random.normal(ds["width_mean"], ds["width_std"], n)
        h_samples = np.random.normal(ds["height_mean"], ds["height_std"], n)
        w_samples = np.clip(w_samples, ds["min_w"], ds["max_w"])
        h_samples = np.clip(h_samples, ds["min_h"], ds["max_h"])

        axes[0].scatter(w_samples, h_samples, alpha=0.3, s=10, color="blue")
        axes[0].axvline(ds["width_mean"], color="red", linestyle="--", linewidth=2)
        axes[0].axhline(ds["height_mean"], color="red", linestyle="--", linewidth=2)
        axes[0].set_xlabel("Width (px)")
        axes[0].set_ylabel("Height (px)")
        axes[0].set_title(
            f"Image Dimensions (n={ds['count']} sampled)\n"
            f"Mean: {ds['width_mean']:.0f}×{ds['height_mean']:.0f}"
        )
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(
            ds["aspect_ratios"], bins=30, color="purple", edgecolor="black", alpha=0.7
        )
        axes[1].axvline(
            np.mean(ds["aspect_ratios"]),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean AR: {np.mean(ds['aspect_ratios']):.2f}",
        )
        axes[1].set_xlabel("Aspect Ratio (W/H)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Aspect Ratio Distribution")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/03_dimensions.png", dpi=150)
        plt.close()
        print(f"[INFO] Saved: 03_dimensions.png")

    # 4. Confusing pairs
    if confusing_pairs:
        fig, ax = plt.subplots(figsize=(14, min(3 + len(confusing_pairs) * 0.4, 18)))
        ax.axis("off")
        rows = []
        for p in confusing_pairs[:30]:
            rows.append(
                [p["reason"], ", ".join(p["classes"]), ", ".join(map(str, p["counts"]))]
            )
        table = ax.table(
            cellText=rows,
            colLabels=["Reason", "Classes", "Counts"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        plt.title("Potentially Confusing Class Pairs", fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/04_confusing_pairs.png", dpi=150)
        plt.close()
        print(f"[INFO] Saved: 04_confusing_pairs.png")


def save_report(stats, confusing_pairs, taxonomy_conf, taxonomy_data, output_dir):
    path = os.path.join(output_dir, "dataset_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("KHANA DATASET ANALYSIS REPORT\n")
        f.write("=" * 75 + "\n\n")

        f.write(f"Base Directory: {BASE_DIR}\n")
        f.write(f"Image Root: {DATA_ROOT}\n\n")

        f.write("--- FILE STRUCTURE ---\n")
        f.write(f"Total classes (folders): {stats['num_classes']}\n")
        f.write(f"Total images: {stats['total_images']}\n")
        f.write(f"Min class size: {stats['min_class_size']}\n")
        f.write(f"Max class size: {stats['max_class_size']}\n")
        f.write(f"Median class size: {stats['median_class_size']}\n")
        f.write(f"Mean class size: {stats['mean_class_size']}\n\n")

        if stats["dim_stats"]:
            ds = stats["dim_stats"]
            f.write("--- IMAGE DIMENSIONS ---\n")
            f.write(f"Sampled: {ds['count']} images\n")
            f.write(
                f"Width:  {ds['width_mean']:.1f} ± {ds['width_std']:.1f} (range: {ds['min_w']}-{ds['max_w']})\n"
            )
            f.write(
                f"Height: {ds['height_mean']:.1f} ± {ds['height_std']:.1f} (range: {ds['min_h']}-{ds['max_h']})\n"
            )
            f.write(f"Mean aspect ratio: {np.mean(ds['aspect_ratios']):.3f}\n")
            f.write(f"Color modes: {json.dumps(ds['mode_distribution'])}\n\n")

        f.write("--- ALL CLASS COUNTS (sorted by count) ---\n")
        for cls, cnt in stats["class_counts"].items():
            f.write(f"  {cls}: {cnt}\n")

        f.write("\n--- CONFUSING PAIRS (name-based) ---\n")
        for p in confusing_pairs[:30]:
            f.write(f"  [{p['reason']}] {p['classes']} -> {p['counts']}\n")

        if taxonomy_conf:
            f.write("\n--- TAXONOMY GROUPS ---\n")
            for t in taxonomy_conf[:20]:
                f.write(
                    f"  [{t['group_col']}={t['group_val']}] {t['classes']} -> {t['counts']}\n"
                )

        if taxonomy_data:
            f.write("\n--- TAXONOMY SAMPLE (first 10 rows) ---\n")
            for row in taxonomy_data[:10]:
                f.write(f"  {json.dumps(row)}\n")

    print(f"[INFO] Saved report: {path}")


def main():
    print("=" * 75)
    print("KHANA DATASET ANALYZER (NO-EXTENSION FIX)")
    print("=" * 75)

    class_folders = get_class_folders(DATA_ROOT)
    print(f"\n[INFO] Found {len(class_folders)} class folders")
    print(f"  First 5: {[n for n, _ in class_folders[:5]]}")
    print(f"  Last 5:  {[n for n, _ in class_folders[-5:]]}")

    labels_list = parse_labels(LABELS_PATH)
    taxonomy_data = parse_taxonomy(TAXONOMY_PATH)

    print("\n[INFO] Collecting images (no extension filter)...")
    class_images, all_images = collect_images(class_folders)
    print(f"  Total images found: {len(all_images)}")

    sample_cls = list(class_images.keys())[0]
    print(f"  Sample from '{sample_cls}': {class_images[sample_cls][:3]}")

    print("\n[INFO] Computing statistics...")
    stats = compute_stats(class_images)
    print(f"  Classes: {stats['num_classes']}")
    print(f"  Total images: {stats['total_images']}")
    print(
        f"  Min/Max/Median/Mean: {stats['min_class_size']} / {stats['max_class_size']} / {stats['median_class_size']} / {stats['mean_class_size']}"
    )

    print("\n[INFO] Finding confusing pairs...")
    cp, tc = find_confusing_pairs(stats["class_counts"], taxonomy_data)
    print(f"  Name-based: {len(cp)}, Taxonomy-based: {len(tc)}")

    print("\n[INFO] Generating outputs...")
    os.makedirs(OUTDIR, exist_ok=True)
    generate_plots(stats, cp, tc, OUTDIR)
    save_report(stats, cp, tc, taxonomy_data, OUTDIR)

    print("\n" + "=" * 75)
    print("DONE")
    print(f"Output: {OUTDIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
