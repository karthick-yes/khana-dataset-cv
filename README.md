# YOLO and SAM Detection Commands

This document lists commands to run detection using YOLO and SAM for single images and batches, including debug modes.

---

## YOLO — Single Image

```bash
uv run python task2_detect.py \
  --method yolo \
  --single_image "~/data/task2_images/Plate 146.jpg" \
  --output_dir ~/data/task2_yolo_single \
  --low_vram \
  --detector_max_side 960
```

---

## YOLO — Single Image (Debug)

```bash
uv run python task2_detect.py \
  --method yolo \
  --single_image "~/data/task2_images/Plate 146.jpg" \
  --output_dir ~/data/task2_yolo_debug \
  --debug \
  --low_vram \
  --detector_max_side 960
```

---

## YOLO — Batch

```bash
uv run python task2_detect.py \
  --method yolo \
  --input_dir ~/data/task2_images \
  --output_dir ~/data/task2_yolo_batch \
  --low_vram \
  --detector_max_side 960
```

---

## SAM — Single Image

```bash
uv run python task2_detect.py \
  --method sam \
  --single_image "~/data/task2_images/Plate 146.jpg" \
  --output_dir ~/data/task2_sam_single \
  --sam_checkpoint ~/data/sam_vit_b_01ec64.pth \
  --sam_points_per_side 16
```

---

## SAM — Single Image (Debug)

```bash
uv run python task2_detect.py \
  --method sam \
  --single_image "~/data/task2_images/Plate 146.jpg" \
  --output_dir ~/data/task2_sam_debug \
  --debug \
  --sam_checkpoint ~/data/sam_vit_b_01ec64.pth \
  --sam_points_per_side 16
```

---

## SAM — Batch (with YOLO fallback)

```bash
uv run python task2_detect.py \
  --method sam \
  --input_dir ~/data/task2_images \
  --output_dir ~/data/task2_sam_batch \
  --sam_checkpoint ~/data/sam_vit_b_01ec64.pth \
  --sam_points_per_side 16
```

---

## SAM — Strict (no fallback)

```bash
uv run python task2_detect.py \
  --method sam \
  --no_sam_fallback_yolo \
  --single_image "~/data/task2_images/Plate 146.jpg" \
  --output_dir ~/data/task2_sam_strict \
  --sam_checkpoint ~/data/sam_vit_b_01ec64.pth
```

---

## Full-quality runs (no low-VRAM adjustments)

Use these when you want to run without `--low_vram` and without detector downscaling.

### Task 2 YOLO (full resolution)

```bash
uv run python task2_detect.py \
  --method yolo \
  --single_image "~/data/task2_images/Plate 146.jpg" \
  --output_dir ~/data/task2_yolo_fullres_single
```

```bash
uv run python task2_detect.py \
  --method yolo \
  --single_image "~/data/task2_images/Plate 146.jpg" \
  --debug \
  --output_dir ~/data/task2_yolo_fullres_debug
```

```bash
uv run python task2_detect.py \
  --method yolo \
  --input_dir ~/data/task2_images \
  --output_dir ~/data/task2_yolo_fullres_batch
```

### Task 2 SAM (full settings)

```bash
uv run python task2_detect.py \
  --method sam \
  --single_image "~/data/task2_images/Plate 146.jpg" \
  --sam_checkpoint ~/data/sam_vit_b_01ec64.pth \
  --sam_points_per_side 16 \
  --output_dir ~/data/task2_sam_fullres_single
```

```bash
uv run python task2_detect.py \
  --method sam \
  --input_dir ~/data/task2_images \
  --sam_checkpoint ~/data/sam_vit_b_01ec64.pth \
  --sam_points_per_side 16 \
  --output_dir ~/data/task2_sam_fullres_batch
```

### Legacy SAM scripts (force-disable low-VRAM mode)

```bash
uv run python scripts/legacy/task2/task2_sam_grid.py \
  --image "~/data/task2_images/Plate 146.jpg" \
  --no_low_vram \
  --points_per_side 16 \
  --sam_device cuda
```

```bash
uv run python scripts/legacy/task2/sam_with_grid_filter.py \
  --image "~/data/task2_images/Plate 146.jpg" \
  --mode grid \
  --no_low_vram \
  --pts_per_side 16 \
  --sam_device cuda
```

```bash
uv run python scripts/legacy/task2/sam_inspector.py \
  --image "~/data/task2_images/Plate 146.jpg" \
  --no_low_vram \
  --pts_per_side 16 \
  --sam_device cuda
```

---

## Debug Output Files

The following debug artifacts may be generated in the specified output directory:

- `*_yolo_raw_boxes.jpg`
- `*_yolo_filtered_boxes.jpg`
- `*_sam_raw_segments.jpg`
- `*_sam_raw_boxes.jpg`
- `*_sam_filtered_boxes.jpg`
- `*_classified_pre_nms.jpg`
- `*_detected.jpg`

---

## Notes

- Use `--low_vram` for systems with limited GPU memory.
- Adjust `--detector_max_side` (e.g., 640 or lower) if memory issues occur.
- Legacy/experimental Task 2 scripts are now in `scripts/legacy/task2/`.

## Legacy SAM scripts (low-VRAM defaults enabled)

These scripts now default to low-VRAM behavior automatically.
You can disable it with `--no_low_vram`.

### `scripts/legacy/task2/task2_sam_grid.py`

```bash
uv run python scripts/legacy/task2/task2_sam_grid.py \
  --image "~/data/task2_images/Plate 146.jpg"
```

Optional controls:
- `--detector_max_side 960`
- `--sam_device auto|cuda|cpu`
- `--points_per_side 12`
- `--no_low_vram`

### `scripts/legacy/task2/sam_with_grid_filter.py`

```bash
uv run python scripts/legacy/task2/sam_with_grid_filter.py \
  --image "~/data/task2_images/Plate 146.jpg" \
  --mode grid
```

Optional controls:
- `--detector_max_side 960`
- `--sam_device auto|cuda|cpu`
- `--pts_per_side 12`
- `--no_low_vram`

---

## TF evaluation wrappers

Expected submission structure:

```text
name/
├── predict.py
├── best_model.pt (or your final checkpoint)
└── helper files/scripts
```

### Task 1 (`predict.py`)

- File added: `predict.py`
- Required API: `predict(image)` where `image` is a PIL image
- Return type: class name (`str`)
- Uses local checkpoint only (offline)
- Override checkpoint path with:
  - `TASK1_CHECKPOINT=/path/to/best_model.pt`

### Task 2 dictionary output

```bash
uv run python task2_predict_dict.py \
  --method yolo \
  --input_dir ~/data/task2_images \
  --output_json ~/data/task2_predictions_dict.json
```

Full-quality (no low-VRAM flags):

```bash
uv run python task2_predict_dict.py \
  --method yolo \
  --input_dir ~/data/task2_images \
  --output_json ~/data/task2_predictions_dict_fullres.json
```

Single image:

```bash
uv run python task2_predict_dict.py \
  --method sam \
  --single_image "~/data/task2_images/Plate 146.jpg" \
  --sam_checkpoint ~/data/sam_vit_b_01ec64.pth \
  --output_json ~/data/task2_one_dict.json
```

### Task 3 dictionary output

```bash
uv run python task3_predict_dict.py \
  --input_dir ~/data/task3_samples \
  --sam_checkpoint ~/data/sam_vit_b_01ec64.pth \
  --output_json ~/data/task3_predictions_dict.json
```

Single image:

```bash
uv run python task3_predict_dict.py \
  --single_image "~/data/task3_samples/thali_natural.jpg" \
  --sam_checkpoint ~/data/sam_vit_b_01ec64.pth \
  --output_json ~/data/task3_one_dict.json
```

Both Task 2/3 wrappers emit JSON in this exact format:

```json
{
  "thali_2": ["aloo_paratha", "poha", "dosa"],
  "thali_5": ["methi_paratha", "upma", "idli"]
}
```

### `scripts/legacy/task2/sam_inspector.py`

```bash
uv run python scripts/legacy/task2/sam_inspector.py \
  --image "~/data/task2_images/Plate 146.jpg"
```

Optional controls:
- `--detector_max_side 960`
- `--sam_device auto|cuda|cpu`
- `--pts_per_side 12`
- `--no_low_vram`
