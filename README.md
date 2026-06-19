I'll create a GitHub README for your Computer Vision Final Project based on the Khana Indian Cuisine Dataset. Let me first review the uploaded materials to understand the full scope of your project.
Now I have a comprehensive understanding of the project. Let me create a professional GitHub README for Karthik's project.

---

# Khana Indian Cuisine Computer Vision Final Project

> **Course:** Computer Vision Final Project  
> **Author:** Karthik  
> **Dataset:** [Khana](https://khana.omkar.xyz/) — A Comprehensive Indian Cuisine Dataset  
> **Paper:** [arXiv:2509.06006](https://arxiv.org/pdf/2509.06006)

## Overview

This repository contains my implementation for the Computer Vision Final Project using the **Khana** dataset — a large-scale food classification benchmark featuring 131,000+ images across **80 classes** of Indian cuisine. The project tackles three challenging tasks: fine-grained food classification, thali (multi-dish plate) detection with correct Khana labels, and bird's-eye view (BEV) transformation for natural-angle thali images.

The Khana dataset addresses a critical gap in food AI research: the severe underrepresentation of Indian cuisine in existing benchmarks, which predominantly focus on Western and East Asian dishes. Indian food presents unique challenges including high intra-class variation (different preparations, plating styles, lighting), high inter-class similarity (visually overlapping dishes like misal pav vs. pav bhaji), and complex regional diversity.

---

## Dataset

| Property | Value |
|----------|-------|
| **Total Images** | 131,000+ |
| **Classes** | 80 food dishes |
| **Resolution** | 500 × 500 pixels |
| **Split** | 70% Train / 15% Validation / 15% Test |
| **Super-classes** | Breakfast, Main Course, Snacks, Beverages, Desserts, Flatbread, Curry, South Indian |
| **Source** | Web search engines, Swiggy, Zomato |

The dataset exhibits a **long-tail distribution** — popular dishes like paneer and biryani have thousands of samples, while niche varieties like neer dosa and chikki have fewer. This imbalance makes the classification task particularly challenging and rewards robust data augmentation strategies.

---

## Tasks

### Task 1: Food Classification (80-Class)

**Objective:** Design and train a classifier to exceed the baseline validation accuracy of **~91%** on the 80 Khana food classes.

**Evaluation:** Models are scored on a held-out test set of 20–30 images. Performance is ranked on a leaderboard based on accuracy and macro F1-score.

**My Approach:** I used **ConvNeXT** as the backbone architecture, which proved to be the most effective model among the class participants. ConvNeXT's modernized pure convolutional design (inspired by Vision Transformers but without self-attention) provides an excellent balance of accuracy and efficiency for fine-grained food recognition.

**Key Results:**
- **Model:** ConvNeXT
- **Validation Accuracy:** 86.67% (see leaderboard context)
- **Macro F1:** 85.02
- **Final Score:** 86.5 (0.9 × accuracy + 0.1 × macro_f1)
- **Leaderboard Rank:** 🥇 **1st Place**
- **Error Rate:** 13.33% (16 mistakes out of 120 test images)

**Why ConvNeXT Works Here:** Unlike ResNet's residual blocks or EfficientNet's compound scaling, ConvNeXT employs depthwise separable convolutions and larger kernel sizes (7×7) that better capture the spatial texture patterns critical for distinguishing Indian dishes — grain textures in rice dishes, sauce consistency in curries, and bread char patterns.

---

### Task 2: Thali Detection with Correct Khana Labels

**Objective:** Detect individual dishes within a **thali** (traditional Indian meal plate) and label them with the correct **80 Khana classes** (not generic COCO labels like "bowl" or "dining table").

**The Challenge:** Standard object detectors (YOLO, Faster R-CNN) trained on COCO will output generic labels like "bowl," "cup," or "sandwich" — completely missing the specific Indian dish identity. The task requires mapping detection outputs to the fine-grained Khana taxonomy.

**My Method:**
1. **Detection:** Use an object detector to generate region proposals for each compartment/dish in the thali image
2. **Classification:** Crop each detected region and run the Task 1 ConvNeXT classifier to predict the specific Khana dish label
3. **Post-processing:** Apply confidence thresholding and non-maximum suppression to handle overlapping detections

**Evaluation Metrics:**
- Evaluated on **10 held-out thali images**
- Each thali contains exactly **3 dishes** that co-exist in the Khana dataset
- **Semantic grace** is applied for interchangeable dishes (e.g., `kheer` ↔ `phirni`, `pongal` ↔ `dal khichdi`, `vada pav` ↔ `pav bhaji` ↔ `misal pav` ↔ `dabeli`)

**Scoring:**
| Metric | Formula |
|--------|---------|
| `dish_slot_accuracy` | `(correct_dish_slots / 30) × 100` |
| `avg_plate_score` | Average of per-plate accuracy |
| `perfect_plate_accuracy` | `(perfect_plates / 10) × 100` |
| **Final Score** | `0.7 × avg_plate_score + 0.3 × perfect_plate_accuracy` |

A **perfect plate** requires all 3 dish slots to be correctly identified.

---

### Task 3: Bird's-Eye View (BEV) Transformation

**Objective:** Convert **natural-angle thali photographs** (taken from side angles, with perspective distortion) into clean bird's-eye view images suitable for Task 2 detection.

**The Problem:** Natural images suffer from:
- Perspective distortion (elliptical plates instead of circular)
- Occlusion (some dishes partially hidden behind others)
- Lighting variation and shadow casting
- Rotation and scale inconsistency

**Approach:** Implement homography-based perspective correction:
1. Detect the plate boundary (ellipse fitting in natural view → circle in BEV)
2. Compute the homography matrix mapping the detected ellipse to a canonical top-down view
3. Warp the image using the inverse perspective transform
4. Run Task 2 detection pipeline on the rectified BEV image

**Bonus (Task 4):** Neural Radiance Field (NRF) reconstruction using multi-view consistency for novel viewpoint synthesis.

---

## Project Structure

```
khana-cv-final-project/
├── task1_classification/
│   ├── train.py                 # ConvNeXT training script
│   ├── configs/
│   │   └── convnext_small.yaml  # Hyperparameters
│   ├── data/
│   │   ├── train/               # 70% Khana training images
│   │   ├── val/                 # 15% validation images
│   │   └── test/                # Held-out test set
│   └── models/
│       └── convnext_best.pth    # Best checkpoint (1st place)
│
├── task2_thali_detection/
│   ├── detector.py              # Object detection + Khana label mapping
│   ├── classify_regions.py      # Crop classifier using Task 1 model
│   └── evaluate_thali.py        # Precision/recall evaluation
│
├── task3_bev_transform/
│   ├── perspective_correction.py # Homography estimation
│   ├── ellipse_detector.py      # Plate boundary detection
│   └── bev_pipeline.py          # End-to-end BEV + detection
│
├── task4_nrf_bonus/             # (Optional) Neural Radiance Fields
│   └── nerf_reconstruction.py
│
├── utils/
│   ├── khana_taxonomy.py        # 80-class label mappings
│   ├── semantic_grace.py        # Interchangeable dish rules
│   └── metrics.py               # Evaluation utilities
│
└── README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/karthik/khana-cv-final-project.git
cd khana-cv-final-project

# Create environment
conda create -n khana python=3.10
conda activate khana

# Install dependencies
pip install torch torchvision timm opencv-python numpy pandas matplotlib seaborn scikit-learn
```

---

## Usage

### Task 1: Train Classification Model

```bash
cd task1_classification
python train.py \
  --model convnext_small \
  --data_dir ../data/khana \
  --epochs 50 \
  --batch_size 64 \
  --lr 0.001 \
  --output_dir ./checkpoints
```

### Task 2: Run Thali Detection

```bash
cd task2_thali_detection
python detector.py \
  --image path/to/thali.jpg \
  --classifier ../task1_classification/models/convnext_best.pth \
  --taxonomy ../utils/khana_taxonomy.csv \
  --output output_labeled.jpg
```

### Task 3: BEV Transformation + Detection

```bash
cd task3_bev_transform
python bev_pipeline.py \
  --input path/to/natural_angle_thali.jpg \
  --detector ../task2_thali_detection/detector.py \
  --output bev_result.jpg
```

---

## Results & Leaderboard

### Task 1: Food Classification Leaderboard

| Rank | Name | Model | Accuracy | Macro F1 | Final Score | Avg Time/img | # Mistakes | Error % | Performance % | Score |
|------|------|-------|----------|----------|-------------|--------------|------------|---------|---------------|-------|
| 🥇 **1** | **Karthik** | **ConvNeXT** | **86.67** | **85.02** | **86.5** | **0.1564** | **16** | **13.33** | **100%** | **6** |
| 2 | Veer | ConvNeXT | 85.83 | 79.69 | 85.22 | 0.3205 | 17 | 14.17 | 95% | 5.7 |
| 3 | Karttikeya | ConvNeXT | 85 | 80.29 | 84.53 | 0.1491 | 18 | 15 | 90% | 5.4 |
| 4 | Vaance | ConvNeXT | 85 | 78.14 | 84.31 | 1.6479 | 18 | 15 | 85% | 5.1 |
| 5 | Shreenand | ConvNeXT | 81.67 | 74.82 | 80.98 | 0.1936 | 22 | 18.33 | 80% | 4.8 |
| 6 | Sampurna | ConvNeXT | 80.83 | 72.55 | 80.01 | 3.4986 | 23 | 19.17 | 75% | 4.5 |
| 7 | Shubham | ResNet | 80 | 73.37 | 79.34 | 0.0386 | 24 | 20 | 70% | 4.2 |
| 8 | Parth | ResNet | 75.83 | 66.06 | 74.86 | 0.036 | 29 | 24.17 | 65% | 3.9 |
| 9 | Neha | EfficientNet | 71.67 | 58.9 | 70.39 | 0.0424 | 34 | 28.33 | 60% | 3.6 |

**Key Insight:** The top 6 performers all used **ConvNeXT**, confirming its superiority for this fine-grained food classification task. The gap between ConvNeXT and ResNet/EfficientNet is substantial (~6-16% accuracy), likely due to ConvNeXT's larger receptive fields and modernized architecture better capturing the textural nuances of Indian cuisine.

---

## Common Failure Modes & Analysis

Based on the mistaken classification analysis across all participants, the most challenging dish pairs include:

- **Misal Pav ↔ Pav Bhaji** — Both feature bread rolls with saucy vegetable toppings; distinguished by gravy consistency and garnish
- **Vada Pav ↔ Dabeli** — Similar bun-based street foods; vada pav has a fried potato patty while dabeli has sweet-spicy potato filling with pomegranate
- **South Indian Combo Dishes** — Idli, dosa, uttapam varieties often confused when presented in combo plates
- **Paneer Variants** — Paneer tikka, paneer butter masala, and shahi paneer share similar orange-red gravies

My model achieved the **lowest error rate (13.33%)** with only 16 mistakes out of 120 test images, suggesting strong generalization to these visually similar classes.

---

## References

```bibtex
@article{prabhu2025khana,
  title={Khana: A Comprehensive Indian Cuisine Dataset},
  author={Prabhu, Omkar},
  journal={arXiv preprint arXiv:2509.06006},
  year={2025}
}
```

**Dataset & Resources:**
- 📄 [Research Paper (arXiv)](https://arxiv.org/pdf/2509.06006)
- 💻 [Dataset GitHub](https://github.com/prabhuomkar/Khana)
- 🌐 [Dataset Website](https://khana.omkar.xyz/)
- 📦 [Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1PWyJdkizw5ABBd8BIAnr_FZq91YZ2Uo0)
- 🖼️ [Task 2 Sample Images](https://drive.google.com/drive/folders/1S9VTKefgB7FNGyOjGPvKdW-FP5fhwlHm)
- 🖼️ [Task 3 Sample Images](https://drive.google.com/drive/folders/1z5xnU_yvRM0G2bHfObaZm6S--OsTJf-a)

---

## License

This project is for academic and educational purposes only. The Khana dataset images are compiled from web sources and food delivery platforms; copyright remains with original image owners. See the [Khana repository](https://github.com/prabhuomkar/Khana) for full licensing terms.

---

## Contact

**Karthik** 
Project for Computer Vision Course — Final Project Submission

---

*Built with PyTorch, timm, and OpenCV. Ranked 1st on Task 1 Leaderboard.*
